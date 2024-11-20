import os
import re
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from dotenv import load_dotenv
from chromadb import Client
from chromadb.config import Settings
from cerebras.cloud.sdk import Cerebras

class NFLRankingsProcessor:
    def __init__(self, embedding_model_name: str, cerebras_client):
        self.embeddings = FastEmbedEmbeddings(model_name=embedding_model_name)
        self.chroma_client = Client(Settings())
        self.collection = self.chroma_client.create_collection(name="nfl_rankings")
        self.cerebras_client = cerebras_client

    def clean_text(self, text: str) -> str:
        return ' '.join(text.strip().split())

    def scrape_rankings(self, input_file: str, week: int) -> list:
        with open(input_file, 'r', encoding='utf-8') as file:
            text_content = file.read()

        matches = re.findall(
            r'Rank\s+(\d+)\s+([\w\s]+)\s+([\w\s]+)\s+\d+-\d+\s+(.*?)(?=Rank\s+\d+|$)', 
            text_content, 
            re.DOTALL
        )
        rankings = []
        for rank, team_name, team_info, details in matches:
            cleaned_details = self.clean_text(details)
            rankings.append({
                "rank": int(rank),
                "week": week,
                "content": f"Week {week} of 2024, Rank {rank}: {team_name} - {team_info}\n{cleaned_details}"
            })
        return rankings

    def save_to_chroma(self, rankings: list) -> None:
        for ranking in rankings:
            content = ranking["content"]
            vector = self.embeddings.embed_query(content)
            self.collection.add(
                documents=[content],
                metadatas=[{"rank": ranking["rank"], "week": ranking["week"]}],
                ids=[f"{ranking['week']}_{ranking['rank']}"],
                embeddings=[vector]
            )

    def similarity_search(self, query: str, k: int = 3) -> list:
        query_vector = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        return [Document(page_content=doc, metadata={"week": meta["week"], "rank": meta["rank"]}) 
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

    def get_answer(self, question: str) -> str:
        similar_docs = self.similarity_search(question)
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                              for i, doc in enumerate(similar_docs)])
        print("Context:", context)
        prompt = f"""Based on the following NFL Power Rankings context, please answer this question: {question}

Context:
{context}

Answer:"""
        response = self.cerebras_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3.1-8b"
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("CEREBRAS_API_KEY")

    if not api_key:
        print("Error: Missing required environment variable (CEREBRAS_API_KEY)")
    else:
        cerebras_client = Cerebras(api_key=api_key)
        processor = NFLRankingsProcessor(
            embedding_model_name="BAAI/bge-large-en-v1.5",
            cerebras_client=cerebras_client
        )
        text_files = ['week10.txt', 'week11.txt', 'week12.txt']
        for file_path in text_files:
            week = int(re.search(r'week(\d+)', file_path, re.IGNORECASE).group(1))
            rankings = processor.scrape_rankings(file_path, week)
            processor.save_to_chroma(rankings)
            print(f"Saved Week {week} rankings to ChromaDB.")
        question = "What is the Lions' power ranking in week 10 of 2024?"
        answer = processor.get_answer(question)
        print("\nQuestion:", question)
        print("\nAnswer:", answer)