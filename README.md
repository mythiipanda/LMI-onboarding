Notes:
I modified existing code to take in .txt files, preprocess them, and convert and save them to a ChromaDB vectorstore. I'd like to acknowledge that ChatGPT can already do this with their search function, but LLMs cannot search for recent information without using tools. This uses llama3.1-8b, and I am using 4o-mini for inference.

Pipeline: Langchain for Documents, Huggingface for embeddings, ChromaDB for local vectorstore, and 4o-mini for inference/LLM. I copied some text from the most recent NFL rankings into a .txt file. The program preprocesses that text for nicer chunks and saves the embeddings to ChromaDB. It then uses cosine similarity to retrieve the top k (default: 3) documents and feeds that to the LLM.

This took around an hour since I already had an existing implementation and modified some parts. 