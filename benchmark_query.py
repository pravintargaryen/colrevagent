import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
import os
from persist import ask_llm

# Initialize the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed_texts(texts: list[str], model: str = "gemini-embedding-001", task_type="SEMANTIC_SIMILARITY"):
    """Embed a list of texts using Gemini embedding model."""
    response = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    # Each embedding is returned with .values
    return [np.array(e.values) for e in response.embeddings]

def benchmark_query_with_gemini(query: str, results: list[str]):
    texts = [query] + results
    embeddings = embed_texts(texts)
    q_emb, r_embs = embeddings[0], embeddings[1:]
    
    sims = cosine_similarity([q_emb], r_embs)[0]
    print(f"\nQuery: {query}")
    print(f"Top-1 similarity: {sims[0]:.4f}")
    print(f"Average similarity: {np.mean(sims):.4f}")
    print("Similarities:", [round(s, 3) for s in sims])

def run_benchmark_demo():
    queries = [
        "AI in medicine",
        "quantum computing applications"
    ]
    for query in queries:
        results = ask_llm(query)  # returns list of titles or summaries
        benchmark_query_with_gemini(query, results)