import pytest
import numpy as np
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


@pytest.fixture
def sample_texts():
    return [
        "AI in medicine",
        "Latest NLP trends",
        "quantum computing applications",
        "genome sequencing analysis",
        "climate change impact on agriculture"
    ]


def test_embeddings_shape(sample_texts):
    """Ensure embeddings are returned for all texts and are vectors."""
    result = [
        np.array(e.values) for e in client.models.embed_content(
            model="gemini-embedding-001",
            contents=sample_texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        ).embeddings
    ]
    assert len(result) == len(sample_texts)
    for emb in result:
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] > 0   # non-empty vector


def test_cosine_similarity_valid_range(sample_texts):
    """Check cosine similarity values are within [-1, 1] with tolerance."""
    result = [
        np.array(e.values) for e in client.models.embed_content(
            model="gemini-embedding-001",
            contents=sample_texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        ).embeddings
    ]
    embeddings_matrix = np.array(result)
    similarity_matrix = cosine_similarity(embeddings_matrix)

    # allow tolerance for floating point rounding errors
    assert np.all(similarity_matrix <= 1.0 + 1e-6)
    assert np.all(similarity_matrix >= -1.0 - 1e-6)


def test_semantic_sanity(sample_texts):
    """Check that similar texts are more related than unrelated ones."""
    texts = ["AI in medicine", "Latest NLP trends", "baking a cake"]
    result = [
        np.array(e.values) for e in client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        ).embeddings
    ]

    embeddings_matrix = np.array(result)
    similarity_matrix = cosine_similarity(embeddings_matrix)

    sim_ai_nlp = similarity_matrix[0, 1]  # AI vs NLP
    sim_ai_cake = similarity_matrix[0, 2] # AI vs baking

    assert sim_ai_nlp > sim_ai_cake, "Expected AI/NLP similarity > AI/baking"
