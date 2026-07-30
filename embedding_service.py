import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1",
)

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL"
)


if not OLLAMA_EMBEDDING_MODEL:
    raise ValueError(
        "OLLAMA_EMBEDDING_MODEL is missing "
        "from the .env file."
    )


embedding_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)


def create_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Create embeddings for multiple text inputs.
    """

    if not texts:
        return []

    cleaned_texts = [
        text.strip()
        for text in texts
    ]

    if any(
        not text
        for text in cleaned_texts
    ):
        raise ValueError(
            "Cannot create embeddings for empty text."
        )
    
    print(f"Creating {len(cleaned_texts)} embeddings in one request...")

    response = embedding_client.embeddings.create(
        model=OLLAMA_EMBEDDING_MODEL,
        input=cleaned_texts,
    )

    sorted_data = sorted(
        response.data,
        key=lambda item: item.index,
    )

    embeddings = [
        item.embedding
        for item in sorted_data
    ]

    if len(embeddings) != len(cleaned_texts):
        raise ValueError(
            "The embedding service returned an "
            "unexpected number of embeddings."
        )

    return embeddings


def create_embedding(
    text: str,
) -> list[float]:
    """
    Create one embedding while reusing the batch function.
    """

    embeddings = create_embeddings(
        [text]
    )

    return embeddings[0]