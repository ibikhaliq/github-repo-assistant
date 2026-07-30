from math import sqrt
from typing import Any

from embedding_service import create_embedding


def calculate_cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.

    A result closer to 1 means the vectors are more similar.
    A result closer to 0 means they are less similar.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Embedding vectors must have the same dimensions."
        )

    dot_product = sum(
        value_a * value_b
        for value_a, value_b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = sqrt(
        sum(value ** 2 for value in vector_a)
    )

    magnitude_b = sqrt(
        sum(value ** 2 for value in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


def retrieve_semantic_matches(
    question: str,
    embedding_records: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Embed the user's question and rank repository files
    by semantic similarity.
    """

    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    question_embedding = create_embedding(
        question
    )

    ranked_matches = []

    for record in embedding_records:
        file_embedding = record.get(
            "embedding"
        )

        if not file_embedding:
            continue

        similarity_score = (
            calculate_cosine_similarity(
                question_embedding,
                file_embedding,
            )
        )

        ranked_matches.append(
            {
                "file_path": record.get(
                    "file_path",
                    ""
                ),
                "embedding_text": record.get(
                    "embedding_text",
                    ""
                ),
                "similarity_score": similarity_score,
            }
        )

    ranked_matches.sort(
        key=lambda match: match[
            "similarity_score"
        ],
        reverse=True,
    )

    return ranked_matches[:top_k]

def get_semantic_file_paths(
    matches: list[dict[str, Any]],
) -> list[str]:
    """
    Extract file paths from semantic retrieval results.
    """

    return [
        match["file_path"]
        for match in matches
        if match.get("file_path")
    ]