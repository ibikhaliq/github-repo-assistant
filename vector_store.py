import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


from embedding_service import create_embeddings



def format_field(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(
            f"- {item}"
            for item in value
        )
    return str(value)


def build_embedding_text(
    file_summary: dict[str, Any],
) -> str:
    """
    Build the text that will be embedded for semantic search.
    """

    file_path = file_summary.get(
        "file_path",
        ""
    )

    summary = file_summary.get(
        "summary",
        {}
    )

    purpose = summary.get(
        "purpose",
        ""
    )

    important_components = format_field(
        summary.get(
            "important_components",
            []
        )
    )

    dependencies = format_field(
        summary.get(
            "dependencies",
            []
        )
    )

    role_in_project = summary.get(
        "role_in_project",
        ""
    )

    potential_issues = format_field(
        summary.get(
            "potential_issues",
            []
        )
    )

    return (
        f"File path: {file_path}\n\n"
        f"Purpose:\n{purpose}\n\n"
        f"Important components:\n"
        f"{important_components}\n\n"
        f"Dependencies:\n"
        f"{dependencies}\n\n"
        f"Role in project:\n"
        f"{role_in_project}\n\n"
        f"Potential issues:\n"
        f"{potential_issues}"
    )

EMBEDDING_BATCH_SIZE = 16


def create_repository_embeddings(
    repository_data: dict[str, Any],
) -> list[dict[str, Any]]:
    file_summaries = repository_data.get(
        "file_summaries",
        [],
    )

    if not file_summaries:
        return []

    embedding_texts = [
        build_embedding_text(file_summary)
        for file_summary in file_summaries
    ]

    all_embeddings = []

    for batch_start in range(
        0,
        len(embedding_texts),
        EMBEDDING_BATCH_SIZE,
    ):
        batch_texts = embedding_texts[
            batch_start:
            batch_start + EMBEDDING_BATCH_SIZE
        ]

        batch_embeddings = create_embeddings(
            batch_texts
        )

        all_embeddings.extend(
            batch_embeddings
        )

    if len(all_embeddings) != len(file_summaries):
        raise ValueError(
            "The number of generated embeddings does not "
            "match the number of repository files."
        )

    embedding_records = []

    for (
        file_summary,
        embedding_text,
        embedding,
    ) in zip(
        file_summaries,
        embedding_texts,
        all_embeddings,
    ):
        embedding_records.append(
            {
                "file_path": file_summary.get(
                    "file_path",
                    "",
                ),
                "embedding_text": embedding_text,
                "embedding": embedding,
            }
        )

    return embedding_records

def get_embeddings_file_path(
    repository_path: Path,
) -> Path:
    """
    Return the path where repository embeddings are stored.
    """
    metadata_directory = (
        repository_path / ".repo_assistant"
    )

    return metadata_directory / "embeddings.json"


def save_repository_embeddings(
    repository_path: Path,
    repository_name: str,
    embedding_records: list[dict[str, Any]],
) -> Path:
    """
    Save repository file embeddings to embeddings.json.
    """
    embeddings_file_path = get_embeddings_file_path(
        repository_path
    )

    embeddings_file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_data = {
        "repository": repository_name,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "files_embedded": len(embedding_records),
        "embedding_dimensions": (
            len(embedding_records[0]["embedding"])
            if embedding_records
            else 0
        ),
        "embeddings": embedding_records,
    }

    embeddings_file_path.write_text(
        json.dumps(
            embedding_data,
            indent=2,
        ),
        encoding="utf-8",
    )

    return embeddings_file_path


def load_repository_embeddings(
    repository_path: Path,
) -> dict[str, Any]:
    """
    Load saved repository embeddings.
    """
    embeddings_file_path = get_embeddings_file_path(
        repository_path
    )

    if not embeddings_file_path.exists():
        raise FileNotFoundError(
            "No saved repository embeddings were found."
        )

    try:
        return json.loads(
            embeddings_file_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "The saved repository embeddings contain "
            "invalid JSON."
        ) from error


def repository_embeddings_exist(
    repository_path: Path,
) -> bool:
    """
    Check whether embeddings.json exists.
    """
    return get_embeddings_file_path(
        repository_path
    ).exists()