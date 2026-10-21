
import json
import subprocess

from datetime import datetime, timezone
from pathlib import Path
from summary_service import (
    load_repository_summaries,
    summarize_repository_service,
)
from vector_store import (
    create_repository_embeddings,
    load_repository_embeddings,
    save_repository_embeddings,
)

def get_current_commit_hash(
    repository_path: Path,
) -> str:
    """
    Return the current Git commit hash for a repository.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_path),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Could not determine the repository commit hash."
        ) from error

    commit_hash = result.stdout.strip()

    if not commit_hash:
        raise RuntimeError(
            "Git returned an empty commit hash."
        )

    return commit_hash


def get_index_metadata_path(
    repository_path: Path,
) -> Path:
    return (
        repository_path
        / ".repo_assistant"
        / "index_metadata.json"
    )


def load_index_metadata(
    repository_path: Path,
) -> dict:
    metadata_path = get_index_metadata_path(
        repository_path
    )

    if not metadata_path.exists():
        return {}

    try:
        return json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError:
        return {}


def save_index_metadata(
    repository_path: Path,
    repository_name: str,
    commit_hash: str,
) -> Path:
    metadata_path = get_index_metadata_path(
        repository_path
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "repository": repository_name,
        "commit_hash": commit_hash,
        "indexed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata_path

def index_repository_service(
    owner: str,
    repo_name: str,
    repositories_dir: Path,
    force: bool = False,
) -> dict:
    """
    Index a repository or reuse its saved index when
    the repository has not changed.
    """

    repository_name = f"{owner}/{repo_name}"

    repository_path = (
        repositories_dir / f"{owner}__{repo_name}"
    )

    if not repository_path.exists():
        raise FileNotFoundError(
            "Repository not found. Clone it first."
        )

    current_commit_hash = get_current_commit_hash(
        repository_path
    )

    summaries_file_path = (
        repository_path
        / ".repo_assistant"
        / "summaries.json"
    )

    embeddings_file_path = (
        repository_path
        / ".repo_assistant"
        / "embeddings.json"
    )

    metadata = load_index_metadata(
        repository_path
    )

    saved_commit_hash = metadata.get(
        "commit_hash"
    )

    cache_is_valid = (
        not force
        and saved_commit_hash
        == current_commit_hash
        and summaries_file_path.exists()
        and embeddings_file_path.exists()
    )

    print("\n----- Cache Debug -----")
    print("Current commit:", current_commit_hash)
    print("Saved commit:  ", saved_commit_hash)
    print(
        "Summaries exist:",
        summaries_file_path.exists(),
    )
    print(
        "Embeddings exist:",
        embeddings_file_path.exists(),
    )
    print("Cache valid:", cache_is_valid)
    print("-----------------------\n")

    if cache_is_valid:
        repository_data = load_repository_summaries(
            repository_path
        )

        embedding_data = load_repository_embeddings(
            repository_path
        )

        return {
            "status": "success",
            "source": "cache",
            "repository": repository_name,
            "commit_hash": current_commit_hash,
            "files_analyzed": len(
                repository_data.get(
                    "file_summaries",
                    [],
                )
            ),
            "files_embedded": embedding_data.get(
                "files_embedded",
                0,
            ),
            "embedding_dimensions": (
                embedding_data.get(
                    "embedding_dimensions",
                    0,
                )
            ),
            "summary_file": str(
                summaries_file_path
            ),
            "embeddings_file": str(
                embeddings_file_path
            ),
            "message": (
                "The repository has not changed. "
                "The saved index was reused."
            ),
        }

    print("Generating new summaries...")

    summary_data = summarize_repository_service(
        owner=owner,
        repo_name=repo_name,
        repositories_dir=repositories_dir,
    )

    repository_data = load_repository_summaries(
        repository_path
    )

    embedding_records = (
        create_repository_embeddings(
            repository_data=repository_data,
        )
    )

    embeddings_file_path = (
        save_repository_embeddings(
            repository_path=repository_path,
            repository_name=repository_name,
            embedding_records=embedding_records,
        )
    )

    metadata_path = save_index_metadata(
        repository_path=repository_path,
        repository_name=repository_name,
        commit_hash=current_commit_hash,
    )

    return {
        "status": "success",
        "source": "generated",
        "repository": repository_name,
        "commit_hash": current_commit_hash,
        "files_analyzed": summary_data[
            "files_analyzed"
        ],
        "files_embedded": len(
            embedding_records
        ),
        "embedding_dimensions": (
            len(
                embedding_records[0]["embedding"]
            )
            if embedding_records
            else 0
        ),
        "summary_file": summary_data[
            "saved_to"
        ],
        "embeddings_file": str(
            embeddings_file_path
        ),
        "metadata_file": str(
            metadata_path
        ),
        "message": (
            "Repository indexed successfully. "
            "It is now ready for semantic search "
            "and question answering."
        ),
    }