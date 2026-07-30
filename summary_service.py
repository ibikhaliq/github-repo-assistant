import json
from pathlib import Path
from typing import Any


SUMMARY_DIRECTORY_NAME = ".repo_assistant"
SUMMARY_FILE_NAME = "summaries.json"


def get_summary_file_path(repository_path: Path) -> Path:
    summary_directory = repository_path / SUMMARY_DIRECTORY_NAME

    return summary_directory / SUMMARY_FILE_NAME


def save_repository_summaries(
    repository_path: Path,
    summary_data: dict[str, Any],
) -> Path:
    summary_file_path = get_summary_file_path(repository_path)

    summary_file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file_path.write_text(
        json.dumps(summary_data, indent=2),
        encoding="utf-8",
    )

    return summary_file_path


def load_repository_summaries(
    repository_path: Path,
) -> dict[str, Any]:
    summary_file_path = get_summary_file_path(repository_path)

    if not summary_file_path.exists():
        raise FileNotFoundError(
            "No saved repository summary was found."
        )

    try:
        return json.loads(
            summary_file_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "The saved repository summary contains invalid JSON."
        ) from error


def repository_summary_exists(repository_path: Path) -> bool:
    summary_file_path = get_summary_file_path(repository_path)

    return summary_file_path.exists()
