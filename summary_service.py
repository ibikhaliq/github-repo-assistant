import json

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_service import (
    summarize_file_with_ai,
    summarize_repository_with_ai,
)
from repository_service import (
    MAX_FILE_SIZE_BYTES,
    get_repository_files,
)

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

def summarize_single_repository_file(
    repository_path: Path,
    file_path: Path,
) -> dict | None:
    """
    Read and summarize one repository file.

    Returns None when the file should be skipped.
    """

    full_path = repository_path / file_path

    if full_path.stat().st_size > MAX_FILE_SIZE_BYTES:
        return None

    try:
        file_content = full_path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        return None

    try:
        summary = summarize_file_with_ai(
            file_path=str(file_path),
            file_content=file_content,
        )
    except ValueError:
        return None

    return {
        "file_path": str(file_path),
        "summary": summary.model_dump(),
    }

def summarize_repository_service(
    owner: str,
    repo_name: str,
    repositories_dir: Path,
) -> dict:
    """
    Summarize all supported files in a repository
    and save the resulting repository summary.
    """

    repository_path = (
        repositories_dir / f"{owner}__{repo_name}"
    )

    if not repository_path.exists():
        raise FileNotFoundError(
            "Repository has not been cloned."
        )

    repository_files = get_repository_files(
        repository_path
    )

    file_summaries = []
    total_files = len(repository_files)

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:
        future_to_file = {
            executor.submit(
                summarize_single_repository_file,
                repository_path,
                file_path,
            ): file_path
            for file_path in repository_files
        }

        completed_files = 0

        for future in as_completed(
            future_to_file
        ):
            file_path = future_to_file[future]

            try:
                file_summary = future.result()

                if file_summary is not None:
                    file_summaries.append(
                        file_summary
                    )

                completed_files += 1

                print(
                    f"Processed {completed_files}/"
                    f"{total_files}: {file_path}"
                )

            except OSError as error:
                print(
                    f"Could not read {file_path}: "
                    f"{error}"
                )

    if not file_summaries:
        raise ValueError(
            "No files could be summarized."
        )

    repository_summary = summarize_repository_with_ai(
        repository_name=f"{owner}/{repo_name}",
        file_summaries=file_summaries,
    )

    summary_data = {
        "repository": f"{owner}/{repo_name}",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "files_analyzed": len(file_summaries),
        "file_summaries": file_summaries,
        "repository_summary": (
            repository_summary.model_dump()
        ),
    }

    summary_file_path = save_repository_summaries(
        repository_path=repository_path,
        summary_data=summary_data,
    )

    return {
        **summary_data,
        "saved_to": str(summary_file_path),
    }