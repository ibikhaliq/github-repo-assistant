from pathlib import Path


MAX_FILE_SIZE_BYTES = 100_000
MAX_TOTAL_CONTENT_BYTES = 500_000

IGNORED_DIRECTORIES = {
    ".git",
    ".repo_assistant",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

SUPPORTED_FILE_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".md",
    ".json",
    ".txt",
    ".yml",
    ".yaml",
    ".toml",
}


def get_repository_files(repository_path: Path) -> list[str]:
    repository_files = []

    for path in repository_path.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(repository_path)

        if any(part in IGNORED_DIRECTORIES for part in relative_path.parts):
            continue

        if path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
            continue

        repository_files.append(str(relative_path))

    return sorted(repository_files)


def read_repository_files(repository_path: Path) -> list[dict]:
    repository_files = get_repository_files(repository_path)

    file_contents = []
    total_content_size = 0

    for relative_file_path in repository_files:
        full_file_path = repository_path / relative_file_path
        file_size = full_file_path.stat().st_size

        if file_size > MAX_FILE_SIZE_BYTES:
            continue

        if total_content_size + file_size > MAX_TOTAL_CONTENT_BYTES:
            break

        try:
            content = full_file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        file_contents.append(
            {
                "path": relative_file_path,
                "size_bytes": file_size,
                "content": content,
            }
        )

        total_content_size += file_size

    return file_contents