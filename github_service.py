from pathlib import Path
from urllib.parse import urlparse

from git import Repo


REPOSITORIES_DIR = Path("repositories")
REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_github_repo(repo_url: str) -> bool:
    parsed_url = urlparse(repo_url)

    if parsed_url.scheme != "https":
        return False

    if parsed_url.netloc != "github.com":
        return False

    path_parts = parsed_url.path.strip("/").split("/")

    if len(path_parts) != 2:
        return False

    owner, repo_name = path_parts

    if not owner or not repo_name:
        return False

    return True


def clone_repository(repo_url: str) -> dict:
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip("/").split("/")

    owner = path_parts[0]
    repo_name = path_parts[1]

    destination_path = (
        REPOSITORIES_DIR
        / f"{owner}__{repo_name}"
    )

    # ----------------------------------
    # Repository already exists locally
    # ----------------------------------
    if destination_path.exists():
        repository = Repo(destination_path)

        previous_commit = repository.head.commit.hexsha

        # Download and apply the latest
        # changes from GitHub.
        repository.remotes.origin.pull()

        current_commit = repository.head.commit.hexsha

        return {
            "owner": owner,
            "repo_name": repo_name,
            "path": str(destination_path),
            "already_exists": True,
            "updated": previous_commit != current_commit,
            "previous_commit": previous_commit,
            "current_commit": current_commit,
        }

    # ----------------------------------
    # Repository is not stored locally
    # ----------------------------------
    repository = Repo.clone_from(
        repo_url,
        destination_path,
    )

    current_commit = repository.head.commit.hexsha

    return {
        "owner": owner,
        "repo_name": repo_name,
        "path": str(destination_path),
        "already_exists": False,
        "updated": True,
        "previous_commit": None,
        "current_commit": current_commit,
    }