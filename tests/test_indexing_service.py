from unittest.mock import patch

from indexing_service import index_repository_service


def test_index_repository_uses_cache(
    tmp_path,
):
    repository_path = (
        tmp_path
        / "ibrahim__repo"
    )

    repository_path.mkdir()

    summaries_path = (
        repository_path
        / ".repo_assistant"
        / "summaries.json"
    )

    embeddings_path = (
        repository_path
        / ".repo_assistant"
        / "embeddings.json"
    )

    summaries_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries_path.write_text(
        "{}",
        encoding="utf-8",
    )

    embeddings_path.write_text(
        "{}",
        encoding="utf-8",
    )

    fake_summary_data = {
        "file_summaries": [
            {"file_path": "main.py"},
            {"file_path": "models.py"},
        ]
    }

    fake_embedding_data = {
        "files_embedded": 2,
        "embedding_dimensions": 768,
    }

    with (
        patch(
            "indexing_service.get_current_commit_hash",
            return_value="abc123",
        ),
        patch(
            "indexing_service.load_index_metadata",
            return_value={
                "commit_hash": "abc123"
            },
        ),
        patch(
            "indexing_service.load_repository_summaries",
            return_value=fake_summary_data,
        ),
        patch(
            "indexing_service.load_repository_embeddings",
            return_value=fake_embedding_data,
        ),
    ):
        result = index_repository_service(
            owner="ibrahim",
            repo_name="repo",
            repositories_dir=tmp_path,
            force=False,
        )

    assert result["status"] == "success"
    assert result["source"] == "cache"
    assert result["files_analyzed"] == 2
    assert result["files_embedded"] == 2
    assert result["embedding_dimensions"] == 768