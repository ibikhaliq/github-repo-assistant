from unittest.mock import MagicMock, patch


from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


@patch("main.index_queue")
def test_index_repository_queues_job(
    mock_index_queue,
    tmp_path,
):
    fake_job = MagicMock()
    fake_job.id = "test-job-123"

    mock_index_queue.enqueue.return_value = fake_job

    with patch(
        "main.REPOSITORIES_DIR",
        tmp_path,
    ):
        repository_path = (
            tmp_path
            / "ibikhaliq__github-repo-assistant"
        )

        repository_path.mkdir()

        response = client.post(
            "/repos/ibikhaliq/github-repo-assistant/index"
        )

    assert response.status_code == 200

    assert response.json() == {
        "status": "queued",
        "job_id": "test-job-123",
        "repository": (
            "ibikhaliq/github-repo-assistant"
        ),
    }

    mock_index_queue.enqueue.assert_called_once()


def test_index_repository_missing_repo():
    response = client.post(
        "/repos/nonexistent/repository/index"
    )

    assert response.status_code == 404
    



@patch("main.Job.fetch")
def test_get_finished_job_status(
    mock_job_fetch,
):
    fake_job = MagicMock()
    fake_job.id = "job-123"
    fake_job.get_status.return_value.value = "finished"
    fake_job.is_finished = True
    fake_job.is_failed = False
    fake_job.result = {
        "status": "success",
        "source": "cache",
    }

    mock_job_fetch.return_value = fake_job

    response = client.get(
        "/jobs/job-123"
    )

    assert response.status_code == 200

    assert response.json() == {
        "job_id": "job-123",
        "status": "finished",
        "result": {
            "status": "success",
            "source": "cache",
        },
    }


@patch("main.Job.fetch")
def test_get_missing_job_returns_404(
    mock_job_fetch,
):
    mock_job_fetch.side_effect = Exception(
        "missing job"
    )

    response = client.get(
        "/jobs/does-not-exist"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }