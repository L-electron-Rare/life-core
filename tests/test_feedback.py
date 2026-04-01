"""Tests for the /feedback endpoint."""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


def _get_app():
    """Import app fresh to avoid lifespan issues."""
    from life_core.api import app
    return app


def test_feedback_success():
    with patch("life_core.langfuse_tracing.score_trace") as mock_score:
        client = TestClient(_get_app(), raise_server_exceptions=False)
        response = client.post("/feedback", json={
            "trace_id": "a" * 32,
            "score": 0.8,
            "comment": "Good response",
        })

    assert response.status_code == 200
    mock_score.assert_called_once_with(
        trace_id="a" * 32,
        name="user-feedback",
        value=0.8,
        comment="Good response",
    )


def test_feedback_without_comment():
    with patch("life_core.langfuse_tracing.score_trace"):
        client = TestClient(_get_app(), raise_server_exceptions=False)
        response = client.post("/feedback", json={
            "trace_id": "b" * 32,
            "score": 0.5,
        })

    assert response.status_code == 200


def test_feedback_invalid_score():
    client = TestClient(_get_app(), raise_server_exceptions=False)
    response = client.post("/feedback", json={
        "trace_id": "c" * 32,
        "score": 1.5,
    })
    assert response.status_code == 422
