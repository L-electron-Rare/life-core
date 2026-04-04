"""Extended tests for logs_api — BufferHandler, source filtering, limit behavior."""

from __future__ import annotations

import logging
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import life_core.logs_api as logs_module
from life_core.logs_api import logs_router, add_log, BufferHandler


@pytest.fixture(autouse=True)
def reset_logs():
    logs_module._log_buffer.clear()
    yield
    logs_module._log_buffer.clear()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(logs_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# add_log helper
# ---------------------------------------------------------------------------


def test_add_log_appends_entry():
    add_log("INFO", "test message")
    assert len(logs_module._log_buffer) == 1


def test_add_log_stores_level():
    add_log("ERROR", "error msg")
    assert logs_module._log_buffer[-1]["level"] == "ERROR"


def test_add_log_stores_message():
    add_log("WARN", "warn msg")
    assert logs_module._log_buffer[-1]["message"] == "warn msg"


def test_add_log_stores_source():
    add_log("INFO", "msg", source="custom-source")
    assert logs_module._log_buffer[-1]["source"] == "custom-source"


def test_add_log_default_source_is_life_core():
    add_log("INFO", "msg")
    assert logs_module._log_buffer[-1]["source"] == "life-core"


def test_add_log_stores_timestamp():
    add_log("INFO", "msg")
    entry = logs_module._log_buffer[-1]
    assert "timestamp" in entry
    assert isinstance(entry["timestamp"], str)


# ---------------------------------------------------------------------------
# BufferHandler
# ---------------------------------------------------------------------------


def test_buffer_handler_captures_life_core_logs():
    handler = BufferHandler()
    record = logging.LogRecord(
        name="life_core.services",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="test log",
        args=(),
        exc_info=None,
    )
    initial_count = len(logs_module._log_buffer)
    handler.emit(record)
    assert len(logs_module._log_buffer) == initial_count + 1


def test_buffer_handler_ignores_non_life_core_logs():
    handler = BufferHandler()
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="GET /health 200",
        args=(),
        exc_info=None,
    )
    initial_count = len(logs_module._log_buffer)
    handler.emit(record)
    assert len(logs_module._log_buffer) == initial_count


def test_buffer_handler_stores_correct_level():
    handler = BufferHandler()
    record = logging.LogRecord(
        name="life_core.router",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="router warning",
        args=(),
        exc_info=None,
    )
    handler.emit(record)
    last = logs_module._log_buffer[-1]
    assert last["level"] == "WARNING"
    assert last["message"] == "router warning"


# ---------------------------------------------------------------------------
# /logs/recent endpoint
# ---------------------------------------------------------------------------


def test_recent_logs_limit_respected(client):
    for i in range(20):
        add_log("INFO", f"message {i}")
    response = client.get("/logs/recent?limit=5")
    data = response.json()
    assert len(data["logs"]) == 5


def test_recent_logs_total_reflects_full_buffer(client):
    for i in range(10):
        add_log("INFO", f"message {i}")
    response = client.get("/logs/recent?limit=3")
    data = response.json()
    assert data["total"] == 10
    assert len(data["logs"]) == 3


def test_recent_logs_returns_latest_entries(client):
    for i in range(5):
        add_log("INFO", f"message {i}")
    response = client.get("/logs/recent?limit=2")
    data = response.json()
    messages = [l["message"] for l in data["logs"]]
    # last 2 should be message 3 and message 4
    assert "message 4" in messages
    assert "message 3" in messages


def test_recent_logs_default_limit(client):
    for i in range(60):
        add_log("INFO", f"msg {i}")
    response = client.get("/logs/recent")
    data = response.json()
    assert len(data["logs"]) <= 50  # default limit=50
