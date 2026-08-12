"""Tests for API-level video question quota handling."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import app


client = TestClient(app.app)


def reset_api_state() -> None:
    app.video_stores.clear()
    app.video_question_counts.clear()
    app.active_video_id = None


def set_active_video(video_id: str = "abcdefghijk") -> None:
    app.active_video_id = video_id
    app.video_stores[video_id] = Mock()
    app.video_question_counts[video_id] = 0


def test_failed_question_does_not_use_quota() -> None:
    reset_api_state()
    set_active_video()

    with patch("app.answer_question", side_effect=RuntimeError("Groq unavailable")):
        response = client.post("/ask", json={"question": "What happened?"})

    assert response.status_code == 500
    assert app.video_question_counts[app.active_video_id] == 0


def test_successful_question_uses_quota() -> None:
    reset_api_state()
    set_active_video()

    with patch("app.answer_question", return_value={"answer": "It worked", "sources": []}):
        response = client.post("/ask", json={"question": "What happened?"})

    assert response.status_code == 200
    assert app.video_question_counts[app.active_video_id] == 1


def test_sixth_question_is_rejected() -> None:
    reset_api_state()
    set_active_video()
    app.video_question_counts[app.active_video_id] = app.MAX_QUESTIONS_PER_VIDEO

    response = client.post("/ask", json={"question": "One more question?"})

    assert response.status_code == 429
    assert "maximum limit" in response.json()["detail"]


def test_analyzing_a_video_resets_its_quota() -> None:
    reset_api_state()
    video_id = "abcdefghijk"
    app.video_question_counts[video_id] = app.MAX_QUESTIONS_PER_VIDEO
    vector_store = Mock()
    vector_store.index.ntotal = 4

    with patch("app.create_video_vector_store", return_value=vector_store):
        response = client.post(
            "/analyze", json={"video_url": f"https://www.youtube.com/watch?v={video_id}"}
        )

    assert response.status_code == 200
    assert app.video_question_counts[video_id] == 0
