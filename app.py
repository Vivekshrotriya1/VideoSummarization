"""FastAPI backend for Video Summarizer."""

import logging
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_service import answer_question, create_video_vector_store, explain_documents

app = FastAPI(title="Video Summarizer API", version="0.1.0")
logger = logging.getLogger(__name__)

# The extension runs in a different browser origin, so it needs CORS.
# Restrict this to the extension's chrome-extension:// ID before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary in-memory storage. Restarting the API clears these indexes.
video_stores: dict[str, Any] = {}
active_video_id: str | None = None
MAX_QUESTIONS_PER_VIDEO = 5
video_question_counts: dict[str, int] = {}


class AnalyzeRequest(BaseModel):
    video_url: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=3000)


class MomentRequest(BaseModel):
    timestamp: float = Field(ge=0)

def extract_video_id(video_url: str) -> str:
    """Extract an 11-character ID from common YouTube URL formats."""
    parsed = urlparse(video_url.strip())
    host = parsed.netloc.lower().removeprefix("www.")

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/")):
            video_id = parsed.path.strip("/").split("/")[1]
        else:
            video_id = ""
    else:
        video_id = ""

    if len(video_id) != 11:
        raise ValueError("Please provide a valid YouTube video link.")
    return video_id


def check_question_quota() -> None:
    """Reject a request when the active video has reached its question limit."""
    if active_video_id is None:
        raise HTTPException(
            status_code=404,
            detail="No video analyzed yet. Call POST /analyze with a YouTube link first.",
        )

    used_questions = video_question_counts.get(active_video_id, 0)
    if used_questions >= MAX_QUESTIONS_PER_VIDEO:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Your maximum limit of {MAX_QUESTIONS_PER_VIDEO} questions for this "
                "video has been exceeded. Analyze a new video to start again."
            ),
        )


def record_question() -> None:
    """Record one successful AI response for the active video."""
    if active_video_id is None:
        raise HTTPException(
            status_code=404,
            detail="No video analyzed yet. Call POST /analyze with a YouTube link first.",
        )

    video_question_counts[active_video_id] = (
        video_question_counts.get(active_video_id, 0) + 1
    )


@app.get("/")
def health_check() -> dict[str, str]:
    """Confirm that the API server is running."""
    return {"message": "Video Summarizer API is running"}


@app.post("/analyze")
def analyze_video(request: AnalyzeRequest) -> dict[str, str | int]:
    """Analyze a YouTube link and make it the active video for /ask."""
    global active_video_id
    try:
        video_id = extract_video_id(request.video_url)
        vector_store = create_video_vector_store(video_id)
        video_stores[video_id] = vector_store
        active_video_id = video_id
        video_question_counts[video_id] = 0
        return {
            "message": "Video analyzed successfully",
            "video_id": video_id,
            "chunk_count": vector_store.index.ntotal,
        }
    except Exception as error:
        logger.exception("Video analysis failed for URL: %s", request.video_url)
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/ask")
def ask_question(request: AskRequest) -> dict[str, Any]:
    """Answer a question about the most recently analyzed video."""
    if active_video_id is None:
        raise HTTPException(
            status_code=404,
            detail="No video analyzed yet. Call POST /analyze with a YouTube link first.",
        )

    vector_store = video_stores.get(active_video_id)
    if vector_store is None:
        raise HTTPException(
            status_code=404,
            detail="The active video is unavailable. Call POST /analyze again.",
        )

    check_question_quota()
    try:
        result = answer_question(vector_store, request.question)
        record_question()
        return result
    except Exception as error:
        logger.exception("Question answering failed for video: %s", active_video_id)
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/explain-moment")
def explain_moment(request: MomentRequest) -> dict[str, Any]:
    """Explain the transcript section nearest to the current video timestamp."""
    if active_video_id is None:
        raise HTTPException(
            status_code=404,
            detail="No video analyzed yet. Call POST /analyze with a YouTube link first.",
        )

    vector_store = video_stores.get(active_video_id)
    if vector_store is None:
        raise HTTPException(status_code=404, detail="The active video is unavailable.")

    nearby_documents = [
        document
        for document in vector_store.docstore._dict.values()
        if document.metadata["start_time"] <= request.timestamp + 30
        and document.metadata["end_time"] >= request.timestamp - 30
    ]
    if not nearby_documents:
        raise HTTPException(
            status_code=404,
            detail="No transcript section was found near this moment.",
        )

    nearby_documents.sort(
        key=lambda document: abs(
            ((document.metadata["start_time"] + document.metadata["end_time"]) / 2)
            - request.timestamp
        )
    )
    check_question_quota()
    try:
        result = explain_documents(nearby_documents[:3], request.timestamp)
        result["timestamp"] = request.timestamp
        record_question()
        return result
    except Exception as error:
        logger.exception("Moment explanation failed for video: %s", active_video_id)
        raise HTTPException(status_code=500, detail=str(error)) from error
# app.py