"""Reusable RAG functions for question-answering over YouTube transcripts."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# Free multilingual model. It supports semantic search across many languages.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class TranscriptNotAvailableError(Exception):
    """Raised when YouTube does not provide usable captions for a video."""


def get_transcript(video_id: str) -> list[dict[str, Any]]:
    """Fetch the best available YouTube caption language with timestamps.

    Manually created captions are preferred over auto-generated captions. The
    first available caption language is used, so callers do not need to supply
    a language code.
    """
    try:
        available_transcripts = list(YouTubeTranscriptApi().list(video_id))
        if not available_transcripts:
            raise TranscriptNotAvailableError

        transcript_choice = next(
            (item for item in available_transcripts if not item.is_generated),
            available_transcripts[0],
        )
        transcript = transcript_choice.fetch()
    except TranscriptNotAvailableError:
        raise
    except Exception as error:
        raise TranscriptNotAvailableError(
            "Transcript not available for this video."
        ) from error

    return [
        {
            "text": snippet.text.strip(),
            "start": float(snippet.start),
            "duration": float(snippet.duration),
        }
        for snippet in transcript
        if snippet.text.strip()
    ]


def transcript_to_documents(
    transcript: list[dict[str, Any]], video_id: str, chunk_size: int = 1000
) -> list[Document]:
    """Group captions into searchable chunks with a start/end timestamp."""
    documents: list[Document] = []
    texts: list[str] = []
    start_time: float | None = None
    end_time: float | None = None

    def add_document() -> None:
        if not texts or start_time is None or end_time is None:
            return
        documents.append(
            Document(
                page_content=" ".join(texts),
                metadata={
                    "video_id": video_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s",
                },
            )
        )

    for caption in transcript:
        text = caption["text"]
        if texts and len(" ".join(texts)) + len(text) + 1 > chunk_size:
            add_document()
            texts, start_time, end_time = [], None, None
        if start_time is None:
            start_time = caption["start"]
        texts.append(text)
        end_time = caption["start"] + caption["duration"]

    add_document()
    if not documents:
        raise TranscriptNotAvailableError("Transcript not available for this video.")
    return documents


def build_vector_store(documents: list[Document]) -> FAISS:
    """Embed transcript documents in an in-memory FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(documents, embeddings)


def answer_question(vector_store: FAISS, question: str, k: int = 4) -> dict[str, Any]:
    """Return a grounded answer using diverse, relevant transcript chunks."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

    # MMR keeps results relevant while avoiding near-duplicate transcript chunks.
    sources = vector_store.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=min(max(k * 5, 12), vector_store.index.ntotal),
        lambda_mult=0.65,
    )
    context = "\n\n".join(
        f"[Source {index + 1}, {document.metadata['start_time']:.1f}s-"
        f"{document.metadata['end_time']:.1f}s]\n{document.page_content}"
        for index, document in enumerate(sources)
    )
    prompt = f"""You are a helpful assistant answering questions about a YouTube video.
Answer only using the transcript context below. If it does not contain enough
information, say exactly: \"I don't know based on this video's transcript.\"

Transcript context:
{context}

Question: {question}
"""
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL), api_key=api_key)
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "sources": [
            {
                "text": document.page_content,
                "start_time": document.metadata["start_time"],
                "end_time": document.metadata["end_time"],
                "url": document.metadata["youtube_url"],
            }
            for document in sources
        ],
    }


def explain_documents(documents: list[Document], timestamp: float) -> dict[str, Any]:
    """Explain only the supplied transcript documents near a video moment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

    context = "\n\n".join(
        f"[{document.metadata['start_time']:.1f}s-{document.metadata['end_time']:.1f}s]\n"
        f"{document.page_content}"
        for document in documents
    )
    prompt = f"""You are a helpful assistant explaining one exact moment in a YouTube video.
Explain simply what is being discussed around {timestamp:.0f} seconds.
Use ONLY the nearby transcript context below. Do not use previous questions,
general video topics, or information from elsewhere.

Nearby transcript context:
{context}
"""
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL), api_key=api_key)
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "sources": [
            {
                "text": document.page_content,
                "start_time": document.metadata["start_time"],
                "end_time": document.metadata["end_time"],
                "url": document.metadata["youtube_url"],
            }
            for document in documents
        ],
    }


def create_video_vector_store(video_id: str) -> FAISS:
    """Convenience function used by the future API and Chrome extension."""
    transcript = get_transcript(video_id)
    documents = transcript_to_documents(transcript, video_id)
    return build_vector_store(documents)
