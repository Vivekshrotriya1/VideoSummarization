"""Run: python test_rag.py VIDEO_ID \"your question\""""

import argparse

from rag_service import answer_question, create_video_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about a YouTube video.")
    parser.add_argument("video_id", help="The 11-character YouTube video ID")
    parser.add_argument("question", help="Question to ask about the transcript")
    args = parser.parse_args()

    vector_store = create_video_vector_store(args.video_id)
    result = answer_question(vector_store, args.question)
    print(f"\nAnswer:\n{result['answer']}\n\nSources:")
    for source in result["sources"]:
        print(f"- {source['start_time']:.1f}s–{source['end_time']:.1f}s: {source['url']}")


if __name__ == "__main__":
    main()
