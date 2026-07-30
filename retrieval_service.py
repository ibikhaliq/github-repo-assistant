import re
from typing import Any

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "used",
    "what",
    "where",
    "which",
    "with",
    "today",
    "currently",
    "please",
    "tell",
    "show",
    "explain",
}

TOKEN_ALIASES = {
    "javascript": {"javascript", "js"},
    "typescript": {"typescript", "ts"},
    "python": {"python", "py"},
    "stylesheet": {"stylesheet", "css"},
    "html": {"html"},
}

def expand_tokens(tokens: set[str]) -> set[str]:
    expanded_tokens = set(tokens)

    for token in tokens:
        aliases = TOKEN_ALIASES.get(token, set())
        expanded_tokens.update(aliases)

    return expanded_tokens


def tokenize_text(text: str) -> set[str]:
    raw_tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )

    filtered_tokens = {
        token
        for token in raw_tokens
        if token not in STOP_WORDS
        and len(token) > 1
    }

    return expand_tokens(filtered_tokens)


def calculate_relevance_score(
    question_tokens: set[str],
    file_summary: dict[str, Any],
) -> tuple[int, set[str], set[str]]:
    summary = file_summary.get("summary", {})

    file_path = str(
        file_summary.get("file_path", "")
    )

    file_path_tokens = tokenize_text(file_path)

    summary_text = " ".join(
        [
            str(summary.get("purpose", "")),
            " ".join(
                summary.get(
                    "important_components",
                    [],
                )
            ),
            " ".join(
                summary.get(
                    "dependencies",
                    [],
                )
            ),
            str(
                summary.get(
                    "role_in_project",
                    "",
                )
            ),
            " ".join(
                summary.get(
                    "potential_issues",
                    [],
                )
            ),
        ]
    )

    summary_tokens = tokenize_text(summary_text)

    file_path_matches = question_tokens.intersection(
        file_path_tokens
    )

    summary_matches = question_tokens.intersection(
        summary_tokens
    )

    file_path_score = len(file_path_matches) * 3
    summary_score = len(summary_matches)

    total_score = file_path_score + summary_score

    return total_score, file_path_matches, summary_matches


def retrieve_relevant_file_summaries(
    repository_data: dict[str, Any],
    question: str,
    top_k: int = 5,
    min_score: int = 2,
) -> tuple[list[dict[str, Any]], str]:
    question_tokens = tokenize_text(question)

    file_summaries = repository_data.get(
        "file_summaries",
        [],
    )

    scored_summaries = []

    for file_summary in file_summaries:
        score, file_path_matches, summary_matches = (
            calculate_relevance_score(
                question_tokens=question_tokens,
                file_summary=file_summary,
            )
        )

        scored_summaries.append(
            {
                "score": score,
                "summary": file_summary,
                "file_path_matches": sorted(
                    file_path_matches
                ),
                "summary_matches": sorted(
                    summary_matches
                ),
            }
        )

    scored_summaries.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    standard_results = [
        item
        for item in scored_summaries
        if item["score"] >= min_score
    ]

    if standard_results:
        return standard_results[:top_k], "standard"

    fallback_results = [
        item
        for item in scored_summaries
        if item["file_path_matches"]
        or len(item["summary_matches"]) >= 2
    ]

    if fallback_results:
        return fallback_results[:top_k], "fallback"

    return [], "none"