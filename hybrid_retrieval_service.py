from typing import Any


KEYWORD_WEIGHT = 0.45
SEMANTIC_WEIGHT = 0.55
OVERLAP_BONUS = 0.10


def rank_hybrid_matches(
    keyword_results: list[dict[str, Any]],
    semantic_matches: list[dict[str, Any]],
    file_summaries: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Combine keyword and semantic retrieval results into
    one ranked list.

    Files that appear in both retrieval methods receive
    an overlap bonus.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    summary_by_path = {
        file_summary.get("file_path", ""): file_summary
        for file_summary in file_summaries
        if file_summary.get("file_path")
    }

    hybrid_records: dict[str, dict[str, Any]] = {}

    highest_keyword_score = max(
        (
            result.get("score", 0)
            for result in keyword_results
        ),
        default=0,
    )

    # ----------------------------------
    # Add keyword retrieval results
    # ----------------------------------
    for result in keyword_results:
        file_summary = result.get("summary", {})

        file_path = file_summary.get(
            "file_path",
            "",
        )

        if not file_path:
            continue

        raw_keyword_score = result.get(
            "score",
            0,
        )

        if highest_keyword_score > 0:
            normalized_keyword_score = (
                raw_keyword_score
                / highest_keyword_score
            )
        else:
            normalized_keyword_score = 0.0

        hybrid_records[file_path] = {
            "file_path": file_path,
            "summary": file_summary,
            "keyword_score": raw_keyword_score,
            "normalized_keyword_score": (
                normalized_keyword_score
            ),
            "semantic_score": 0.0,
            "matched_by_keyword": True,
            "matched_by_semantic": False,
        }

    # ----------------------------------
    # Add semantic retrieval results
    # ----------------------------------
    for match in semantic_matches:
        file_path = match.get(
            "file_path",
            "",
        )

        if not file_path:
            continue

        semantic_score = match.get(
            "similarity_score",
            0.0,
        )

        if file_path not in hybrid_records:
            hybrid_records[file_path] = {
                "file_path": file_path,
                "summary": summary_by_path.get(
                    file_path,
                    {},
                ),
                "keyword_score": 0,
                "normalized_keyword_score": 0.0,
                "semantic_score": semantic_score,
                "matched_by_keyword": False,
                "matched_by_semantic": True,
            }
        else:
            hybrid_records[file_path][
                "semantic_score"
            ] = semantic_score

            hybrid_records[file_path][
                "matched_by_semantic"
            ] = True

    # ----------------------------------
    # Calculate final hybrid score
    # ----------------------------------
    ranked_results = []

    for record in hybrid_records.values():
        hybrid_score = (
            KEYWORD_WEIGHT
            * record["normalized_keyword_score"]
            + SEMANTIC_WEIGHT
            * record["semantic_score"]
        )

        if (
            record["matched_by_keyword"]
            and record["matched_by_semantic"]
        ):
            hybrid_score += OVERLAP_BONUS

        record["hybrid_score"] = hybrid_score

        ranked_results.append(record)

    ranked_results.sort(
        key=lambda record: record["hybrid_score"],
        reverse=True,
    )

    return ranked_results[:top_k]