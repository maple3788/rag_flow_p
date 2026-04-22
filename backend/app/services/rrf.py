from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    if rrf_k < 1:
        rrf_k = 1

    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        seen: set[int] = set()
        for rank, chunk_id in enumerate(ranked, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    # Sort by fused score desc, then chunk_id asc for deterministic tie breaks.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
