from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

_HEURISTIC_GROUP_ORDER = (
    "energy-leaning",
    "balanced",
    "communication-leaning",
)


def _as_float(value: Any, default: float = math.inf) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _deduplicate(heuristics: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    unique: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in heuristics:
        expr = item.get("expr")
        if not isinstance(expr, str):
            continue
        if expr in seen:
            continue
        unique.append(item)
        seen.add(expr)
    return unique


def _energy_score(item: Mapping[str, Any]) -> float:
    return _as_float(item.get("energy_norm"))


def _energy_value(item: Mapping[str, Any]) -> float:
    return _as_float(item.get("energy"), default=-math.inf)


def _communication_score(item: Mapping[str, Any]) -> float:
    return _as_float(item.get("communication_norm"))


def _knee_score(item: Mapping[str, Any]) -> float:
    return math.hypot(_energy_score(item), _communication_score(item))


def sample_heuristics(
    heuristics: Sequence[Mapping[str, Any]],
    *,
    min_heuristics: int,
    max_heuristics: int | None,
) -> List[Mapping[str, Any]]:
    """Return representative heuristics sampled across the Pareto front."""

    filtered = _deduplicate(heuristics)
    if not filtered:
        return []

    if len(filtered) <= min_heuristics:
        return list(filtered)

    if max_heuristics is not None:
        if max_heuristics <= 0:
            return []
        target_total = min(len(filtered), max_heuristics)
    else:
        target_total = len(filtered)

    if target_total <= 0:
        return []

    if len(filtered) < 3:
        return filtered[:target_total]

    diff_sorted = sorted(
        filtered,
        key=lambda item: _energy_score(item) - _communication_score(item),
    )
    chunk = max(1, len(filtered) // 3)
    energy_bucket = diff_sorted[:chunk]
    balanced_bucket = diff_sorted[chunk : len(filtered) - chunk]
    communication_bucket = list(reversed(diff_sorted[-chunk:]))

    if not balanced_bucket:
        mid = diff_sorted[len(filtered) // 2]
        balanced_bucket = (
            [mid] if mid not in energy_bucket + communication_bucket else []
        )
        if not balanced_bucket:
            balanced_bucket = [
                item
                for item in diff_sorted
                if item not in energy_bucket and item not in communication_bucket
            ]

    buckets: Dict[str, List[Mapping[str, Any]]] = {
        "energy-leaning": energy_bucket,
        "balanced": balanced_bucket,
        "communication-leaning": communication_bucket,
    }

    category_by_expr: Dict[str, str] = {}
    for name, bucket in buckets.items():
        for item in bucket:
            expr = item.get("expr")
            if isinstance(expr, str) and expr not in category_by_expr:
                category_by_expr[expr] = name

    if target_total >= 9:
        per_category_limit = 3
    elif target_total >= 6:
        per_category_limit = 2
    else:
        per_category_limit = 1

    counts = {name: 0 for name in _HEURISTIC_GROUP_ORDER}
    selected: List[Mapping[str, Any]] = []
    seen: set[str] = set()

    def add(item: Mapping[str, Any], *, respect_limits: bool) -> bool:
        expr = item.get("expr")
        if not isinstance(expr, str) or expr in seen:
            return False
        category = category_by_expr.get(expr)
        if respect_limits and category and counts[category] >= per_category_limit:
            return False
        selected.append(item)
        seen.add(expr)
        if category:
            counts[category] += 1
        return True

    energy_end = min(filtered, key=_energy_score, default=None)
    knee = min(filtered, key=_knee_score, default=None)
    communication_end = min(filtered, key=_communication_score, default=None)

    anchors = [energy_end, knee, communication_end]
    for anchor in anchors:
        if anchor is None:
            continue
        add(anchor, respect_limits=False)
        if len(selected) >= target_total:
            return selected[:target_total]

    positions = {name: 0 for name in buckets}

    def pull_next(category: str) -> bool:
        bucket = buckets.get(category, [])
        while positions[category] < len(bucket):
            item = bucket[positions[category]]
            positions[category] += 1
            if add(item, respect_limits=True):
                return True
        return False

    while len(selected) < target_total:
        progress = False
        for category in _HEURISTIC_GROUP_ORDER:
            if counts.get(category, 0) >= per_category_limit:
                continue
            if len(selected) >= target_total:
                break
            if pull_next(category):
                progress = True
            if len(selected) >= target_total:
                break
        if not progress:
            break

    if len(selected) < target_total:
        for item in filtered:
            if len(selected) >= target_total:
                break
            add(item, respect_limits=False)

    final = selected[:target_total]
    final.sort(key=_energy_value, reverse=True)

    annotated: List[Mapping[str, Any]] = []
    for item in final:
        expr = item.get("expr")
        group = category_by_expr.get(expr if isinstance(expr, str) else "")
        annotated_item = dict(item)
        if group:
            annotated_item["group"] = group
        annotated.append(annotated_item)

    return annotated


__all__ = ["sample_heuristics"]
