"""Ad-hoc harness to exercise get_llm_seeding with recorded heuristics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llm_seeding import get_llm_seeding
from llm_seeding.utils import sample_heuristics

DEFAULT_RESULTS_JSON = os.path.join(
    _REPO_ROOT,
    "results",
    "training",
    "20250920-0824",
    "nsw_Bitbrains_3OS_NSGP2_0_core_8.json",
)

TERMINAL_DESCRIPTIONS: Dict[str, str] = {
    "container_cpu": "Requested CPU of the container.",
    "container_memories": "Requested memory footprint of the container.",
    "remaining_cpu_capacity": "Unused CPU capacity on the selected host.",
    "remaining_memory_capacity": "Unused memory capacity on the selected host.",
    "vm_cpu_overhead": "CPU overhead of running the VM/container combination.",
    "vm_memory_overhead": "Memory overhead of running the VM/container combination.",
    "vm_pm_innerc": "Intra-PM communication cost for the VM.",
    "vm_pm_outerc": "Inter-PM communication cost for the VM.",
    "affinity": "Affinity score between the workload and host.",
    "vm_cpu_capacity": "CPU capacity allocated to the VM.",
    "vm_memory_capacity": "Memory capacity allocated to the VM.",
    "pm_cpu_capacity": "Total CPU capacity of the physical machine.",
    "pm_memory_capacity": "Total memory capacity of the physical machine.",
    "pm_core": "Number of cores on the physical machine.",
    "pm_innerc": "Intra-PM communication overhead.",
    "pm_outerc": "Inter-PM communication overhead.",
}

DEFAULT_MIN_HEURISTICS = 6
DEFAULT_MAX_HEURISTICS = 8
RAW_HEURISTICS_LIMIT = DEFAULT_MAX_HEURISTICS * 3


def _load_first_front_heuristics(json_path: str, limit: int) -> List[Dict[str, Any]]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Unable to locate JSON at {json_path}")

    with open(json_path, "r") as handle:
        payload = json.load(handle)

    generations = payload.get("generation")
    if not isinstance(generations, Mapping):
        raise ValueError("Invalid JSON structure: expected 'generation' dict")

    def _sort_key(key: str):
        key_str = str(key)
        try:
            return (0, int(key_str))
        except ValueError:
            return (1, key_str)

    for gen_key in sorted(generations.keys(), key=_sort_key):
        first_front = generations[gen_key].get("first_front")
        if isinstance(first_front, list) and first_front:
            trimmed: List[Dict[str, Any]] = []
            for entry in first_front[:limit]:
                if not isinstance(entry, Mapping):
                    continue
                trimmed.append(
                    {
                        field: entry.get(field)
                        for field in (
                            "expr",
                            "energy",
                            "energy_norm",
                            "communication",
                            "communication_norm",
                        )
                    }
                )
            if trimmed:
                return trimmed
    raise ValueError("No first_front entries found in the provided JSON")


def run_demo(json_path: str) -> None:
    raw_heuristics = _load_first_front_heuristics(
        json_path, limit=RAW_HEURISTICS_LIMIT
    )
    heuristics = sample_heuristics(
        raw_heuristics,
        min_heuristics=DEFAULT_MIN_HEURISTICS,
        max_heuristics=DEFAULT_MAX_HEURISTICS,
    )
    suggestions = get_llm_seeding(
        heuristics,
        TERMINAL_DESCRIPTIONS,
        api_key="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",
    )

    print(
        f"Generated {len(suggestions)} suggestions from {len(heuristics)} heuristics\n"
        f"using data from {json_path}:"
    )
    for idx, suggestion in enumerate(suggestions, start=1):
        expr = suggestion.get("expr", "<missing expr>")
        expected = suggestion.get("expected", "<missing expected>")
        reason = suggestion.get("reason", "<missing reason>")
        group = suggestion.get("group", "<missing group>")
        print(
            f"  {idx}. group={group}, expr={expr}\n"
            f"     expected={expected}\n"
            f"     reason={reason}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangChain seeding demo harness")
    parser.add_argument(
        "--json",
        dest="json_path",
        default=DEFAULT_RESULTS_JSON,
        help="Path to the results JSON file containing heuristic fronts.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        run_demo(args.json_path)
    except Exception as exc:  # pragma: no cover - CLI helper
        print(f"LLM demo failed: {exc}")


if __name__ == "__main__":
    main()
