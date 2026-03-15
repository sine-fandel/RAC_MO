"""Helpers to derive LLM-seeded individuals from evaluated populations."""

from __future__ import annotations

import ast
import re
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence

from deap import tools

from optim.multi_tree_gp import MultiPrimitiveTree

from .langchain_llm import get_llm_seeding
from .langchain_cx import get_llm_cx
from .utils import sample_heuristics

_DECIMAL_PRECISION = 6

_VM_TERMINALS = {
    "container_cpu",
    "container_memories",
    "remaining_cpu_capacity",
    "remaining_memory_capacity",
    "vm_cpu_overhead",
    "vm_memory_overhead",
    "vm_pm_innerc",
    "vm_pm_outerc",
    "affinity",
}

_PM_TERMINALS = {
    "vm_cpu_capacity",
    "vm_memory_capacity",
    "remaining_cpu_capacity",
    "remaining_memory_capacity",
    "pm_cpu_capacity",
    "pm_memory_capacity",
    "pm_core",
    "pm_innerc",
    "pm_outerc",
    "affinity",
}

_FUNCTION_NAMES = {"add", "subtract", "multiply", "protectedDiv"}

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*")

_VM_TERMINAL_DESCRIPTIONS = {
    "container_cpu": "Requested CPU of the container.",
    "container_memories": "Requested memory footprint of the container.",
    "vm_cpu_overhead": "CPU overhead of running the VM/container combination.",
    "vm_memory_overhead": "Memory overhead of running the VM/container combination.",
    "vm_pm_innerc": "Intra-PM communication cost for the VM.",
    "vm_pm_outerc": "Inter-PM communication cost for the VM.",
}

_PM_TERMINAL_DESCRIPTIONS = {
    "vm_cpu_capacity": "CPU capacity allocated to the VM.",
    "vm_memory_capacity": "Memory capacity allocated to the VM.",
    "pm_cpu_capacity": "Total CPU capacity of the physical machine.",
    "pm_memory_capacity": "Total memory capacity of the physical machine.",
    "pm_core": "Number of cores on the physical machine.",
    "pm_innerc": "Intra-PM communication overhead.",
    "pm_outerc": "Inter-PM communication overhead.",
}

_COMMON_TERMINAL_DESCRIPTIONS = {
    "remaining_cpu_capacity": "Unused CPU capacity on the selected host.",
    "remaining_memory_capacity": "Unused memory capacity on the selected host.",
    "affinity": "Affinity score between the workload and host.",
}

_TERMINAL_DESCRIPTIONS: Mapping[str, Mapping[str, str]] = {
    "pm terminals": _PM_TERMINAL_DESCRIPTIONS,
    "vm terminals": _VM_TERMINAL_DESCRIPTIONS,
    "common": _COMMON_TERMINAL_DESCRIPTIONS,
}


def build_prompt_terminal_context() -> str:
    """Return a strictly formatted terminal inventory for LLM prompts."""
    vm_terms = ", ".join(sorted(_VM_TERMINALS))
    pm_terms = ", ".join(sorted(_PM_TERMINALS))
    vm_context = f"vm_expr terminals: {vm_terms}."
    pm_context = f"pm_expr terminals: {pm_terms}."
    restrictions = (
        "Rules: MUST use only vm_expr terminals for vm_expr and only pm_expr terminals for pm_expr; "
        "expressions may combine them with add, subtract, multiply, or protectedDiv; "
        "do not invent numeric constants or new identifiers."
    )
    return "\n".join([vm_context, pm_context, restrictions])


def _norm(value: float, vmin: float, vmax: float) -> float:
    if vmax == vmin:
        return 0.0
    return (value - vmin) / (vmax - vmin)


def _round(value: float) -> float:
    return round(value, _DECIMAL_PRECISION)


def _extract_first_front(population: Sequence) -> List[Mapping[str, Any]]:
    if not population:
        return []

    fronts = tools.sortNondominated(
        population, k=len(population), first_front_only=True
    )
    if not fronts:
        return []

    front: List = list(fronts[0])

    energies = [ind.fitness.values[0] for ind in front]
    comms = [ind.fitness.values[1] for ind in front]
    e_min, e_max = (min(energies), max(energies)) if energies else (0.0, 0.0)
    c_min, c_max = (min(comms), max(comms)) if comms else (0.0, 0.0)

    summary: List[Mapping[str, Any]] = []
    for ind in front:
        if not ind.fitness.valid:
            continue
        e_val, c_val = ind.fitness.values[:2]
        summary.append(
            {
                "expr": str(ind),
                "energy": _round(e_val),
                "communication": _round(c_val),
                "energy_norm": _round(_norm(e_val, e_min, e_max)),
                "communication_norm": _round(_norm(c_val, c_min, c_max)),
            }
        )

    return summary


def _fallback_terminal(terminals: Sequence[str]) -> str:
    if "affinity" in terminals:
        return "affinity"
    return terminals[0]


def _sanitize_expression(expr: str, type_key: str) -> str:
    terminals = sorted(_VM_TERMINALS if type_key == "vm" else _PM_TERMINALS)
    fallback = _fallback_terminal(terminals)

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in _FUNCTION_NAMES or token in terminals:
            return token
        if re.fullmatch(r"\d+\.?\d*", token):
            print(
                f"Replacing numeric literal '{token}' in {type_key} expression with '{fallback}'."
            )
            return fallback
        print(
            f"Replacing unsupported token '{token}' in {type_key} expression with '{fallback}'."
        )
        return fallback

    return _TOKEN_PATTERN.sub(_replace, expr)


def _is_valid_expression(expr: str, type_key: str) -> bool:
    terminals = _VM_TERMINALS if type_key == "vm" else _PM_TERMINALS
    tokens = _TOKEN_PATTERN.findall(expr)
    for token in tokens:
        if token in _FUNCTION_NAMES or token in terminals:
            continue
        if re.fullmatch(r"\d+\.?\d*", token):
            return False
        return False
    return True


def _parse_expr(expr: str | Mapping[str, Any]) -> MutableMapping[str, str]:
    data: Any
    if isinstance(expr, str):
        try:
            data = ast.literal_eval(expr)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Unable to parse expr '{expr}': {exc}") from exc
    else:
        data = expr

    if not isinstance(data, MutableMapping):
        raise ValueError(f"Expression must be dict-like, got {type(data)!r}")

    vm = data.get("vm")
    pm = data.get("pm")
    if not isinstance(vm, str) or not isinstance(pm, str):
        raise ValueError("Expression must contain 'vm' and 'pm' string entries")

    vm = _sanitize_expression(vm.strip(), "vm")
    pm = _sanitize_expression(pm.strip(), "pm")

    if not _is_valid_expression(vm, "vm"):
        raise ValueError(f"vm expression contains unsupported tokens: {vm}")
    if not _is_valid_expression(pm, "pm"):
        raise ValueError(f"pm expression contains unsupported tokens: {pm}")

    return {"vm": vm, "pm": pm}


def _to_multi_tree(expr: str | Mapping[str, Any], pset) -> MultiPrimitiveTree:
    parsed = _parse_expr(expr)
    return MultiPrimitiveTree.from_string(parsed, pset)


def generate_llm_cx_candidates(
    first_front: Sequence,
    pset,
    *,
    feedback = None,
    max_heuristics = 12,
    min_heuristics: int = 6,
    api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
    temperature: float = 0.2,
) -> List[MultiPrimitiveTree]:
    """Return MultiPrimitiveTree candidates suggested by the LLM.

    population
        Evaluated individuals used to build the heuristic context.
    pset
        Primitive set mapping used to instantiate new trees.
    """
    # heuristics = []
    # for pop in population:
    # first_front = _extract_first_front(population)
    heuristics = sample_heuristics(
        first_front,
        min_heuristics=min_heuristics,
        max_heuristics=max_heuristics,
    )

    if not heuristics:
        return []

    terminal_context = dict(_TERMINAL_DESCRIPTIONS)
    terminal_context["RULES"] = build_prompt_terminal_context()

    suggestions = get_llm_cx(
        heuristics,
        terminal_context,
        feedback=feedback,
        api_key=api_key,
        model=model,
        temperature=temperature,
    )

    if not suggestions:
        return []

    # seen = {str(ind) for ind in population}
    candidates: List[MultiPrimitiveTree] = []
    for suggestion in suggestions:
        expr = suggestion.get("expr") if isinstance(suggestion, Mapping) else None
        if not isinstance(expr, str):
            continue
        try:
            tree = _to_multi_tree(expr, pset)
        except ValueError as exc:
            print(f"Skipping malformed LLM expression: {exc}")
            continue
        tree_repr = str(tree)
        # if tree_repr in seen:
        #     continue
        # seen.add(tree_repr)
        candidates.append(tree)

    return candidates


__all__ = ["generate_llm_seed_candidates"]
