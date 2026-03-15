"""Generate LLM-backed seeding suggestions that balance energy and communication."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence
from typing import Literal
from pydantic import BaseModel, Field

try:  # Prefer modern import path exposed by langchain-core.
    from langchain_core.output_parsers import PydanticOutputParser
except ImportError:  # Backwards compatibility for older LangChain releases.
    from langchain.output_parsers import PydanticOutputParser  # type: ignore

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import ast
# from langchain_community.chat_models import ChatDeepSeek

HeuristicGroup = Literal["energy-leaning", "balanced", "communication-leaning"]

_GROUP_NAMES: tuple[HeuristicGroup, ...] = (
    "energy-leaning",
    "balanced",
    "communication-leaning",
)

_SUGGESTIONS_PER_GROUP = 2 #2, 3, 4
_TOTAL_SUGGESTIONS = _SUGGESTIONS_PER_GROUP * len(_GROUP_NAMES)

VM_TERMINALS = ["container_cpu", "container_memories", "remaining_cpu_capacity", "remaining_memory_capacity", "vm_cpu_overhead", "vm_memory_overhead", "vm_pm_innerc", "vm_pm_outerc", "affinity"]
PM_TERMINALS = ["vm_cpu_capacity", "vm_memory_capacity", "remaining_cpu_capacity", "remaining_memory_capacity", "pm_cpu_capacity", "pm_memory_capacity", "pm_core", "pm_innerc", "pm_outerc", "affinity"]


class _SeedingSuggestion(BaseModel):
    group: HeuristicGroup = Field(
        ...,
        description="Target group: energy-leaning, balanced, or communication-leaning.",
    )
    vm_expr: str = Field(
        ...,
        description="Expression guiding VM-level selection logic (string form).",
    )
    pm_expr: str = Field(
        ...,
        description="Expression guiding PM-level selection logic (string form).",
    )
    # expected: str = Field(
    #     ...,
    #     description="Concise expectation of impact, e.g. 'energy -3%, communication -2%'.",
    # )
    # reason: str = Field(
    #     ..., description="Brief justification grounded in supplied heuristics.",
    # )


class _SeedingResponse(BaseModel):
    suggestions: Sequence[_SeedingSuggestion] = Field(
        ...,
        min_items=_TOTAL_SUGGESTIONS,
        description="Ordered list of proposed seeding actions.",
    )


def _build_structured_output_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=_SeedingResponse)

def _build_feedback(feedback: Dict):
    if len(feedback["Excellent"]) == 0 and \
        len(feedback["Good"]) == 0 and \
        len(feedback["Bad"]) == 0:
        return "(no feedback provided)"
    
    reflection = "Here is the feedback for the previously generated heuristics " + \
                "(three levels: Excellent, Good, and Bad):\n"
    for key, value in feedback.items():
        if len(value) != 0:
            reflection += f"{key}: "
            for ind in value:
                reflection += f"{ind}\n"
    
    return reflection

def _build_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
             (
                "system",
                "You are an expert synthesizing heuristic seeding expressions."
                "{format_instructions}",
            ),
            (
                "human",
                "You are given two genetic programming parent individuals for each task. "
                "Each individual contains two trees: 'vm' and 'pm'.\n"
                "Primitive sets:\n"
                "- Arithmetic operators: add(a,b),subtract(a,b), multiply(a,b),protectedDiv(a,b)\n"
                "- VM terminals: {vm_terminals}\n"
                "- PM terminals: {pm_terminals}\n"
                "All produced individuals must strictly use only the provided terminals and operators.\n"
                
                "Task 1\n"
                " Parent A: vm: {vm_group1_a}, pm: {pm_group1_a}\n"
                " Parent B: vm: {vm_group1_b}, pm: {pm_group1_b}\n"
                " Produce up to 2 offspring individuals likely to minimize energy by combining useful subtrees and simplifying where possible"
                "Task 2\n"
                " Parent A: vm: {vm_group2_a}, pm: {pm_group2_a}\n"
                " Parent B: vm: {vm_group2_b}, pm: {pm_group2_b}\n"
                " Produce up to 2 offspring individuals likely to balance energy and communication by combining useful subtrees and simplifying where possible"
                "Task 3\n"
                " Parent A: vm: {vm_group3_a}, pm: {pm_group3_a}\n"
                " Parent B: vm: {vm_group3_b}, pm: {pm_group3_b}\n"
                " Produce up to 2 offspring individuals likely to minimize communication by combining useful subtrees and simplifying where possible"
            ),
        ]
    )
    return prompt.partial(format_instructions=parser.get_format_instructions())


def _build_chat_model(
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
    temperature: float = 1,
) -> ChatOpenAI:
    print(model)
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError("Missing OpenAI API key. Pass api_key or set OPENAI_API_KEY.")

    if model.lower().startswith("deepseek"):
        return ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=resolved_key, openai_api_base="https://api.deepseek.com",)

    return ChatOpenAI(model=model, temperature=temperature, api_key=resolved_key)


def _format_heuristics(heuristics: Sequence[Mapping[str, Any]]) -> str:
    if not heuristics:
        return "(no heuristics provided)"

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for item in heuristics:
        group = str(item.get("group", "(unassigned)"))
        grouped.setdefault(group, []).append(item)

    lines: List[str] = []
    for group in _GROUP_NAMES:
        items = grouped.pop(group, None)
        if not items:
            continue
        lines.append(f"{group.capitalize()} heuristics:")
        for idx, item in enumerate(items, start=1):
            expr = item.get("expr", "<missing expr>")
            energy = item.get("energy", "?")
            energy_norm = item.get("energy_norm", "?")
            comm = item.get("communication", "?")
            comm_norm = item.get("communication_norm", "?")
            lines.append(
                f"  #{idx}: {expr}\n"
                f"     energy={energy} (norm={energy_norm}),"
                f" communication={comm} (norm={comm_norm})"
            )

    for group, items in grouped.items():
        lines.append(f"{group} heuristics:")
        for idx, item in enumerate(items, start=1):
            expr = item.get("expr", "<missing expr>")
            energy = item.get("energy", "?")
            energy_norm = item.get("energy_norm", "?")
            comm = item.get("communication", "?")
            comm_norm = item.get("communication_norm", "?")
            lines.append(
                f"  #{idx}: {expr}\n"
                f"     energy={energy} (norm={energy_norm}),"
                f" communication={comm} (norm={comm_norm})"
            )

    return "\n".join(lines)


def _format_terminal_nodes(terminal_nodes: Mapping[str, Any]) -> str:
    if not terminal_nodes:
        return "(no terminal nodes described)"

    lines = []
    for category, entries in terminal_nodes.items():
        if isinstance(entries, Mapping):
            lines.append(f"{category}:")
            for name, description in entries.items():
                lines.append(f"  - {name}: {description}")
        else:
            lines.append(f"- {category}: {entries}")
    return "\n".join(lines)


def _model_to_dict(model: BaseModel) -> Mapping[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump()
    fields = getattr(model, "__fields__", None)
    if fields:
        return {name: getattr(model, name) for name in fields}
    raise TypeError("Unsupported Pydantic model implementation; expected model_dump or __fields__.")


def get_llm_cx(
    heuristics: Sequence[Mapping[str, Any]],
    terminal_nodes: Mapping[str, Any],
    *,
    feedback=None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Sequence[Mapping[str, Any]]:
    """Return LLM-generated seeding suggestions informed by existing heuristics."""
    parser = _build_structured_output_parser()
    prompt = _build_prompt(parser)
    chat_model = _build_chat_model(api_key=api_key, model=model, temperature=temperature)

    chain = prompt | chat_model | parser

    # print(formatted_prompt)
    response = chain.invoke(
        {
            "heuristics": _format_heuristics(heuristics),
            "vm_terminals": VM_TERMINALS,
            "pm_terminals": PM_TERMINALS,
            "vm_group1_a": ast.literal_eval(heuristics[0]["expr"])["vm"],
            "pm_group1_a": ast.literal_eval(heuristics[0]["expr"])["pm"],
            "vm_group1_b": ast.literal_eval(heuristics[1]["expr"])["vm"],
            "pm_group1_b": ast.literal_eval(heuristics[1]["expr"])["pm"],
            "vm_group2_a": ast.literal_eval(heuristics[2]["expr"])["vm"],
            "pm_group2_a": ast.literal_eval(heuristics[2]["expr"])["pm"],
            "vm_group2_b": ast.literal_eval(heuristics[3]["expr"])["vm"],
            "pm_group2_b": ast.literal_eval(heuristics[3]["expr"])["pm"],
            "vm_group3_a": ast.literal_eval(heuristics[4]["expr"])["vm"],
            "pm_group3_a": ast.literal_eval(heuristics[4]["expr"])["pm"],
            "vm_group3_b": ast.literal_eval(heuristics[5]["expr"])["vm"],
            "pm_group3_b": ast.literal_eval(heuristics[5]["expr"])["pm"],
            "suggestions_per_group": _SUGGESTIONS_PER_GROUP,
            "total_suggestions": _TOTAL_SUGGESTIONS,
        }
    )
    formatted: List[Mapping[str, Any]] = []
    for suggestion in getattr(response, "suggestions", []):
        data = _model_to_dict(suggestion)
        vm_expr = data.get("vm_expr", "")
        pm_expr = data.get("pm_expr", "")
        expr = f"{{'vm': '{vm_expr}', 'pm': '{pm_expr}'}}"
        formatted.append(
            {
                "expr": expr,
                # "expected": data.get("expected", ""),
                # "reason": data.get("reason", ""),
                "group": data.get("group", ""),
            }
        )

    return formatted


__all__ = ["get_llm_seeding"]
