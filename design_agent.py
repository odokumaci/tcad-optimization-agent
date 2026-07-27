"""LangGraph agent for constrained MOSFET optimization and TCAD verification."""

import argparse
import asyncio
import json
import operator
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field


ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
TRACE_PATH: Path | None = None


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if any(word in key.lower() for word in ("key", "secret", "token"))
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def trace(event: str, data: Any) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "data": _redact(data),
    }
    line = json.dumps(record, default=str)
    print(line, flush=True)
    if TRACE_PATH is not None:
        with TRACE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class DesignTargets(BaseModel):
    max_ioff_na_per_um: float = Field(
        default=1.0,
        gt=0,
        description="Maximum off-current in nA/um.",
    )
    max_ss_mv_per_dec: float = Field(default=85.0, gt=0)
    max_dibl_mv_per_v: float = Field(default=50.0, gt=0)
    min_threshold_voltage_v: float | None = None


class AgentState(TypedDict):
    user_request: str
    max_verification_attempts: int
    iteration: int
    events: Annotated[list[str], operator.add]
    targets: NotRequired[dict[str, Any]]
    search_constraints: NotRequired[dict[str, Any]]
    domain: NotRequired[dict[str, Any]]
    optimization: NotRequired[dict[str, Any]]
    verification: NotRequired[dict[str, Any]]
    final_report: NotRequired[str]
    error: NotRequired[str]


def language_model() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in .env or the environment")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )


async def call_mcp_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    timeout_seconds: int = 240,
) -> dict[str, Any]:
    trace("mcp.input", {"tool": name, "arguments": arguments})
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "tcad_mcp_server.py")],
        cwd=str(ROOT),
        env={
            **os.environ,
            "SURROGATE_API_URL": os.getenv(
                "SURROGATE_API_URL", "http://127.0.0.1:8000"
            ),
        },
    )
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.call_tool(
                name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=timeout_seconds),
            )
            if result.isError:
                raise RuntimeError(f"MCP tool {name} failed: {result.content}")
            if result.structuredContent is None:
                raise RuntimeError(
                    f"MCP tool {name} returned no structured content"
                )
            trace(
                "mcp.output",
                {"tool": name, "result": result.structuredContent},
            )
            return result.structuredContent


def parse_requirements(state: AgentState) -> dict[str, Any]:
    parser = language_model().with_structured_output(DesignTargets)
    messages = [
        SystemMessage(
            content=(
                "Extract MOSFET optimization constraints. Convert off-current "
                "to nA/um. Use defaults when the user does not specify a target. "
                "Do not invent geometry constraints."
            )
        ),
        HumanMessage(content=state["user_request"]),
    ]
    trace(
        "openai.input.parse",
        [{"role": message.type, "content": message.content} for message in messages],
    )
    targets = parser.invoke(messages)
    target_values = targets.model_dump()
    trace("openai.output.parse", target_values)
    search_constraints = {
        "max_ioff_ua_per_um": target_values["max_ioff_na_per_um"] / 1000.0,
        "max_ss_mv_per_dec": target_values["max_ss_mv_per_dec"],
        "max_dibl_mv_per_v": target_values["max_dibl_mv_per_v"],
        "min_threshold_voltage_v": target_values["min_threshold_voltage_v"],
    }
    return {
        "targets": target_values,
        "search_constraints": search_constraints,
        "events": [f"Parsed targets: {target_values}"],
    }


def inspect_domain(_: AgentState) -> dict[str, Any]:
    domain = asyncio.run(call_mcp_tool("get_valid_design_ranges", {}))
    return {
        "domain": domain,
        "events": ["Loaded validated surrogate design ranges through MCP."],
    }


def optimize(state: AgentState) -> dict[str, Any]:
    constraints = state["search_constraints"]
    result = asyncio.run(
        call_mcp_tool(
            "optimize_device_design",
            {
                **constraints,
                "samples": 4096,
                "seed": 90 + state["iteration"],
                "top_k": 5,
            },
        )
    )
    update: dict[str, Any] = {
        "optimization": result,
        "events": [
            f"Optimization status={result['status']} after "
            f"{result['evaluated_candidates']} candidates."
        ],
    }
    if result["status"] != "success":
        update["error"] = "No feasible surrogate design was found."
    return update


def verify(state: AgentState) -> dict[str, Any]:
    if state["optimization"]["status"] != "success":
        return {"events": ["Skipped TCAD verification: no feasible design."]}
    design = state["optimization"]["best"]["design"]
    targets = state["targets"]
    try:
        verification = asyncio.run(
            call_mcp_tool(
                "verify_design_with_tcad",
                {
                    "gate_length_nm": design["gate_length_nm"],
                    "oxide_thickness_nm": design["oxide_thickness_nm"],
                    "halo_peak_doping_1e19": (
                        design["halo_peak_doping_cm3"] / 1.0e19
                    ),
                    "junction_depth_nm": design["junction_depth_nm"],
                    "max_ioff_ua_per_um": (
                        targets["max_ioff_na_per_um"] / 1000.0
                    ),
                    "max_ss_mv_per_dec": targets["max_ss_mv_per_dec"],
                    "max_dibl_mv_per_v": targets["max_dibl_mv_per_v"],
                },
                timeout_seconds=240,
            )
        )
    except Exception as error:
        message = f"TCAD verification failed: {error}"
        trace(
            "verification.error",
            {"type": type(error).__name__, "message": str(error)},
        )
        return {
            "error": message,
            "iteration": state["iteration"] + 1,
            "events": [message],
        }
    passed = verification["report"]["tcad_constraint_validation"]["all_passed"]
    return {
        "verification": verification,
        "iteration": state["iteration"] + 1,
        "events": [
            f"DEVSIM verification attempt {state['iteration'] + 1}: "
            f"{'passed' if passed else 'failed'}."
        ],
    }


def route_after_verification(state: AgentState) -> str:
    if state.get("error") or "verification" not in state:
        trace("graph.route", {"from": "verify", "to": "report"})
        return "report"
    passed = state["verification"]["report"]["tcad_constraint_validation"][
        "all_passed"
    ]
    if passed or state["iteration"] >= state["max_verification_attempts"]:
        trace("graph.route", {"from": "verify", "to": "report"})
        return "report"
    trace("graph.route", {"from": "verify", "to": "refine"})
    return "refine"


def refine_constraints(state: AgentState) -> dict[str, Any]:
    current = state["search_constraints"]
    refined = {
        **current,
        "max_ioff_ua_per_um": current["max_ioff_ua_per_um"] * 0.8,
        "max_ss_mv_per_dec": max(60.0, current["max_ss_mv_per_dec"] - 1.5),
        "max_dibl_mv_per_v": max(10.0, current["max_dibl_mv_per_v"] - 3.0),
    }
    trace(
        "constraints.refined",
        {"previous": current, "refined": refined},
    )
    return {
        "search_constraints": refined,
        "events": [
            "Tightened surrogate constraints to add margin after TCAD failure."
        ],
    }


def write_report(state: AgentState) -> dict[str, Any]:
    compact = {
        "request": state["user_request"],
        "targets": state.get("targets"),
        "domain": state.get("domain"),
        "optimization": state.get("optimization"),
        "verification": (
            {
                "run_id": state["verification"]["run_id"],
                "report_path": state["verification"]["report_path"],
                "design": state["verification"]["report"]["design"],
                "tcad_metrics": state["verification"]["report"]["tcad_metrics"],
                "surrogate_metrics": state["verification"]["report"][
                    "surrogate_metrics"
                ],
                "constraint_validation": state["verification"]["report"][
                    "tcad_constraint_validation"
                ],
            }
            if state.get("verification")
            else None
        ),
        "error": state.get("error"),
    }
    messages = [
        SystemMessage(
            content=(
                "Write a concise semiconductor engineering decision report. "
                "Use only the supplied facts. Clearly distinguish surrogate "
                "predictions from DEVSIM verification, state whether all "
                "constraints passed, and mention that the model is valid only "
                "inside its training domain."
            )
        ),
        HumanMessage(content=json.dumps(compact, indent=2)),
    ]
    trace(
        "openai.input.report",
        [{"role": message.type, "content": message.content} for message in messages],
    )
    response = language_model().invoke(messages)
    trace("openai.output.report", response.content)
    return {
        "final_report": str(response.content),
        "events": ["Generated final engineering report."],
    }


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("parse", parse_requirements)
    builder.add_node("domain", inspect_domain)
    builder.add_node("optimize", optimize)
    builder.add_node("verify", verify)
    builder.add_node("refine", refine_constraints)
    builder.add_node("report", write_report)
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "domain")
    builder.add_edge("domain", "optimize")
    builder.add_edge("optimize", "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_verification,
        {"refine": "refine", "report": "report"},
    )
    builder.add_edge("refine", "optimize")
    builder.add_edge("report", END)
    return builder.compile()


def main() -> None:
    global TRACE_PATH

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="+", help="Natural-language design goal")
    parser.add_argument("--max-verification-attempts", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "agent_runs")
    args = parser.parse_args()
    if args.max_verification_attempts < 1:
        raise ValueError("max-verification-attempts must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"design_agent_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    TRACE_PATH = run_dir / "trace.jsonl"
    request = " ".join(args.request)
    trace(
        "agent.start",
        {
            "request": request,
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "max_verification_attempts": args.max_verification_attempts,
        },
    )
    graph = build_graph()
    try:
        result = graph.invoke(
            {
                "user_request": request,
                "max_verification_attempts": args.max_verification_attempts,
                "iteration": 0,
                "events": [],
            }
        )
    except BaseException as error:
        trace(
            "agent.error",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise
    output_path = run_dir / "result.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = run_dir / "report.md"
    report_path.write_text(result["final_report"] + "\n", encoding="utf-8")
    trace(
        "agent.complete",
        {"result_path": str(output_path), "report_path": str(report_path)},
    )
    print(result["final_report"])
    print(f"\nRun record: {output_path}")
    print(f"Plain-English report: {report_path}")


if __name__ == "__main__":
    main()
