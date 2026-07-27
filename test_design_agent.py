"""Unit tests for the LangGraph design-agent control flow."""

import pytest

import design_agent


TARGETS = {
    "max_ioff_na_per_um": 1.0,
    "max_ss_mv_per_dec": 85.0,
    "max_dibl_mv_per_v": 50.0,
    "min_threshold_voltage_v": None,
}
SEARCH_CONSTRAINTS = {
    "max_ioff_ua_per_um": 0.001,
    "max_ss_mv_per_dec": 85.0,
    "max_dibl_mv_per_v": 50.0,
    "min_threshold_voltage_v": None,
}
DESIGN = {
    "gate_length_nm": 50.0,
    "oxide_thickness_nm": 1.2,
    "halo_peak_doping_cm3": 3.0e19,
    "junction_depth_nm": 30.0,
}


def initial_state(max_attempts: int = 2) -> dict:
    return {
        "user_request": "maximize Ion under the stated constraints",
        "max_verification_attempts": max_attempts,
        "iteration": 0,
        "events": [],
    }


def install_common_nodes(monkeypatch) -> None:
    def parse(_state):
        return {
            "targets": TARGETS.copy(),
            "search_constraints": SEARCH_CONSTRAINTS.copy(),
            "events": ["parsed"],
        }

    def domain(_state):
        return {"domain": {"training_domain": {}}, "events": ["domain"]}

    def report(state):
        return {
            "final_report": (
                "verification passed"
                if state.get("verification")
                and state["verification"]["report"][
                    "tcad_constraint_validation"
                ]["all_passed"]
                else f"verification unavailable: {state.get('error')}"
            ),
            "events": ["reported"],
        }

    monkeypatch.setattr(design_agent, "parse_requirements", parse)
    monkeypatch.setattr(design_agent, "inspect_domain", domain)
    monkeypatch.setattr(design_agent, "write_report", report)


def successful_optimization() -> dict:
    return {
        "status": "success",
        "evaluated_candidates": 256,
        "best": {"design": DESIGN.copy()},
    }


def verification_result(passed: bool) -> dict:
    return {
        "report": {
            "tcad_constraint_validation": {
                "all_passed": passed,
                "checks": {},
            }
        }
    }


def test_successful_verification_stops_after_one_attempt(monkeypatch) -> None:
    install_common_nodes(monkeypatch)
    calls = {"optimize": 0, "verify": 0}

    def optimize(_state):
        calls["optimize"] += 1
        return {
            "optimization": successful_optimization(),
            "events": ["optimized"],
        }

    def verify(state):
        calls["verify"] += 1
        return {
            "verification": verification_result(True),
            "iteration": state["iteration"] + 1,
            "events": ["verified"],
        }

    monkeypatch.setattr(design_agent, "optimize", optimize)
    monkeypatch.setattr(design_agent, "verify", verify)

    result = design_agent.build_graph().invoke(initial_state())

    assert result["final_report"] == "verification passed"
    assert result["iteration"] == 1
    assert calls == {"optimize": 1, "verify": 1}


def test_failed_verification_refines_then_retries(monkeypatch) -> None:
    install_common_nodes(monkeypatch)
    searched_constraints = []
    verification_calls = 0

    def optimize(state):
        searched_constraints.append(state["search_constraints"].copy())
        return {
            "optimization": successful_optimization(),
            "events": ["optimized"],
        }

    def verify(state):
        nonlocal verification_calls
        verification_calls += 1
        return {
            "verification": verification_result(verification_calls == 2),
            "iteration": state["iteration"] + 1,
            "events": ["verified"],
        }

    monkeypatch.setattr(design_agent, "optimize", optimize)
    monkeypatch.setattr(design_agent, "verify", verify)

    result = design_agent.build_graph().invoke(initial_state(max_attempts=2))

    assert result["final_report"] == "verification passed"
    assert result["iteration"] == 2
    assert len(searched_constraints) == 2
    assert searched_constraints[1]["max_ioff_ua_per_um"] == pytest.approx(0.0008)
    assert searched_constraints[1]["max_ss_mv_per_dec"] == pytest.approx(83.5)
    assert searched_constraints[1]["max_dibl_mv_per_v"] == pytest.approx(47.0)


def test_retry_loop_stops_at_configured_limit(monkeypatch) -> None:
    install_common_nodes(monkeypatch)
    calls = 0

    def optimize(_state):
        return {
            "optimization": successful_optimization(),
            "events": ["optimized"],
        }

    def verify(state):
        nonlocal calls
        calls += 1
        return {
            "verification": verification_result(False),
            "iteration": state["iteration"] + 1,
            "events": ["verified"],
        }

    monkeypatch.setattr(design_agent, "optimize", optimize)
    monkeypatch.setattr(design_agent, "verify", verify)

    result = design_agent.build_graph().invoke(initial_state(max_attempts=2))

    assert calls == 2
    assert result["iteration"] == 2
    assert "verification unavailable" in result["final_report"]


def test_no_feasible_design_skips_tcad(monkeypatch) -> None:
    install_common_nodes(monkeypatch)

    def optimize(_state):
        return {
            "optimization": {
                "status": "no_feasible_design",
                "evaluated_candidates": 256,
            },
            "error": "No feasible surrogate design was found.",
            "events": ["no feasible design"],
        }

    monkeypatch.setattr(design_agent, "optimize", optimize)

    result = design_agent.build_graph().invoke(initial_state())

    assert "No feasible surrogate design was found" in result["final_report"]
    assert any(
        event.startswith("Skipped TCAD verification")
        for event in result["events"]
    )


def test_verification_timeout_becomes_reportable_error(monkeypatch) -> None:
    async def timeout(*_args, **_kwargs):
        raise TimeoutError("verification exceeded 180 seconds")

    monkeypatch.setattr(design_agent, "call_mcp_tool", timeout)
    state = {
        **initial_state(),
        "targets": TARGETS.copy(),
        "optimization": successful_optimization(),
    }

    result = design_agent.verify(state)

    assert result["iteration"] == 1
    assert "verification exceeded 180 seconds" in result["error"]
    assert "verification" not in result


def test_openai_parse_failure_stops_graph(monkeypatch) -> None:
    def fail(_state):
        raise RuntimeError("OpenAI unavailable")

    monkeypatch.setattr(design_agent, "parse_requirements", fail)

    with pytest.raises(RuntimeError, match="OpenAI unavailable"):
        design_agent.build_graph().invoke(initial_state())
