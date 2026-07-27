"""Constrained MOSFET design search using the trained surrogate."""

import math
import time
from typing import Any

import numpy as np
import torch
from scipy.stats import qmc

from surrogate_inference import SurrogatePredictor


GATE_VOLTAGES = np.round(np.arange(0.0, 1.21, 0.1), 10)
DRAIN_VOLTAGES = np.asarray((0.05, 1.2))


def generate_candidates(samples: int, seed: int) -> np.ndarray:
    """Generate [LG, TOX, halo, junction depth] Sobol candidates."""
    if samples <= 0 or samples & (samples - 1):
        raise ValueError("samples must be a positive power of two")
    unit = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(
        m=int(math.log2(samples))
    )
    candidates = np.empty_like(unit)
    candidates[:, 0] = 40.0 + unit[:, 0] * 20.0
    candidates[:, 1] = 1.0 + unit[:, 1] * 0.5
    candidates[:, 2] = 10.0 ** (
        math.log10(1.0e19)
        + unit[:, 2] * (math.log10(4.0e19) - math.log10(1.0e19))
    )
    candidates[:, 3] = 20.0 + unit[:, 3] * 20.0
    return candidates


@torch.no_grad()
def predict_candidate_metrics(
    predictor: SurrogatePredictor,
    candidates: np.ndarray,
    *,
    batch_size: int = 16384,
) -> dict[str, np.ndarray]:
    """Predict complete curves in batches and derive engineering metrics."""
    bias_pairs = np.asarray(
        [
            [gate_voltage, drain_voltage]
            for drain_voltage in DRAIN_VOLTAGES
            for gate_voltage in GATE_VOLTAGES
        ],
        dtype=np.float32,
    )
    design_features = np.column_stack(
        (
            candidates[:, 0],
            candidates[:, 1],
            np.log10(candidates[:, 2]),
            candidates[:, 3],
        )
    ).astype(np.float32)
    features = np.concatenate(
        (
            np.repeat(design_features, len(bias_pairs), axis=0),
            np.tile(bias_pairs, (len(candidates), 1)),
        ),
        axis=1,
    )
    normalized = (features - predictor.x_mean) / predictor.x_std
    predicted_parts: list[np.ndarray] = []
    for start in range(0, len(normalized), batch_size):
        values = predictor.model(
            torch.from_numpy(normalized[start : start + batch_size])
        ).numpy()
        predicted_parts.append(values)
    predicted_log = np.concatenate(predicted_parts) * predictor.y_std + predictor.y_mean
    curves = (10.0**predicted_log).reshape(
        len(candidates), len(DRAIN_VOLTAGES), len(GATE_VOLTAGES)
    )
    low = curves[:, 0, :]
    high = curves[:, 1, :]

    gm = np.diff(low, axis=1) / np.diff(GATE_VOLTAGES)
    gm_index = np.argmax(gm, axis=1)
    row = np.arange(len(candidates))
    threshold_low = (
        0.5 * (GATE_VOLTAGES[gm_index] + GATE_VOLTAGES[gm_index + 1])
        - 0.5
        * (low[row, gm_index] + low[row, gm_index + 1])
        / gm[row, gm_index]
        - 0.025
    )

    sqrt_current = np.sqrt(np.maximum(high, 0.0))
    sqrt_slope = np.diff(sqrt_current, axis=1) / np.diff(GATE_VOLTAGES)
    sqrt_index = np.argmax(sqrt_slope, axis=1)
    threshold_high = (
        0.5
        * (GATE_VOLTAGES[sqrt_index] + GATE_VOLTAGES[sqrt_index + 1])
        - 0.5
        * (
            sqrt_current[row, sqrt_index]
            + sqrt_current[row, sqrt_index + 1]
        )
        / sqrt_slope[row, sqrt_index]
    )
    local_ss = np.diff(GATE_VOLTAGES) / np.diff(np.log10(high), axis=1)
    ion = high[:, -1]
    ioff = high[:, 0]
    return {
        "threshold_voltage_v": threshold_low,
        "ion_ua_per_um": ion,
        "ioff_ua_per_um": ioff,
        "ion_ioff_ratio": ion / ioff,
        "subthreshold_slope_mv_per_dec": np.min(local_ss, axis=1) * 1.0e3,
        "dibl_mv_per_v": (
            (threshold_low - threshold_high)
            / (DRAIN_VOLTAGES[1] - DRAIN_VOLTAGES[0])
            * 1.0e3
        ),
    }


def candidate_record(
    candidates: np.ndarray,
    metrics: dict[str, np.ndarray],
    index: int,
    constraints: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "design": {
            "gate_length_nm": float(candidates[index, 0]),
            "oxide_thickness_nm": float(candidates[index, 1]),
            "halo_peak_doping_cm3": float(candidates[index, 2]),
            "junction_depth_nm": float(candidates[index, 3]),
        },
        "predicted_metrics": {
            name: float(values[index]) for name, values in metrics.items()
        },
        "constraint_margins": {
            "ioff_ua_per_um": float(
                constraints["max_ioff_ua_per_um"]
                - metrics["ioff_ua_per_um"][index]
            ),
            "subthreshold_slope_mv_per_dec": float(
                constraints["max_ss_mv_per_dec"]
                - metrics["subthreshold_slope_mv_per_dec"][index]
            ),
            "dibl_mv_per_v": float(
                constraints["max_dibl_mv_per_v"]
                - metrics["dibl_mv_per_v"][index]
            ),
            **(
                {
                    "threshold_voltage_v": float(
                        metrics["threshold_voltage_v"][index]
                        - constraints["min_threshold_voltage_v"]
                    )
                }
                if constraints["min_threshold_voltage_v"] is not None
                else {}
            ),
        },
    }


def optimize_design(
    predictor: SurrogatePredictor,
    *,
    max_ioff_ua_per_um: float = 0.001,
    max_ss_mv_per_dec: float = 85.0,
    max_dibl_mv_per_v: float = 50.0,
    min_threshold_voltage_v: float | None = None,
    samples: int = 4096,
    seed: int = 90,
    top_k: int = 5,
) -> dict[str, Any]:
    """Maximize ION subject to leakage and electrostatic constraints."""
    started = time.perf_counter()
    candidates = generate_candidates(samples, seed)
    metrics = predict_candidate_metrics(predictor, candidates)
    feasible = (
        (metrics["ioff_ua_per_um"] <= max_ioff_ua_per_um)
        & (metrics["subthreshold_slope_mv_per_dec"] <= max_ss_mv_per_dec)
        & (metrics["dibl_mv_per_v"] <= max_dibl_mv_per_v)
    )
    if min_threshold_voltage_v is not None:
        feasible &= metrics["threshold_voltage_v"] >= min_threshold_voltage_v
    feasible_indices = np.flatnonzero(feasible)
    constraints: dict[str, float | None] = {
        "max_ioff_ua_per_um": max_ioff_ua_per_um,
        "max_ss_mv_per_dec": max_ss_mv_per_dec,
        "max_dibl_mv_per_v": max_dibl_mv_per_v,
        "min_threshold_voltage_v": min_threshold_voltage_v,
    }

    if not len(feasible_indices):
        return {
            "status": "no_feasible_design",
            "objective": "maximize ion_ua_per_um",
            "constraints": constraints,
            "evaluated_candidates": samples,
            "feasible_candidates": 0,
            "search_seconds": time.perf_counter() - started,
            "recommendation": "Relax one or more constraints and retry.",
        }

    ranked = feasible_indices[
        np.argsort(metrics["ion_ua_per_um"][feasible_indices])[::-1]
    ]
    selected = ranked[: min(top_k, len(ranked))]
    results = [
        candidate_record(candidates, metrics, int(index), constraints)
        for index in selected
    ]
    return {
        "status": "success",
        "objective": "maximize ion_ua_per_um",
        "constraints": constraints,
        "evaluated_candidates": samples,
        "feasible_candidates": int(len(feasible_indices)),
        "search_seconds": time.perf_counter() - started,
        "best": results[0],
        "alternatives": results[1:],
        "verification_required": (
            "Verify the selected design with full DEVSIM before engineering use."
        ),
    }
