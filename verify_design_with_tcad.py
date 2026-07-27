"""Verify a surrogate-selected MOSFET design with full DEVSIM sweeps."""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


STARTUP_TIME = time.perf_counter()


def startup_log(message: str) -> None:
    print(
        f"[startup +{time.perf_counter() - STARTUP_TIME:.3f}s] {message}",
        flush=True,
    )


startup_log(
    f"Python={sys.executable}; cwd={Path.cwd()}; "
    f"DEVSIM_MATH_LIBS={os.getenv('DEVSIM_MATH_LIBS', '<unset>')}"
)
startup_log("Importing devsim")
import devsim
startup_log("Imported devsim")
startup_log("Importing httpx")
import httpx
startup_log("Imported httpx")
startup_log("Importing matplotlib")
import matplotlib.pyplot as plt
startup_log("Imported matplotlib")
startup_log("Importing numpy")
import numpy as np
startup_log("Imported numpy")
startup_log("Importing DEVSIM rampbias")
from devsim.python_packages.ramp import rampbias
startup_log("Imported DEVSIM rampbias")

startup_log("Importing metric extraction")
from generate_dataset import extract_metrics
startup_log("Imported metric extraction")
startup_log("Importing MOS solver")
from mos_2d import Biases, run_simulation
startup_log("Imported MOS solver")
startup_log("Importing MOS parameter model")
from mos_2d_model import MOSParameters
startup_log("Imported MOS parameter model")


ROOT = Path(__file__).parent
DEFAULT_OUTPUT = ROOT / "verification" / "optimized_design"
GATE_VOLTAGES = np.round(np.arange(0.0, 1.21, 0.1), 10)
DRAIN_VOLTAGES = (0.05, 1.2)


def _nm(value: float) -> float:
    return value * 1.0e-7


def drain_current(device: str) -> float:
    return devsim.get_contact_current(
        device=device, contact="drain", equation="ElectronContinuityEquation"
    ) + devsim.get_contact_current(
        device=device, contact="drain", equation="HoleContinuityEquation"
    )


def run_tcad_curve(
    params: MOSParameters,
    drain_voltage: float,
    output_dir: Path,
) -> np.ndarray:
    curve_started = time.perf_counter()
    print(f"[TCAD] Starting Id-Vg curve at Vd={drain_voltage:g} V", flush=True)
    result = run_simulation(
        params=params,
        biases=Biases(gate=float(GATE_VOLTAGES[-1]), drain=drain_voltage),
        output_dir=output_dir,
        bias_step=0.1,
    )
    print(
        f"[TCAD] Vd={drain_voltage:g} V, Vg={GATE_VOLTAGES[-1]:g} V "
        f"solved in {time.perf_counter() - curve_started:.1f} s",
        flush=True,
    )
    currents = [result["currents"]["drain"]["total"]]
    for gate_voltage in GATE_VOLTAGES[-2::-1]:
        point_started = time.perf_counter()
        print(
            f"[TCAD] Ramping Vd={drain_voltage:g} V to Vg={gate_voltage:g} V",
            flush=True,
        )
        rampbias(
            device=params.device,
            contact="gate",
            end_bias=float(gate_voltage),
            step_size=0.1,
            min_step=1.0e-4,
            max_iter=50,
            rel_error=1.0e-4,
            abs_error=1.0e30,
            callback=lambda _device: None,
        )
        currents.append(drain_current(params.device))
        print(
            f"[TCAD] Vd={drain_voltage:g} V, Vg={gate_voltage:g} V "
            f"solved in {time.perf_counter() - point_started:.1f} s",
            flush=True,
        )
    print(
        f"[TCAD] Completed Vd={drain_voltage:g} V curve in "
        f"{time.perf_counter() - curve_started:.1f} s",
        flush=True,
    )
    return np.asarray(currents[::-1]) * 100.0


def compare_metrics(
    tcad: dict[str, float], surrogate: dict[str, float]
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for name, actual in tcad.items():
        predicted = surrogate[name]
        comparison[name] = {
            "tcad": actual,
            "surrogate": predicted,
            "absolute_error": predicted - actual,
            "absolute_percentage_error": (
                abs(predicted - actual) / abs(actual) * 100.0
                if actual != 0
                else float("nan")
            ),
        }
    return comparison


def constraint_results(
    metrics: dict[str, float],
    *,
    max_ioff_ua_per_um: float,
    max_ss_mv_per_dec: float,
    max_dibl_mv_per_v: float,
) -> dict[str, Any]:
    checks = {
        "ioff": {
            "value": metrics["ioff_ua_per_um"],
            "limit": max_ioff_ua_per_um,
            "passed": metrics["ioff_ua_per_um"] <= max_ioff_ua_per_um,
        },
        "subthreshold_slope": {
            "value": metrics["subthreshold_slope_mv_per_dec"],
            "limit": max_ss_mv_per_dec,
            "passed": (
                metrics["subthreshold_slope_mv_per_dec"] <= max_ss_mv_per_dec
            ),
        },
        "dibl": {
            "value": metrics["dibl_mv_per_v"],
            "limit": max_dibl_mv_per_v,
            "passed": metrics["dibl_mv_per_v"] <= max_dibl_mv_per_v,
        },
    }
    return {
        "all_passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def write_curves(
    path: Path,
    tcad_curves: dict[float, np.ndarray],
    surrogate_curves: dict[float, list[float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gate_voltage_v",
                "tcad_id_vd_0.05_ua_per_um",
                "surrogate_id_vd_0.05_ua_per_um",
                "tcad_id_vd_1.2_ua_per_um",
                "surrogate_id_vd_1.2_ua_per_um",
            ]
        )
        writer.writerows(
            zip(
                GATE_VOLTAGES,
                tcad_curves[0.05],
                surrogate_curves[0.05],
                tcad_curves[1.2],
                surrogate_curves[1.2],
                strict=True,
            )
        )


def plot_curves(
    path: Path,
    tcad_curves: dict[float, np.ndarray],
    surrogate_curves: dict[float, list[float]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for drain_voltage in DRAIN_VOLTAGES:
        axes[0].plot(
            GATE_VOLTAGES,
            tcad_curves[drain_voltage],
            marker="o",
            label=f"TCAD VD={drain_voltage:g} V",
        )
        axes[0].plot(
            GATE_VOLTAGES,
            surrogate_curves[drain_voltage],
            linestyle="--",
            label=f"Surrogate VD={drain_voltage:g} V",
        )
        axes[1].semilogy(
            GATE_VOLTAGES,
            tcad_curves[drain_voltage],
            marker="o",
            label=f"TCAD VD={drain_voltage:g} V",
        )
        axes[1].semilogy(
            GATE_VOLTAGES,
            surrogate_curves[drain_voltage],
            linestyle="--",
            label=f"Surrogate VD={drain_voltage:g} V",
        )
    axes[0].set_title("Linear scale")
    axes[1].set_title("Logarithmic scale")
    for axis in axes:
        axis.set_xlabel("Gate voltage (V)")
        axis.set_ylabel("|Drain current| (µA/µm)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Optimized design: surrogate versus DEVSIM")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def request_surrogate_curves(
    api_url: str,
    *,
    gate_length_nm: float,
    oxide_thickness_nm: float,
    halo_peak_doping_cm3: float,
    junction_depth_nm: float,
) -> tuple[dict[float, list[float]], float]:
    response = httpx.post(
        f"{api_url.rstrip('/')}/predict/curve",
        json={
            "gate_length_nm": gate_length_nm,
            "oxide_thickness_nm": oxide_thickness_nm,
            "halo_peak_doping_cm3": halo_peak_doping_cm3,
            "junction_depth_nm": junction_depth_nm,
            "gate_voltages_v": GATE_VOLTAGES.tolist(),
            "drain_voltages_v": list(DRAIN_VOLTAGES),
        },
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    curves = {
        float(curve["drain_voltage_v"]): [
            float(point["drain_current_ua_per_um"])
            for point in curve["points"]
        ]
        for curve in payload["curves"]
    }
    return curves, float(payload["model_inference_latency_ms"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-length-nm", type=float, default=41.88142841681838)
    parser.add_argument(
        "--oxide-thickness-nm", type=float, default=1.1511103687807918
    )
    parser.add_argument(
        "--halo-peak-doping-cm3", type=float, default=3.95188971963523e19
    )
    parser.add_argument(
        "--junction-depth-nm", type=float, default=24.214158467948437
    )
    parser.add_argument("--max-ioff-ua-per-um", type=float, default=0.001)
    parser.add_argument("--max-ss-mv-per-dec", type=float, default=85.0)
    parser.add_argument("--max-dibl-mv-per-v", type=float, default=50.0)
    parser.add_argument(
        "--api-url",
        default=os.getenv("SURROGATE_API_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    params = MOSParameters(
        gate_length=_nm(args.gate_length_nm),
        oxide_thickness=_nm(args.oxide_thickness_nm),
        junction_depth=_nm(args.junction_depth_nm),
        halo_peak_doping=args.halo_peak_doping_cm3,
    )
    tcad_curves = {
        drain_voltage: run_tcad_curve(
            params,
            drain_voltage,
            output_dir / f"tcad_vd_{drain_voltage:g}",
        )
        for drain_voltage in DRAIN_VOLTAGES
    }
    surrogate_curves, inference_latency_ms = request_surrogate_curves(
        args.api_url,
        gate_length_nm=args.gate_length_nm,
        oxide_thickness_nm=args.oxide_thickness_nm,
        halo_peak_doping_cm3=args.halo_peak_doping_cm3,
        junction_depth_nm=args.junction_depth_nm,
    )

    tcad_metrics = extract_metrics(
        GATE_VOLTAGES,
        tcad_curves[0.05],
        tcad_curves[1.2],
        0.05,
        1.2,
    )
    surrogate_metrics = extract_metrics(
        GATE_VOLTAGES,
        np.asarray(surrogate_curves[0.05]),
        np.asarray(surrogate_curves[1.2]),
        0.05,
        1.2,
    )
    constraints = constraint_results(
        tcad_metrics,
        max_ioff_ua_per_um=args.max_ioff_ua_per_um,
        max_ss_mv_per_dec=args.max_ss_mv_per_dec,
        max_dibl_mv_per_v=args.max_dibl_mv_per_v,
    )
    report = {
        "design": {
            "gate_length_nm": args.gate_length_nm,
            "oxide_thickness_nm": args.oxide_thickness_nm,
            "halo_peak_doping_cm3": args.halo_peak_doping_cm3,
            "junction_depth_nm": args.junction_depth_nm,
        },
        "model_parameters": asdict(params),
        "tcad_metrics": tcad_metrics,
        "surrogate_metrics": surrogate_metrics,
        "metric_comparison": compare_metrics(tcad_metrics, surrogate_metrics),
        "tcad_constraint_validation": constraints,
        "surrogate_inference_latency_ms": inference_latency_ms,
    }
    (output_dir / "verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_curves(
        output_dir / "verification_curves.csv",
        tcad_curves,
        surrogate_curves,
    )
    plot_curves(
        output_dir / "verification_curves.png",
        tcad_curves,
        surrogate_curves,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
