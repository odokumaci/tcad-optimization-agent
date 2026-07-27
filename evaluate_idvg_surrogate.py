"""Validate surrogate-derived device metrics on held-out TCAD devices."""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_idvg_surrogate import IdVgSurrogate


ROOT = Path(__file__).parent
DEFAULT_DATASET = ROOT / "dataset" / "sobol_256"
DEFAULT_MODEL = ROOT / "models" / "idvg_surrogate.pt"


def load_summary(dataset_dir: Path) -> dict[str, dict[str, str]]:
    with (dataset_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return {
            row["case_id"]: row
            for row in csv.DictReader(handle)
            if row["status"] == "completed"
        }


def load_curve(
    dataset_dir: Path, case_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = dataset_dir / case_id / "transfer_curves.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    vg = np.asarray([float(row["gate_voltage_v"]) for row in rows])
    low = np.asarray(
        [float(row["drain_current_vd_0.05_ua_per_um"]) for row in rows]
    )
    high = np.asarray(
        [float(row["drain_current_vd_1.2_ua_per_um"]) for row in rows]
    )
    return vg, low, high


def extract_metrics(
    vg: np.ndarray,
    low_current: np.ndarray,
    high_current: np.ndarray,
) -> dict[str, float]:
    gm = np.diff(low_current) / np.diff(vg)
    gm_index = int(np.argmax(gm))
    threshold_low = (
        0.5 * (vg[gm_index] + vg[gm_index + 1])
        - 0.5 * (low_current[gm_index] + low_current[gm_index + 1])
        / gm[gm_index]
        - 0.025
    )

    sqrt_current = np.sqrt(np.maximum(high_current, 0.0))
    sqrt_slope = np.diff(sqrt_current) / np.diff(vg)
    sqrt_index = int(np.argmax(sqrt_slope))
    threshold_high = (
        0.5 * (vg[sqrt_index] + vg[sqrt_index + 1])
        - 0.5
        * (sqrt_current[sqrt_index] + sqrt_current[sqrt_index + 1])
        / sqrt_slope[sqrt_index]
    )

    positive = (high_current[:-1] > 0) & (high_current[1:] > 0)
    local_ss = np.diff(vg)[positive] / np.diff(np.log10(high_current))[positive]
    ion = float(high_current[-1])
    ioff = float(high_current[0])
    return {
        "threshold_voltage_v": float(threshold_low),
        "ion_ua_per_um": ion,
        "ioff_ua_per_um": ioff,
        "log10_ioff_ua_per_um": math.log10(ioff),
        "subthreshold_slope_mv_per_dec": float(np.min(local_ss) * 1.0e3),
        "dibl_mv_per_v": float(
            (threshold_low - threshold_high) / (1.2 - 0.05) * 1.0e3
        ),
    }


def build_features(
    device_row: dict[str, str], vg: np.ndarray
) -> np.ndarray:
    base = [
        float(device_row["gate_length_nm"]),
        float(device_row["oxide_thickness_nm"]),
        math.log10(float(device_row["halo_peak_doping_cm3"])),
        float(device_row["junction_depth_nm"]),
    ]
    return np.asarray(
        [
            [*base, gate_voltage, drain_voltage]
            for drain_voltage in (0.05, 1.2)
            for gate_voltage in vg
        ],
        dtype=np.float32,
    )


@torch.no_grad()
def predict_curves(
    model: IdVgSurrogate,
    features: np.ndarray,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    normalization = checkpoint["normalization"]
    mean = np.asarray(normalization["x_mean"], dtype=np.float32)
    std = np.asarray(normalization["x_std"], dtype=np.float32)
    normalized = (features - mean) / std
    predicted_normalized = (
        model(torch.from_numpy(normalized).to(device)).cpu().numpy()
    )
    predicted_log = (
        predicted_normalized * float(normalization["y_std"])
        + float(normalization["y_mean"])
    )
    predicted_current = 10.0**predicted_log
    points_per_curve = features.shape[0] // 2
    return (
        predicted_current[:points_per_curve],
        predicted_current[points_per_curve:],
    )


def metric_statistics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    residual = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - float(np.sum(residual**2)) / denominator,
    }


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_metric_parity(
    records: list[dict[str, Any]],
    output_path: Path,
    statistics: dict[str, dict[str, float]],
) -> None:
    plots = (
        ("threshold_voltage_v", "Threshold voltage (V)", False),
        ("ion_ua_per_um", "ION (µA/µm)", False),
        ("ioff_ua_per_um", "IOFF (µA/µm)", True),
        ("subthreshold_slope_mv_per_dec", "SS (mV/dec)", False),
        ("dibl_mv_per_v", "DIBL (mV/V)", False),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (metric, label, logarithmic) in zip(axes.flat, plots, strict=False):
        actual = np.asarray([row[f"actual_{metric}"] for row in records])
        predicted = np.asarray([row[f"predicted_{metric}"] for row in records])
        low = float(min(actual.min(), predicted.min()))
        high = float(max(actual.max(), predicted.max()))
        ax.scatter(actual, predicted, s=28, alpha=0.7)
        ax.plot([low, high], [low, high], color="black", linestyle="--")
        if logarithmic:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(f"TCAD {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(
            f"{label}\nMAE={statistics[metric]['mae']:.3g}, "
            f"R²={statistics[metric]['r2']:.4f}"
        )
        ax.grid(True, alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("Held-out engineering metric validation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


@torch.no_grad()
def benchmark_inference(
    model: IdVgSurrogate,
    normalized_features: np.ndarray,
    device: torch.device,
    repeats: int = 2000,
) -> float:
    inputs = torch.from_numpy(normalized_features).to(device)
    for _ in range(20):
        model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats


def benchmark_model_load(
    model_path: Path,
    normalized_features: np.ndarray,
    repeats: int = 20,
) -> float:
    inputs = torch.from_numpy(normalized_features)
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
        model = IdVgSurrogate()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            model(inputs)
        durations.append(time.perf_counter() - started)
    return float(np.median(durations))


def benchmark_process_cold_start(
    model_path: Path,
    repeats: int = 5,
) -> float:
    worker = ROOT / "surrogate_cold_start_worker.py"
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(worker),
                "--model-path",
                str(model_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        durations.append(time.perf_counter() - started)
    return float(np.median(durations))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    model_path = args.model_path.resolve()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IdVgSurrogate()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    summary = load_summary(dataset_dir)
    test_ids = checkpoint["split_case_ids"]["test"]
    records: list[dict[str, Any]] = []
    first_features: np.ndarray | None = None
    for case_id in test_ids:
        vg, actual_low, actual_high = load_curve(dataset_dir, case_id)
        features = build_features(summary[case_id], vg)
        if first_features is None:
            first_features = features
        predicted_low, predicted_high = predict_curves(
            model, features, checkpoint, device
        )
        actual_metrics = extract_metrics(vg, actual_low, actual_high)
        predicted_metrics = extract_metrics(vg, predicted_low, predicted_high)
        record: dict[str, Any] = {"case_id": case_id}
        for metric, value in actual_metrics.items():
            record[f"actual_{metric}"] = value
            record[f"predicted_{metric}"] = predicted_metrics[metric]
        records.append(record)

    metric_names = (
        "threshold_voltage_v",
        "ion_ua_per_um",
        "ioff_ua_per_um",
        "log10_ioff_ua_per_um",
        "subthreshold_slope_mv_per_dec",
        "dibl_mv_per_v",
    )
    statistics = {
        metric: metric_statistics(
            np.asarray([row[f"actual_{metric}"] for row in records]),
            np.asarray([row[f"predicted_{metric}"] for row in records]),
        )
        for metric in metric_names
    }

    assert first_features is not None
    normalization = checkpoint["normalization"]
    normalized_features = (
        first_features
        - np.asarray(normalization["x_mean"], dtype=np.float32)
    ) / np.asarray(normalization["x_std"], dtype=np.float32)
    inference_seconds = benchmark_inference(
        model, normalized_features.astype(np.float32), device
    )
    model_load_seconds = benchmark_model_load(
        model_path, normalized_features.astype(np.float32)
    )
    process_cold_start_seconds = benchmark_process_cold_start(model_path)
    tcad_seconds = float(
        np.mean([float(summary[case_id]["runtime_seconds"]) for case_id in test_ids])
    )
    benchmark = {
        "inference_device": str(device),
        "curve_points": int(first_features.shape[0]),
        "warm_inference_seconds": inference_seconds,
        "model_load_and_first_inference_seconds": model_load_seconds,
        "process_cold_start_seconds": process_cold_start_seconds,
        "mean_tcad_seconds_per_two_curve_simulation": tcad_seconds,
        "warm_inference_speedup": tcad_seconds / inference_seconds,
        "model_load_speedup": tcad_seconds / model_load_seconds,
        "process_cold_start_speedup": tcad_seconds / process_cold_start_seconds,
    }

    model_dir = model_path.parent
    write_records(model_dir / "idvg_surrogate_metric_validation.csv", records)
    report = {
        "held_out_devices": len(test_ids),
        "metric_statistics": statistics,
        "runtime_benchmark": benchmark,
    }
    (model_dir / "idvg_surrogate_engineering_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    plot_metric_parity(
        records,
        ROOT / "assets" / "surrogate_engineering_metric_parity.png",
        statistics,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
