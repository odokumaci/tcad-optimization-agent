"""Plan or run a configurable DEVSIM design-of-experiments dataset."""

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import devsim
import numpy as np
from devsim.python_packages.ramp import rampbias
from scipy.stats import qmc

from mos_2d import Biases, run_simulation
from mos_2d_model import MOSParameters


ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "doe_config.json"
ALLOWED_GRID_PARAMETERS = {
    "gate_length_nm",
    "oxide_thickness_nm",
    "junction_depth_nm",
    "bulk_doping_cm3",
    "halo_peak_doping_cm3",
    "halo_depth_nm",
    "halo_offset_nm",
    "halo_lateral_sigma_nm",
    "halo_vertical_sigma_nm",
    "mobility_scale",
}


def _nm(value: float) -> float:
    return value * 1.0e-7


def _um(value: float) -> float:
    return value * 1.0e-4


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "output_directory", "base_parameters", "bias_sweep"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")

    has_grid = "parameter_grid" in config
    has_sobol = "sobol_sampling" in config
    if has_grid == has_sobol:
        raise ValueError("Specify exactly one of parameter_grid or sobol_sampling")

    if has_grid:
        unknown = set(config["parameter_grid"]) - ALLOWED_GRID_PARAMETERS
        if unknown:
            raise ValueError(f"Unsupported grid parameters: {sorted(unknown)}")
        if not config["parameter_grid"]:
            raise ValueError("parameter_grid must not be empty")
        if any(not values for values in config["parameter_grid"].values()):
            raise ValueError("Every parameter grid entry must contain at least one value")
    else:
        sampling = config["sobol_sampling"]
        parameters = sampling.get("parameters", {})
        unknown = set(parameters) - ALLOWED_GRID_PARAMETERS
        if unknown:
            raise ValueError(f"Unsupported Sobol parameters: {sorted(unknown)}")
        samples = int(sampling.get("samples", 0))
        if samples <= 0 or samples & (samples - 1):
            raise ValueError("Sobol sample count must be a positive power of two")
        for name, bounds in parameters.items():
            if bounds["min"] >= bounds["max"]:
                raise ValueError(f"Invalid Sobol bounds for {name}")
            if bounds.get("scale", "linear") not in ("linear", "log10"):
                raise ValueError(f"Unsupported scale for {name}")

    sweep = config["bias_sweep"]
    if len(sweep["drain_voltages_v"]) != 2:
        raise ValueError("Exactly two drain voltages are required for DIBL extraction")
    if sweep["gate_step_v"] <= 0 or sweep["gate_stop_v"] <= sweep["gate_start_v"]:
        raise ValueError("Gate sweep limits or step are invalid")
    return config


def enumerate_cases(config: dict[str, Any]) -> list[dict[str, float]]:
    if "parameter_grid" in config:
        grid = config["parameter_grid"]
        names = list(grid)
        return [
            dict(zip(names, values, strict=True))
            for values in itertools.product(*(grid[name] for name in names))
        ]

    sampling = config["sobol_sampling"]
    definitions = sampling["parameters"]
    names = list(definitions)
    sampler = qmc.Sobol(
        d=len(names),
        scramble=True,
        seed=int(sampling.get("seed", 0)),
    )
    unit_samples = sampler.random_base2(
        m=int(np.log2(int(sampling["samples"])))
    )
    cases: list[dict[str, float]] = []
    for sample in unit_samples:
        case: dict[str, float] = {}
        for name, unit_value in zip(names, sample, strict=True):
            bounds = definitions[name]
            low = float(bounds["min"])
            high = float(bounds["max"])
            if bounds.get("scale", "linear") == "log10":
                value = 10.0 ** (np.log10(low) + unit_value * (np.log10(high) - np.log10(low)))
            else:
                value = low + unit_value * (high - low)
            case[name] = float(value)
        cases.append(case)
    return cases


def build_parameters(base: dict[str, float], case: dict[str, float]) -> MOSParameters:
    values = {**base, **case}
    params = MOSParameters(
        device_width=_um(values["device_width_um"]),
        gate_length=_nm(values["gate_length_nm"]),
        oxide_thickness=_nm(values["oxide_thickness_nm"]),
        gate_thickness=_nm(values["gate_thickness_nm"]),
        device_thickness=_um(values["device_depth_um"]),
        junction_depth=_nm(values["junction_depth_nm"]),
        bulk_doping=values["bulk_doping_cm3"],
        body_doping=values["body_doping_cm3"],
        source_doping=values["source_doping_cm3"],
        drain_doping=values["drain_doping_cm3"],
        gate_doping=values["gate_doping_cm3"],
        halo_peak_doping=values["halo_peak_doping_cm3"],
        halo_depth=_nm(values["halo_depth_nm"]),
        halo_lateral_offset=_nm(values["halo_offset_nm"]),
        halo_lateral_sigma=_nm(values["halo_lateral_sigma_nm"]),
        halo_vertical_sigma=_nm(values["halo_vertical_sigma_nm"]),
        mobility_scale=values["mobility_scale"],
        temperature=values["temperature_k"],
    )
    params.validate()
    return params


def gate_values(sweep: dict[str, Any]) -> np.ndarray:
    start = float(sweep["gate_start_v"])
    stop = float(sweep["gate_stop_v"])
    step = float(sweep["gate_step_v"])
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1)
    if not np.isclose(values[-1], stop):
        raise ValueError("Gate sweep range must be divisible by gate_step_v")
    return values


def drain_current(device: str) -> float:
    electron = devsim.get_contact_current(
        device=device, contact="drain", equation="ElectronContinuityEquation"
    )
    hole = devsim.get_contact_current(
        device=device, contact="drain", equation="HoleContinuityEquation"
    )
    return electron + hole


def run_gate_sweep(
    params: MOSParameters,
    drain_voltage: float,
    vg_values: np.ndarray,
    output_dir: Path,
    bias_step: float,
) -> np.ndarray:
    result = run_simulation(
        params=params,
        biases=Biases(gate=float(vg_values[-1]), drain=drain_voltage),
        output_dir=output_dir,
        bias_step=bias_step,
    )
    currents = [result["currents"]["drain"]["total"]]
    for voltage in vg_values[-2::-1]:
        rampbias(
            device=params.device,
            contact="gate",
            end_bias=float(voltage),
            step_size=min(bias_step, float(vg_values[1] - vg_values[0])),
            min_step=1.0e-4,
            max_iter=50,
            rel_error=1.0e-4,
            abs_error=1.0e30,
            callback=lambda _device: None,
        )
        currents.append(drain_current(params.device))
    return np.asarray(currents[::-1]) * 100.0


def extract_metrics(
    vg: np.ndarray,
    low_current: np.ndarray,
    high_current: np.ndarray,
    low_vd: float,
    high_vd: float,
) -> dict[str, float]:
    gm = np.diff(low_current) / np.diff(vg)
    gm_index = int(np.argmax(gm))
    midpoint_vg = 0.5 * (vg[gm_index] + vg[gm_index + 1])
    midpoint_id = 0.5 * (low_current[gm_index] + low_current[gm_index + 1])
    threshold_low = midpoint_vg - midpoint_id / gm[gm_index] - 0.5 * low_vd

    sqrt_current = np.sqrt(np.maximum(high_current, 0.0))
    sqrt_slope = np.diff(sqrt_current) / np.diff(vg)
    sqrt_index = int(np.argmax(sqrt_slope))
    midpoint_sqrt = 0.5 * (
        sqrt_current[sqrt_index] + sqrt_current[sqrt_index + 1]
    )
    threshold_high = (
        0.5 * (vg[sqrt_index] + vg[sqrt_index + 1])
        - midpoint_sqrt / sqrt_slope[sqrt_index]
    )

    valid = (high_current[:-1] > 0) & (high_current[1:] > 0)
    local_ss = np.diff(vg)[valid] / np.diff(np.log10(high_current))[valid]
    dibl = (threshold_low - threshold_high) / (high_vd - low_vd)
    ion = float(high_current[-1])
    ioff = float(high_current[0])
    return {
        "threshold_voltage_v": float(threshold_low),
        "threshold_voltage_high_vd_v": float(threshold_high),
        "ion_ua_per_um": ion,
        "ioff_ua_per_um": ioff,
        "ion_ioff_ratio": ion / ioff,
        "subthreshold_slope_mv_per_dec": float(np.min(local_ss) * 1.0e3),
        "dibl_mv_per_v": float(dibl * 1.0e3),
    }


def write_curve(
    path: Path,
    vg: np.ndarray,
    low_current: np.ndarray,
    high_current: np.ndarray,
    low_vd: float,
    high_vd: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gate_voltage_v",
                f"drain_current_vd_{low_vd:g}_ua_per_um",
                f"drain_current_vd_{high_vd:g}_ua_per_um",
            ]
        )
        writer.writerows(zip(vg, low_current, high_current, strict=True))


def run_case(
    case_id: str,
    case: dict[str, float],
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    params = build_parameters(config["base_parameters"], case)
    sweep = config["bias_sweep"]
    vg = gate_values(sweep)
    drain_voltages = sorted(float(value) for value in sweep["drain_voltages_v"])
    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    curves = {
        vd: run_gate_sweep(
            params,
            vd,
            vg,
            case_dir / f"vd_{vd:g}",
            float(sweep["bias_step_v"]),
        )
        for vd in drain_voltages
    }
    metrics = extract_metrics(
        vg,
        curves[drain_voltages[0]],
        curves[drain_voltages[1]],
        drain_voltages[0],
        drain_voltages[1],
    )
    write_curve(
        case_dir / "transfer_curves.csv",
        vg,
        curves[drain_voltages[0]],
        curves[drain_voltages[1]],
        drain_voltages[0],
        drain_voltages[1],
    )
    record = {
        "case_id": case_id,
        "status": "completed",
        **case,
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {"record": record, "parameters_cm": asdict(params), "bias_sweep": sweep},
            indent=2,
        ),
        encoding="utf-8",
    )
    return record


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for record in records:
        for name in record:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_plan(
    config: dict[str, Any], cases: list[dict[str, float]], config_path: Path
) -> None:
    print(f"Experiment: {config['name']}")
    print(f"Cases: {len(cases)}")
    print(json.dumps({"base_parameters": config["base_parameters"]}, indent=2))
    preview_count = min(len(cases), 20)
    for index, case in enumerate(cases[:preview_count], start=1):
        print(f"case_{index:04d}: {json.dumps(case, sort_keys=True)}")
    if len(cases) > preview_count:
        print(f"... {len(cases) - preview_count} additional cases")
    print(f"Dry run only. Review {config_path.name}, then execute with --run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute simulations. Without this flag, only print the plan.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    cases = enumerate_cases(config)
    for case in cases:
        build_parameters(config["base_parameters"], case)

    if not args.run:
        print_plan(config, cases, config_path)
        return

    output_root = Path(config["output_directory"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "config": config,
                "cases": cases,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_id = f"case_{index:04d}"
        metadata_path = output_root / case_id / "metadata.json"
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            if saved.get("record", {}).get("status") == "completed":
                print(f"[{index}/{len(cases)}] {case_id} already completed")
                records.append(saved["record"])
                write_summary(output_root / "summary.csv", records)
                continue
        print(f"[{index}/{len(cases)}] {case_id}")
        try:
            record = run_case(case_id, case, config, output_root)
        except Exception as error:
            record = {
                "case_id": case_id,
                "status": "failed",
                **case,
                "error": str(error),
            }
        records.append(record)
        write_summary(output_root / "summary.csv", records)


if __name__ == "__main__":
    main()
