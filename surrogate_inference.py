"""Reusable inference engine for the trained ID-VG surrogate."""

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_idvg_surrogate import IdVgSurrogate


ROOT = Path(__file__).parent
DEFAULT_MODEL_PATH = ROOT / "models" / "idvg_surrogate.pt"
TRAINING_DOMAIN = {
    "gate_length_nm": (40.0, 60.0),
    "oxide_thickness_nm": (1.0, 1.5),
    "halo_peak_doping_cm3": (1.0e19, 4.0e19),
    "junction_depth_nm": (20.0, 40.0),
    "gate_voltage_v": (0.0, 1.2),
}
TRAINED_DRAIN_VOLTAGES = (0.05, 1.2)


class SurrogatePredictor:
    """Warm-loaded PyTorch surrogate with domain checks and metric extraction."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path).resolve()
        checkpoint = torch.load(
            self.model_path, map_location="cpu", weights_only=True
        )
        self.model = IdVgSurrogate()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.x_mean = np.asarray(
            checkpoint["normalization"]["x_mean"], dtype=np.float32
        )
        self.x_std = np.asarray(
            checkpoint["normalization"]["x_std"], dtype=np.float32
        )
        self.y_mean = float(checkpoint["normalization"]["y_mean"])
        self.y_std = float(checkpoint["normalization"]["y_std"])
        self.test_metrics = checkpoint.get("metrics", {})

    @staticmethod
    def domain_warnings(
        *,
        gate_length_nm: float,
        oxide_thickness_nm: float,
        halo_peak_doping_cm3: float,
        junction_depth_nm: float,
        gate_voltages_v: list[float],
        drain_voltages_v: list[float],
    ) -> list[str]:
        values = {
            "gate_length_nm": gate_length_nm,
            "oxide_thickness_nm": oxide_thickness_nm,
            "halo_peak_doping_cm3": halo_peak_doping_cm3,
            "junction_depth_nm": junction_depth_nm,
        }
        warnings: list[str] = []
        for name, value in values.items():
            low, high = TRAINING_DOMAIN[name]
            if not low <= value <= high:
                warnings.append(
                    f"{name}={value:g} is outside training range [{low:g}, {high:g}]"
                )

        low_vg, high_vg = TRAINING_DOMAIN["gate_voltage_v"]
        if any(not low_vg <= value <= high_vg for value in gate_voltages_v):
            warnings.append(
                f"gate voltage is outside training range [{low_vg:g}, {high_vg:g}] V"
            )
        unsupported_vd = [
            value
            for value in drain_voltages_v
            if not any(
                math.isclose(value, trained, abs_tol=1.0e-12)
                for trained in TRAINED_DRAIN_VOLTAGES
            )
        ]
        if unsupported_vd:
            warnings.append(
                "drain voltage was trained only at "
                f"{list(TRAINED_DRAIN_VOLTAGES)} V; unsupported={unsupported_vd}"
            )
        return warnings

    @torch.no_grad()
    def predict_curves(
        self,
        *,
        gate_length_nm: float,
        oxide_thickness_nm: float,
        halo_peak_doping_cm3: float,
        junction_depth_nm: float,
        gate_voltages_v: list[float],
        drain_voltages_v: list[float],
    ) -> tuple[dict[float, list[float]], float]:
        features = np.asarray(
            [
                [
                    gate_length_nm,
                    oxide_thickness_nm,
                    math.log10(halo_peak_doping_cm3),
                    junction_depth_nm,
                    gate_voltage,
                    drain_voltage,
                ]
                for drain_voltage in drain_voltages_v
                for gate_voltage in gate_voltages_v
            ],
            dtype=np.float32,
        )
        normalized = (features - self.x_mean) / self.x_std
        started = time.perf_counter()
        predicted_normalized = self.model(torch.from_numpy(normalized)).numpy()
        latency_ms = (time.perf_counter() - started) * 1.0e3
        predicted_log = predicted_normalized * self.y_std + self.y_mean
        currents = 10.0**predicted_log

        points_per_curve = len(gate_voltages_v)
        curves = {
            drain_voltage: currents[
                index * points_per_curve : (index + 1) * points_per_curve
            ]
            .astype(float)
            .tolist()
            for index, drain_voltage in enumerate(drain_voltages_v)
        }
        return curves, latency_ms

    @staticmethod
    def extract_metrics(
        gate_voltages_v: list[float],
        low_current: list[float],
        high_current: list[float],
    ) -> dict[str, float]:
        vg = np.asarray(gate_voltages_v)
        low = np.asarray(low_current)
        high = np.asarray(high_current)

        gm = np.diff(low) / np.diff(vg)
        gm_index = int(np.argmax(gm))
        threshold_low = (
            0.5 * (vg[gm_index] + vg[gm_index + 1])
            - 0.5 * (low[gm_index] + low[gm_index + 1]) / gm[gm_index]
            - 0.025
        )

        sqrt_current = np.sqrt(np.maximum(high, 0.0))
        sqrt_slope = np.diff(sqrt_current) / np.diff(vg)
        sqrt_index = int(np.argmax(sqrt_slope))
        threshold_high = (
            0.5 * (vg[sqrt_index] + vg[sqrt_index + 1])
            - 0.5
            * (sqrt_current[sqrt_index] + sqrt_current[sqrt_index + 1])
            / sqrt_slope[sqrt_index]
        )
        local_ss = np.diff(vg) / np.diff(np.log10(high))
        ion = float(high[-1])
        ioff = float(high[0])
        return {
            "threshold_voltage_v": float(threshold_low),
            "ion_ua_per_um": ion,
            "ioff_ua_per_um": ioff,
            "ion_ioff_ratio": ion / ioff,
            "subthreshold_slope_mv_per_dec": float(np.min(local_ss) * 1.0e3),
            "dibl_mv_per_v": float(
                (threshold_low - threshold_high) / (1.2 - 0.05) * 1.0e3
            ),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "training_domain": TRAINING_DOMAIN,
            "trained_drain_voltages_v": TRAINED_DRAIN_VOLTAGES,
            "test_metrics": self.test_metrics,
        }
