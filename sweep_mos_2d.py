"""Generate 50 nm physical-gate-length MOSFET transfer curves."""

import csv
from pathlib import Path

import devsim
import matplotlib.pyplot as plt
import numpy as np
from devsim.python_packages.ramp import rampbias

from mos_2d import Biases, run_simulation
from mos_2d_model import MOSParameters


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "output" / "sweep"
ASSET_DIR = ROOT / "assets"


def drain_current(device: str) -> float:
    return devsim.get_contact_current(
        device=device, contact="drain", equation="ElectronContinuityEquation"
    ) + devsim.get_contact_current(
        device=device, contact="drain", equation="HoleContinuityEquation"
    )


def sweep_gate(params: MOSParameters, vd: float, vg_values: np.ndarray) -> list[float]:
    run_dir = OUT_DIR / f"vd_{vd:g}"
    result = run_simulation(
        params=params,
        biases=Biases(gate=float(vg_values[-1]), drain=vd),
        output_dir=run_dir,
        bias_step=0.1,
    )
    currents = [result["currents"]["drain"]["total"]]
    for vg in vg_values[-2::-1]:
        rampbias(
            device=params.device,
            contact="gate",
            end_bias=float(vg),
            step_size=0.1,
            min_step=1.0e-4,
            max_iter=50,
            rel_error=1.0e-4,
            abs_error=1.0e30,
            callback=lambda _device: None,
        )
        currents.append(drain_current(params.device))
    return currents[::-1]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(exist_ok=True)
    params = MOSParameters()
    vg_values = np.round(np.arange(0.0, 1.21, 0.1), 10)
    drain_biases = (0.05, 1.2)
    curves = {vd: sweep_gate(params, vd, vg_values) for vd in drain_biases}
    curves_ua_per_um = {
        vd: np.asarray(values) * 100.0 for vd, values in curves.items()
    }

    csv_path = ASSET_DIR / "mos_2d_id_vg_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Vg_V", "Id_Vd_0.05_uA_per_um", "Id_Vd_1.2_uA_per_um"])
        writer.writerows(
            zip(vg_values, curves_ua_per_um[0.05], curves_ua_per_um[1.2], strict=True)
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for vd in drain_biases:
        label = f"Vd = {vd:g} V"
        axes[0].plot(vg_values, curves_ua_per_um[vd], marker="o", label=label)
        axes[1].semilogy(
            vg_values,
            np.maximum(np.abs(curves_ua_per_um[vd]), 1e-30),
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("Drain current (µA/µm)")
    axes[1].set_ylabel("|Drain current| (µA/µm)")
    for ax in axes:
        ax.set_xlabel("Gate voltage (V)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_title("Transfer curve — linear scale")
    axes[1].set_title("Transfer curve — logarithmic scale")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "mos_2d_id_vg_sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
