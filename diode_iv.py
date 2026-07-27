"""Sweep top contact bias and record I-V for the 1D diode."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from devsim import get_contact_current, get_parameter, set_parameter, solve

import devsim.python_packages.simple_physics as simple_physics
import diode_common

DEVICE = "MyDevice"
REGION = "MyRegion"
TOP = "top"
ECE = "ElectronContinuityEquation"
HCE = "HoleContinuityEquation"

V_START = -0.5
V_STOP = 0.8
V_STEP = 0.05

OUT_CSV = Path(__file__).with_name("diode_iv.csv")
OUT_PNG = Path(__file__).parent / "assets" / "diode_iv.png"


def contact_current(device: str, contact: str) -> tuple[float, float, float]:
    i_e = get_contact_current(device=device, contact=contact, equation=ECE)
    i_h = get_contact_current(device=device, contact=contact, equation=HCE)
    return i_e, i_h, i_e + i_h


def setup_device() -> None:
    diode_common.CreateMesh(device=DEVICE, region=REGION)
    diode_common.SetParameters(device=DEVICE, region=REGION)
    set_parameter(device=DEVICE, region=REGION, name="taun", value=1e-8)
    set_parameter(device=DEVICE, region=REGION, name="taup", value=1e-8)
    diode_common.SetNetDoping(device=DEVICE, region=REGION)
    diode_common.InitialSolution(DEVICE, REGION)
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)
    diode_common.DriftDiffusionInitialSolution(DEVICE, REGION)
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)


def sweep_bias() -> list[dict[str, float]]:
    bias_name = simple_physics.GetContactBiasName(TOP)
    rows: list[dict[str, float]] = []

    v = V_START
    while v <= V_STOP + 1e-12:
        set_parameter(device=DEVICE, name=bias_name, value=v)
        solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
        i_e, i_h, i_total = contact_current(DEVICE, TOP)
        v_read = float(get_parameter(device=DEVICE, name=bias_name))
        rows.append(
            {
                "V_top_V": v_read,
                "I_e_A": i_e,
                "I_h_A": i_h,
                "I_total_A": i_total,
            }
        )
        print(f"V={v_read:7.3f} V  I={i_total:.6e} A")
        v += V_STEP

    return rows


def save_csv(rows: list[dict[str, float]]) -> None:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict[str, float]]) -> None:
    OUT_PNG.parent.mkdir(exist_ok=True)
    v = [r["V_top_V"] for r in rows]
    i = [r["I_total_A"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(v, i, "o-", ms=3)
    axes[0].set_xlabel("V_top (V)")
    axes[0].set_ylabel("I_top (A)")
    axes[0].set_title("Linear I-V")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axvline(0, color="k", lw=0.5)

    axes[1].semilogy(v, [abs(x) for x in i], "o-", ms=3)
    axes[1].set_xlabel("V_top (V)")
    axes[1].set_ylabel("|I_top| (A)")
    axes[1].set_title("Log |I| vs V")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    setup_device()
    rows = sweep_bias()
    save_csv(rows)
    save_plot(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
