"""Plot threshold voltage versus physical gate length and oxide thickness."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).parent
SUMMARY = ROOT / "dataset" / "pilot" / "summary.csv"
OUTPUT = ROOT / "assets" / "mos_2d_vth_vs_geometry.png"


def main() -> None:
    with SUMMARY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    gate_lengths = sorted({float(row["gate_length_nm"]) for row in rows})
    oxide_thicknesses = sorted(
        {float(row["oxide_thickness_nm"]) for row in rows}
    )
    halo_levels = sorted({float(row["halo_peak_doping_cm3"]) for row in rows})
    x_grid, y_grid = np.meshgrid(gate_lengths, oxide_thicknesses)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = ("tab:blue", "tab:orange")
    legend_handles = []

    for halo, color in zip(halo_levels, colors, strict=True):
        lookup = {
            (float(row["gate_length_nm"]), float(row["oxide_thickness_nm"])): float(
                row["threshold_voltage_v"]
            )
            for row in rows
            if float(row["halo_peak_doping_cm3"]) == halo
        }
        vth_grid = np.array(
            [
                [lookup[(gate_length, oxide)] for gate_length in gate_lengths]
                for oxide in oxide_thicknesses
            ]
        )
        ax.plot_surface(
            x_grid,
            y_grid,
            vth_grid,
            color=color,
            alpha=0.45,
            edgecolor=color,
            linewidth=0.8,
        )
        ax.scatter(
            x_grid,
            y_grid,
            vth_grid,
            color=color,
            s=55,
            edgecolor="black",
            linewidth=0.4,
            depthshade=False,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                label=f"Halo = {halo / 1e19:g}×10¹⁹ cm⁻³",
                markerfacecolor=color,
            )
        )

    ax.set_xlabel("Physical gate length (nm)", labelpad=10)
    ax.set_ylabel("Physical oxide thickness (nm)", labelpad=10)
    ax.set_zlabel("Threshold voltage (V)", labelpad=10)
    ax.set_title("Threshold voltage versus gate length and oxide thickness")
    ax.set_xticks(gate_lengths)
    ax.set_yticks(oxide_thicknesses)
    ax.legend(handles=legend_handles, loc="upper right")
    ax.view_init(elev=25, azim=-55)
    fig.text(
        0.5,
        0.02,
        "Linear-extrapolation VTH at VD = 0.05 V · Source: dataset/pilot/summary.csv",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.08, top=0.92)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
