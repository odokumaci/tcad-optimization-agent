"""Plot the pilot DOE parameter space as a 3D scatter plot."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
SUMMARY = ROOT / "dataset" / "pilot" / "summary.csv"
OUTPUT = ROOT / "assets" / "mos_2d_doe_3d_scatter.png"


def main() -> None:
    with SUMMARY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    gate_length = np.array([float(row["gate_length_nm"]) for row in rows])
    oxide_thickness = np.array([float(row["oxide_thickness_nm"]) for row in rows])
    halo_doping = np.array(
        [float(row["halo_peak_doping_cm3"]) / 1.0e19 for row in rows]
    )
    ion = np.array([float(row["ion_ua_per_um"]) for row in rows])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(
        gate_length,
        oxide_thickness,
        halo_doping,
        c=ion,
        cmap="viridis",
        s=90,
        edgecolor="black",
        linewidth=0.5,
        depthshade=False,
    )

    for row, x, y, z in zip(
        rows, gate_length, oxide_thickness, halo_doping, strict=True
    ):
        ax.text(x, y, z + 0.025, row["case_id"].replace("case_", "C"), fontsize=8)

    ax.set_xlabel("Physical gate length (nm)", labelpad=10)
    ax.set_ylabel("Physical oxide thickness (nm)", labelpad=10)
    ax.set_zlabel("Halo peak doping (×10¹⁹ cm⁻³)", labelpad=10)
    ax.set_title("Pilot DOE parameter space colored by on-current")
    ax.set_xticks(sorted(set(gate_length)))
    ax.set_yticks(sorted(set(oxide_thickness)))
    ax.set_zticks(sorted(set(halo_doping)))
    ax.view_init(elev=24, azim=-55)

    colorbar = fig.colorbar(points, ax=ax, pad=0.12, shrink=0.75)
    colorbar.set_label("ION (µA/µm), VG = VD = 1.2 V")
    fig.text(
        0.5,
        0.02,
        "Source: dataset/pilot/summary.csv",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.92)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
