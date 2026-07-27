"""Plot 1D diode structure and DEVSIM results from diode_1d.dat."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DAT_FILE = Path(__file__).with_name("diode_1d.dat")
OUT_DIR = Path(__file__).parent / "assets"


def parse_tecplot_block(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    lines = path.read_text().splitlines()
    var_line = next(l for l in lines if l.startswith("VARIABLES"))
    names = re.findall(r'"([^"]+)"', var_line)
    zone_idx = next(i for i, l in enumerate(lines) if l.startswith("ZONE"))
    zone = lines[zone_idx]
    n_nodes = int(re.search(r"NODES=(\d+)", zone).group(1))
    data_lines = lines[zone_idx + 1 : zone_idx + 1 + len(names)]
    arrays = {}
    for name, row in zip(names, data_lines):
        arrays[name] = np.fromstring(row, sep=" ", dtype=float)
        if arrays[name].size != n_nodes:
            raise ValueError(f"{name}: expected {n_nodes} values, got {arrays[name].size}")
    return names, arrays


def plot_structure(x_nm: np.ndarray, data: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[1, 2])

    ax0 = axes[0]
    x_max = float(x_nm.max())
    mid = x_max / 2.0
    ax0.set_xlim(-8, x_max + 8)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_xlabel("x (nm)")
    ax0.set_title("Device structure (p-n junction, 0.5 V on top contact)")
    ax0.axvspan(0, mid, color="#4c72b0", alpha=0.35, label="p-region (Acceptors)")
    ax0.axvspan(mid, x_max, color="#dd8452", alpha=0.35, label="n-region (Donors)")
    ax0.axvline(mid, color="k", ls="--", lw=0.8)
    ax0.axvline(0, color="k", lw=2)
    ax0.axvline(x_max, color="k", lw=2)
    ax0.text(-4, 0.5, "top\ncontact", ha="center", va="center", fontsize=9)
    ax0.text(x_max + 4, 0.5, "bot\ncontact", ha="center", va="center", fontsize=9)
    ax0.text(mid, 0.15, "junction", ha="center", fontsize=8)
    ax0.vlines(x_nm, 0, 0.12, colors="0.25", linewidth=0.35, alpha=0.8)
    ax0.text(
        x_max / 2,
        0.92,
        f"mesh: {len(x_nm)} nodes (refined at x = {mid:.0f} nm)",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    ax0.legend(loc="upper center", ncol=2, fontsize=8, frameon=False)

    ax1 = axes[1]
    ax1.semilogy(x_nm, data["Acceptors"], label="Acceptors")
    ax1.semilogy(x_nm, data["Donors"], label="Donors")
    mesh_y = np.full_like(x_nm, 2e15)
    ax1.plot(
        x_nm,
        mesh_y,
        ls="none",
        marker="|",
        color="0.45",
        markersize=7,
        alpha=0.6,
        label=f"mesh nodes ({len(x_nm)})",
    )
    ax1.axhline(1e18, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("Doping (#/cm³)")
    ax1.set_title("Doping profile")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_results(x_nm: np.ndarray, data: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(x_nm, data["Potential"], color="#55a868", lw=1.5)
    ax0.set_ylabel("Potential (V)")
    ax0.set_title("Drift-diffusion solution at V_top = 0.5 V")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.semilogy(x_nm, data["Electrons"], label="Electrons")
    ax1.semilogy(x_nm, data["Holes"], label="Holes")
    ax1.semilogy(x_nm, np.abs(data["NetDoping"]), ls="--", color="gray", label="|NetDoping|")
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("Density (#/cm³)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    if not DAT_FILE.exists():
        raise FileNotFoundError(f"Run diode_1d.py first: {DAT_FILE} not found")

    OUT_DIR.mkdir(exist_ok=True)
    _, data = parse_tecplot_block(DAT_FILE)
    x_nm = data["x"] * 1e7  # cm -> nm

    struct_path = OUT_DIR / "diode_1d_structure.png"
    results_path = OUT_DIR / "diode_1d_results.png"
    plot_structure(x_nm, data, struct_path)
    plot_results(x_nm, data, results_path)
    print(f"Wrote {struct_path}")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
