"""Save p-n junction band diagrams (equilibrium vs forward bias) to assets/."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT = Path(__file__).parent / "assets" / "pn_junction_bands.png"


def draw_panel(ax, mode: str) -> None:
    x0, xmid, x1 = 0.0, 0.5, 1.0
    y_ec_p, y_ev_p = 1.35, 0.55
    y_ef = 0.45
    y_ec_n, y_ev_n = 0.55, -0.25
    y_ec_n_b, y_ev_n_b = 0.62, -0.18

    ax.axvspan(x0, xmid, color="#4c72b0", alpha=0.12)
    ax.axvspan(xmid, x1, color="#dd8452", alpha=0.12)
    ax.axvline(xmid, color="k", ls="--", lw=0.8, alpha=0.5)

    if mode == "equilibrium":
        xs = [x0, 0.35, xmid, 0.65, x1]
        ec = [y_ec_p, y_ec_p, (y_ec_p + y_ec_n) / 2, y_ec_n, y_ec_n]
        ev = [y_ev_p, y_ev_p, (y_ev_p + y_ev_n) / 2, y_ev_n, y_ev_n]
        ax.plot(xs, ec, color="#2e79b5", lw=2.2, label="$E_c$")
        ax.plot(xs, ev, color="#d08770", lw=2.2, label="$E_v$")
        ax.hlines(y_ef, x0, x1, colors="#599ce7", lw=2, ls="--", label="$E_F$ (flat)")
        ax.annotate(
            "",
            xy=(0.04, y_ev_p),
            xytext=(0.04, y_ev_n),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2),
        )
        ax.text(0.06, 0.05, "$qV_{bi}$", fontsize=10, color="0.35")
        title = "Equilibrium ($V_{applied}=0$ V)"
        note = "Voltmeter reads 0 V\nBands bent by $V_{bi}\\approx0.95$ V"
    else:
        xs = [x0, 0.38, xmid, 0.62, x1]
        ec = [y_ec_p, y_ec_p, (y_ec_p + y_ec_n_b) / 2, y_ec_n_b, y_ec_n_b]
        ev = [y_ev_p, y_ev_p, (y_ev_p + y_ev_n_b) / 2, y_ev_n_b, y_ev_n_b]
        ax.plot(xs, ec, color="#2e79b5", lw=2.2)
        ax.plot(xs, ev, color="#d08770", lw=2.2)
        y_efp, y_efn = 0.52, 0.38
        ax.hlines(y_efp, x0, xmid - 0.02, colors="#599ce7", lw=2, ls="--")
        ax.hlines(y_efn, xmid + 0.02, x1, colors="#599ce7", lw=2, ls="--")
        ax.plot([xmid - 0.02, xmid + 0.02], [y_efp, y_efn], color="#599ce7", lw=1.5)
        ax.text(0.05, y_efp + 0.04, "$E_{Fp}$", color="#599ce7", fontsize=10)
        ax.text(0.88, y_efn - 0.06, "$E_{Fn}$", color="#599ce7", fontsize=10)
        ax.annotate(
            "",
            xy=(-0.06, y_efn),
            xytext=(-0.06, y_efp),
            arrowprops=dict(arrowstyle="<->", color="#3fa266", lw=1.2),
        )
        ax.text(-0.11, 0.38, "$qV=0.5$ eV", fontsize=9, color="#3fa266", rotation=90, va="center")
        title = "Forward bias ($V_{applied}=0.5$ V on p)"
        note = "Terminal split $E_{Fp}-E_{Fn}=0.5$ eV\nRemaining bend $\\approx V_{bi}-V=0.45$ V"

    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.45, 1.55)
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(["p", "n"])
    ax.set_ylabel("Energy (arb.)")
    ax.set_title(title, fontsize=11)
    ax.text(0.5, -0.38, note, ha="center", fontsize=9, color="0.35")
    ax.set_yticks([])


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    draw_panel(axes[0], "equilibrium")
    draw_panel(axes[1], "bias")
    legend = [
        Line2D([0], [0], color="#2e79b5", lw=2.2, label="$E_c$ (conduction band)"),
        Line2D([0], [0], color="#d08770", lw=2.2, label="$E_v$ (valence band)"),
        Line2D([0], [0], color="#599ce7", lw=2, ls="--", label="$E_F$ / $E_{Fp}$ / $E_{Fn}$"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("P-N junction: Fermi level vs band bending (diode_1d)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
