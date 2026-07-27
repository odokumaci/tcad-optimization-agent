"""Plot 2D MOS mesh, doping, potential, and carriers."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation
import numpy as np

TEC_FILE = Path(__file__).parent / "output" / "mos_2d.tec"
MSH_FILE = Path(__file__).parent / "output" / "mos_2d_dd.msh"
OUT_DIR = Path(__file__).parent / "assets"
DEVICE = "mymos"
PLOT_REGIONS = ("bulk", "gate", "oxide")
CARRIER_REGIONS = ("bulk", "gate")

REGION_COLORS = {
    "air": "#f0f0f0",
    "bulk": "#dd8452",
    "gate": "#8172b3",
    "oxide": "#55a868",
}

CELL_VARS = 2


@dataclass
class SolutionZone:
    name: str
    x: np.ndarray
    y: np.ndarray
    potential: np.ndarray
    triangles: np.ndarray
    electrons: np.ndarray | None = None
    holes: np.ndarray | None = None


@dataclass
class Zone:
    name: str
    x: np.ndarray
    y: np.ndarray
    net_doping: np.ndarray
    at_contact: np.ndarray
    triangles: np.ndarray
    halo_doping: np.ndarray | None = None


def read_float_block(lines: list[str], pos: int, count: int) -> tuple[np.ndarray, int]:
    values: list[float] = []
    while len(values) < count:
        chunk = np.fromstring(lines[pos], sep=" ")
        if chunk.size == 0:
            raise ValueError(f"Expected {count} float values near line {pos + 1}")
        values.extend(chunk.tolist())
        pos += 1
    return np.array(values[:count], dtype=float), pos


def read_connectivity(lines: list[str], pos: int, n_elements: int) -> tuple[np.ndarray, int]:
    rows: list[np.ndarray] = []
    for i in range(n_elements):
        row = np.fromstring(lines[pos + i], sep=" ", dtype=int)
        if row.size != 3:
            raise ValueError(
                f"Expected 3 connectivity ints at line {pos + i + 1}, got {row.size}"
            )
        rows.append(row)
    return np.vstack(rows) - 1, pos + n_elements


def load_solution_zones() -> list[SolutionZone]:
    if not MSH_FILE.exists():
        raise FileNotFoundError(f"Run mos_2d.py first: {MSH_FILE} not found")

    import devsim
    from devsim import get_element_node_list, get_node_model_values

    sys.path.insert(0, str(MSH_FILE.parent))
    devsim.load_devices(file=str(MSH_FILE))
    import mos_2d_params  # noqa: F401

    zones: list[SolutionZone] = []
    for region in PLOT_REGIONS:
        x = np.array(get_node_model_values(device=DEVICE, region=region, name="x"))
        y = np.array(get_node_model_values(device=DEVICE, region=region, name="y"))
        potential = np.array(
            get_node_model_values(device=DEVICE, region=region, name="Potential")
        )
        electrons = holes = None
        if region in CARRIER_REGIONS:
            electrons = np.array(
                get_node_model_values(device=DEVICE, region=region, name="Electrons")
            )
            holes = np.array(
                get_node_model_values(device=DEVICE, region=region, name="Holes")
            )
        triangles = np.array(
            get_element_node_list(device=DEVICE, region=region), dtype=int
        )
        zones.append(
            SolutionZone(
                name=region,
                x=x,
                y=y,
                potential=potential,
                triangles=triangles,
                electrons=electrons,
                holes=holes,
            )
        )
    return zones


def parse_tecplot_zones(path: Path) -> list[Zone]:
    lines = path.read_text().splitlines()
    var_line = next(l for l in lines if l.startswith("VARIABLES"))
    names = re.findall(r'"([^"]+)"', var_line)
    if "NetDoping" not in names:
        raise ValueError("NetDoping not found in tecplot variables")

    net_idx = names.index("NetDoping")
    contact_idx = names.index("AtContactNode")
    halo_idx = names.index("HaloDoping") if "HaloDoping" in names else None
    nodal_vars = len(names) - CELL_VARS

    zones: list[Zone] = []
    zone_line_idxs = [i for i, l in enumerate(lines) if l.startswith("ZONE")]

    for z_idx, line_idx in enumerate(zone_line_idxs):
        zone_line = lines[line_idx]
        match = re.search(r'T="(\w+)" NODES=(\d+), ELEMENTS=(\d+)', zone_line)
        if not match:
            raise ValueError(f"Could not parse zone header: {zone_line}")

        name = match.group(1)
        n_nodes = int(match.group(2))
        n_elements = int(match.group(3))

        data_start = line_idx + 1
        pos = data_start

        blocks: list[np.ndarray] = []
        for _ in range(nodal_vars):
            values, pos = read_float_block(lines, pos, n_nodes)
            blocks.append(values)

        for _ in range(CELL_VARS):
            _, pos = read_float_block(lines, pos, n_elements)

        triangles, pos = read_connectivity(lines, pos, n_elements)

        zones.append(
            Zone(
                name=name,
                x=blocks[0],
                y=blocks[1],
                net_doping=blocks[net_idx],
                at_contact=blocks[contact_idx],
                triangles=triangles,
                halo_doping=blocks[halo_idx] if halo_idx is not None else None,
            )
        )

    return zones


def um(value_cm: np.ndarray) -> np.ndarray:
    return value_cm * 1e4


# Vertical window used for all "device" plots: from the top of the gate/oxide
# stack down to a shallow depth below the Si surface. This crops out the
# inert deep bulk (device_thickness = 1 um) so the gate/channel/junction
# region isn't squeezed into a sliver at the top of the figure.
DEVICE_DEPTH_UM = 0.2


def _device_ylim(zones: list) -> tuple[float, float]:
    """Top-of-gate-stack to shallow-depth-below-surface window, for plots that
    combine gate/oxide/bulk zones."""
    y_top = min(um(z.y).min() for z in zones)
    return (y_top, DEVICE_DEPTH_UM)


def _surface_ylim() -> tuple[float, float]:
    """Surface to shallow-depth-below-surface window, for bulk-only plots."""
    return (0.0, DEVICE_DEPTH_UM)


def _linear_levels(vmin: float, vmax: float, n: int = 15) -> np.ndarray:
    return np.linspace(vmin, vmax, n)


def _log_levels(vmin: float, vmax: float, n: int = 12) -> np.ndarray:
    return np.logspace(np.log10(vmin), np.log10(vmax), n)


def _apply_device_orientation(ax: plt.Axes) -> None:
    """Surface (gate, S/D) is at small y in the mesh; show it at the top."""
    ax.invert_yaxis()


def _plot_contour_field(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str,
    norm: mcolors.Normalize,
    levels: np.ndarray,
    title: str = "",
    cbar_label: str = "",
    cbar_format: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    label_lines: bool = False,
) -> plt.cm.ScalarMappable:
    tri = mtri.Triangulation(um(x), um(y), triangles)
    cf = ax.tricontourf(
        tri,
        values,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
    )
    cs = ax.tricontour(
        tri,
        values,
        levels=levels,
        colors="0.15",
        linewidths=0.4,
        alpha=0.55,
    )
    if label_lines and len(levels) <= 10:
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
    ax.set_aspect("equal")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    if title:
        ax.set_title(title)
    _apply_device_orientation(ax)
    return cf


def _add_colorbar(
    fig: plt.Figure,
    mappable: plt.cm.ScalarMappable,
    ax: plt.Axes,
    label: str,
    fmt: str | None = None,
) -> None:
    cbar = fig.colorbar(mappable, ax=ax, label=label)
    if fmt == "log":
        cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    elif fmt == "symlog":
        norm = mappable.norm
        max_magnitude = max(abs(norm.vmin), abs(norm.vmax))
        positive_ticks = np.logspace(
            np.log10(norm.linthresh), np.log10(max_magnitude), 7
        )
        cbar.set_ticks(
            np.concatenate((-positive_ticks[::-1], [0.0], positive_ticks))
        )
        cbar.formatter = FuncFormatter(
            lambda value, _: "0" if value == 0 else f"{value:.0e}"
        )
        cbar.update_ticks()


def plot_structure(zones: list[Zone], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for zone in zones:
        if zone.name == "air":
            continue
        color = REGION_COLORS.get(zone.name, "#cccccc")
        tri = mtri.Triangulation(um(zone.x), um(zone.y), zone.triangles)
        ax.tripcolor(
            tri,
            np.zeros(zone.x.size),
            cmap=mcolors.ListedColormap([color]),
            vmin=-0.5,
            vmax=0.5,
            edgecolors="0.35",
            linewidth=0.15,
        )

    for zone in zones:
        contacts = zone.at_contact > 0.5
        if not np.any(contacts):
            continue
        ax.scatter(
            um(zone.x[contacts]),
            um(zone.y[contacts]),
            s=8,
            c="crimson",
            marker="s",
            zorder=5,
        )

    non_air = [z for z in zones if z.name != "air"]
    bulk = next(z for z in zones if z.name == "bulk")
    ax.set_xlim(um(bulk.x.min()), um(bulk.x.max()))
    ax.set_ylim(*_device_ylim(non_air))
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("2D MOS structure (mesh + contacts)")
    ax.grid(True, alpha=0.25)
    _apply_device_orientation(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=REGION_COLORS[z.name], ec="0.35", lw=0.5)
        for z in zones
        if z.name != "air"
    ]
    labels = [z.name for z in zones if z.name != "air"] + ["contacts"]
    handles.append(plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="crimson", markersize=6))
    ax.legend(handles, labels, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _signed_doping_levels(d_max: float, n_per_side: int = 10) -> np.ndarray:
    mag = np.logspace(10, np.log10(d_max), n_per_side)
    return np.concatenate([-mag[::-1], mag])


def _doping_norm(zones: list[Zone]) -> mcolors.SymLogNorm:
    d_max = max(
        np.max(np.abs(z.net_doping))
        for z in zones
        if z.name != "air" and np.any(z.net_doping != 0)
    )
    return mcolors.SymLogNorm(linthresh=1e14, linscale=0.4, vmin=-d_max, vmax=d_max, base=10)


def plot_doping(zones: list[Zone], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    bulk = next(z for z in zones if z.name == "bulk")
    doping_zones = [z for z in zones if z.name in ("bulk", "gate")]
    norm = _doping_norm(zones)
    levels = _signed_doping_levels(norm.vmax)

    cf = None
    for zone in doping_zones:
        tri = mtri.Triangulation(um(zone.x), um(zone.y), zone.triangles)
        cf = ax.tricontourf(
            tri, zone.net_doping, levels=levels, cmap="RdBu", norm=norm, extend="both"
        )
        ax.tricontour(
            tri,
            zone.net_doping,
            levels=levels,
            colors="0.15",
            linewidths=0.4,
            alpha=0.5,
        )
    ax.set_xlim(um(bulk.x.min()), um(bulk.x.max()))
    ax.set_ylim(*_device_ylim(doping_zones))
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("NetDoping including gate (n+ blue, p− red)")
    _apply_device_orientation(ax)
    _add_colorbar(fig, cf, ax, "NetDoping (#/cm³)", fmt="symlog")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_structure_and_doping(zones: list[Zone], out: Path) -> None:
    """Structure plus NetDoping in both bulk silicon and polysilicon gate."""
    fig, ax = plt.subplots(figsize=(10, 7))

    bulk = next(z for z in zones if z.name == "bulk")
    non_air = [z for z in zones if z.name != "air"]
    norm = _doping_norm(zones)
    levels = _signed_doping_levels(norm.vmax)

    cf = None
    for zone in zones:
        if zone.name not in ("bulk", "gate"):
            continue
        tri = mtri.Triangulation(um(zone.x), um(zone.y), zone.triangles)
        cf = ax.tricontourf(
            tri, zone.net_doping, levels=levels, cmap="RdBu", norm=norm, extend="both"
        )
        ax.tricontour(
            tri,
            zone.net_doping,
            levels=levels,
            colors="0.15",
            linewidths=0.4,
            alpha=0.5,
        )

    for zone in zones:
        if zone.name != "oxide":
            continue
        color = REGION_COLORS.get(zone.name, "#cccccc")
        tri = mtri.Triangulation(um(zone.x), um(zone.y), zone.triangles)
        ax.tripcolor(
            tri,
            np.zeros(zone.x.size),
            cmap=mcolors.ListedColormap([color]),
            vmin=-0.5,
            vmax=0.5,
            edgecolors="0.3",
            linewidth=0.25,
        )

    for zone in zones:
        contacts = zone.at_contact > 0.5
        if not np.any(contacts):
            continue
        ax.scatter(
            um(zone.x[contacts]),
            um(zone.y[contacts]),
            s=8,
            c="crimson",
            marker="s",
            zorder=5,
        )

    ax.set_xlim(um(bulk.x.min()), um(bulk.x.max()))
    ax.set_ylim(*_device_ylim(non_air))
    ax.set_aspect("auto")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("2D MOS structure + NetDoping (n+ blue, p− red)")
    _apply_device_orientation(ax)

    _add_colorbar(fig, cf, ax, "NetDoping (#/cm³)", fmt="symlog")

    def contour_fill_color(value: float):
        """Use the exact discrete contour-bin color shown in the plot."""
        index = int(
            np.clip(
                np.searchsorted(levels, value, side="right"),
                0,
                len(cf.cvalues) - 1,
            )
        )
        return cf.to_rgba(cf.cvalues[index])

    region_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc=contour_fill_color(1e20),
            ec="0.35",
            lw=0.5,
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc=contour_fill_color(-1e16),
            ec="0.35",
            lw=0.5,
        ),
    ]
    region_labels = ["gate (n+, 1e20 cm⁻³)", "bulk/channel (p−, 1e16 cm⁻³)"]
    if bulk.halo_doping is not None:
        halo_peak = abs(float(np.min(bulk.halo_doping)))
        region_handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=contour_fill_color(-halo_peak),
                ec="0.35",
                lw=0.5,
            )
        )
        region_labels.append(f"halo pockets (p+, {halo_peak:.0e} cm⁻³ peak)")
    region_handles.append(
        plt.Rectangle(
            (0, 0), 1, 1, fc=REGION_COLORS["oxide"], ec="0.35", lw=0.5
        )
    )
    region_labels.append("oxide")
    region_handles.append(
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="crimson", markersize=6)
    )
    region_labels.append("contacts")
    ax.legend(region_handles, region_labels, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_potential(zones: list[SolutionZone], out: Path, bulk_out: Path) -> None:
    bulk = next(z for z in zones if z.name == "bulk")
    vals = [z.potential for z in zones]
    vmin = min(v.min() for v in vals)
    vmax = max(v.max() for v in vals)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    levels = _linear_levels(vmin, vmax)
    bias_title = "V_g = V_s = V_d = V_b = 0 V"

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = _plot_contour_field(
        ax,
        bulk.x,
        bulk.y,
        bulk.triangles,
        bulk.potential,
        cmap="RdBu_r",
        norm=norm,
        levels=levels,
        title=f"Bulk potential ({bias_title})",
        xlim=(um(bulk.x.min()), um(bulk.x.max())),
        ylim=_surface_ylim(),
        label_lines=True,
    )
    _add_colorbar(fig, cf, ax, "Potential (V)")
    fig.tight_layout()
    fig.savefig(bulk_out, dpi=150)
    plt.close(fig)

    device_ylim = _device_ylim(zones)
    fig, ax = plt.subplots(figsize=(10, 7))
    cf = None
    for zone in zones:
        cf = _plot_contour_field(
            ax,
            zone.x,
            zone.y,
            zone.triangles,
            zone.potential,
            cmap="RdBu_r",
            norm=norm,
            levels=levels,
            title="Device potential at equilibrium",
            xlim=(um(bulk.x.min()), um(bulk.x.max())),
            ylim=device_ylim,
            label_lines=True,
        )
    _add_colorbar(fig, cf, ax, "Potential (V)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _carrier_log_norm(values: list[np.ndarray]) -> mcolors.LogNorm:
    vmax = max(np.max(v) for v in values)
    return mcolors.LogNorm(vmin=1e10, vmax=vmax)


def _plot_carrier_panel(
    ax: plt.Axes,
    zones: list[SolutionZone],
    carrier: str,
    *,
    title: str,
    norm: mcolors.LogNorm,
    levels: np.ndarray,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.cm.ScalarMappable:
    cf = None
    for zone in zones:
        data = zone.electrons if carrier == "electrons" else zone.holes
        if data is None:
            continue
        cf = _plot_contour_field(
            ax,
            zone.x,
            zone.y,
            zone.triangles,
            np.clip(data, norm.vmin, None),
            cmap="magma",
            norm=norm,
            levels=levels,
            title=title,
            xlim=xlim,
            ylim=ylim,
        )
    if cf is None:
        raise ValueError(f"No carrier data for {carrier}")
    return cf


def plot_carriers(
    zones: list[SolutionZone],
    out: Path,
    bulk_electrons_out: Path,
    bulk_holes_out: Path,
) -> None:
    carrier_zones = [z for z in zones if z.name in CARRIER_REGIONS]
    bulk = next(z for z in zones if z.name == "bulk")
    xlim = (um(bulk.x.min()), um(bulk.x.max()))
    ylim = _device_ylim(carrier_zones)
    surface_ylim = _surface_ylim()
    bias_title = "V_g = V_s = V_d = V_b = 0 V"

    electron_norm = _carrier_log_norm([z.electrons for z in carrier_zones])
    hole_norm = _carrier_log_norm([z.holes for z in carrier_zones])
    electron_levels = _log_levels(electron_norm.vmin, electron_norm.vmax)
    hole_levels = _log_levels(hole_norm.vmin, hole_norm.vmax)

    for carrier, norm, levels, bulk_out, label in (
        ("electrons", electron_norm, electron_levels, bulk_electrons_out, "Electrons"),
        ("holes", hole_norm, hole_levels, bulk_holes_out, "Holes"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        cf = _plot_carrier_panel(
            ax,
            [bulk],
            carrier,
            title=f"Bulk {label.lower()} ({bias_title})",
            norm=norm,
            levels=levels,
            xlim=xlim,
            ylim=surface_ylim,
        )
        _add_colorbar(fig, cf, ax, f"{label} (#/cm³)", fmt="log")
        fig.tight_layout()
        fig.savefig(bulk_out, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = (
        (axes[0, 0], [bulk], "electrons", f"Bulk electrons ({bias_title})", electron_norm, electron_levels, "Electrons", xlim, surface_ylim),
        (axes[0, 1], [bulk], "holes", f"Bulk holes ({bias_title})", hole_norm, hole_levels, "Holes", xlim, surface_ylim),
        (axes[1, 0], carrier_zones, "electrons", "Device electrons (bulk + gate)", electron_norm, electron_levels, "Electrons", xlim, ylim),
        (axes[1, 1], carrier_zones, "holes", "Device holes (bulk + gate)", hole_norm, hole_levels, "Holes", xlim, ylim),
    )
    for ax, panel_zones, carrier, title, norm, levels, label, panel_xlim, panel_ylim in panels:
        cf = _plot_carrier_panel(
            ax,
            panel_zones,
            carrier,
            title=title,
            norm=norm,
            levels=levels,
            xlim=panel_xlim,
            ylim=panel_ylim,
        )
        _add_colorbar(fig, cf, ax, f"{label} (#/cm³)", fmt="log")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    if not TEC_FILE.exists():
        raise FileNotFoundError(f"Run mos_2d.py first: {TEC_FILE} not found")

    OUT_DIR.mkdir(exist_ok=True)
    zones = parse_tecplot_zones(TEC_FILE)

    struct_path = OUT_DIR / "mos_2d_structure.png"
    doping_path = OUT_DIR / "mos_2d_doping.png"
    struct_doping_path = OUT_DIR / "mos_2d_structure_doping.png"
    potential_path = OUT_DIR / "mos_2d_potential.png"
    bulk_potential_path = OUT_DIR / "mos_2d_bulk_potential.png"
    carriers_path = OUT_DIR / "mos_2d_carriers.png"
    bulk_electrons_path = OUT_DIR / "mos_2d_bulk_electrons_log.png"
    bulk_holes_path = OUT_DIR / "mos_2d_bulk_holes_log.png"
    plot_structure(zones, struct_path)
    plot_doping(zones, doping_path)
    plot_structure_and_doping(zones, struct_doping_path)
    solution = load_solution_zones()
    plot_potential(solution, potential_path, bulk_potential_path)
    plot_carriers(solution, carriers_path, bulk_electrons_path, bulk_holes_path)
    print(f"Wrote {struct_path}")
    print(f"Wrote {doping_path}")
    print(f"Wrote {struct_doping_path}")
    print(f"Wrote {potential_path}")
    print(f"Wrote {bulk_potential_path}")
    print(f"Wrote {carriers_path}")
    print(f"Wrote {bulk_electrons_path}")
    print(f"Wrote {bulk_holes_path}")


if __name__ == "__main__":
    main()
