"""Parameterized 2D MOSFET mesh and doping model."""

from dataclasses import asdict, dataclass
from pathlib import Path

import devsim


@dataclass(frozen=True)
class MOSParameters:
    """MOSFET process/geometry parameters.

    DEVSIM uses centimetres for length and cm^-3 for doping.
    """

    device: str = "mymos"
    mesh: str = "mos"
    device_width: float = 1.0e-4
    gate_length: float = 5.0e-6
    air_thickness: float = 1.0e-7
    oxide_thickness: float = 1.2e-7
    gate_thickness: float = 1.0e-5
    device_thickness: float = 1.0e-4
    junction_depth: float = 3.0e-6
    lateral_diffusion_decay: float = 1.0e-20
    vertical_diffusion_decay: float = 1.0e-10
    bulk_doping: float = -1.0e16
    body_doping: float = -1.0e19
    source_doping: float = 1.0e20
    drain_doping: float = 1.0e20
    gate_doping: float = 1.0e20
    halo_peak_doping: float = 3.0e19
    halo_depth: float = 3.0e-6
    halo_lateral_offset: float = 1.0e-6
    halo_lateral_sigma: float = 1.5e-6
    halo_vertical_sigma: float = 1.0e-6
    mobility_scale: float = 0.1
    temperature: float = 300.0

    def validate(self) -> None:
        positive_lengths = {
            "device_width": self.device_width,
            "gate_length": self.gate_length,
            "air_thickness": self.air_thickness,
            "oxide_thickness": self.oxide_thickness,
            "gate_thickness": self.gate_thickness,
            "device_thickness": self.device_thickness,
            "junction_depth": self.junction_depth,
            "lateral_diffusion_decay": self.lateral_diffusion_decay,
            "vertical_diffusion_decay": self.vertical_diffusion_decay,
            "halo_depth": self.halo_depth,
            "halo_lateral_offset": self.halo_lateral_offset,
            "halo_lateral_sigma": self.halo_lateral_sigma,
            "halo_vertical_sigma": self.halo_vertical_sigma,
        }
        for name, value in positive_lengths.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.gate_length >= self.device_width:
            raise ValueError("gate_length must be smaller than device_width")
        if self.junction_depth >= self.device_thickness:
            raise ValueError("junction_depth must be smaller than device_thickness")
        if self.halo_depth >= self.device_thickness:
            raise ValueError("halo_depth must be smaller than device_thickness")
        if self.halo_lateral_offset >= 0.5 * self.gate_length:
            raise ValueError("halo_lateral_offset must be less than half the gate length")
        if self.bulk_doping >= 0 or self.body_doping >= 0:
            raise ValueError("bulk_doping and body_doping must be negative (p-type)")
        if min(self.source_doping, self.drain_doping, self.gate_doping) <= 0:
            raise ValueError("source, drain, and gate doping must be positive (n-type)")
        if self.halo_peak_doping <= 0:
            raise ValueError("halo_peak_doping must be a positive p-type magnitude")
        if self.mobility_scale <= 0:
            raise ValueError("mobility_scale must be positive")


def parameters_to_dict(params: MOSParameters) -> dict[str, float | str]:
    return asdict(params)


def create_mos_device(
    params: MOSParameters,
    output_dir: Path | str | None = None,
    *,
    write_structure: bool = True,
) -> dict[str, float]:
    """Create a fresh DEVSIM device and return its derived geometry."""
    params.validate()
    devsim.reset_devsim()

    output_dir = (
        Path(output_dir) if output_dir is not None else Path(__file__).parent / "output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = params.device
    mesh = params.mesh

    x_bulk_left = 0.0
    x_bulk_right = params.device_width
    x_center = 0.5 * params.device_width
    x_gate_left = x_center - 0.5 * params.gate_length
    x_gate_right = x_center + 0.5 * params.gate_length
    x_device_left = x_bulk_left - params.air_thickness
    x_device_right = x_bulk_right + params.air_thickness

    y_bulk_top = 0.0
    y_oxide_top = -params.oxide_thickness
    y_oxide_mid = 0.5 * y_oxide_top
    y_gate_top = y_oxide_top - params.gate_thickness
    y_gate_mid = 0.5 * (y_gate_top + y_oxide_top)
    y_device_top = y_gate_top - params.air_thickness
    y_bulk_bottom = params.device_thickness
    y_bulk_mid = 0.5 * y_bulk_bottom
    y_device_bottom = y_bulk_bottom + params.air_thickness
    y_junction = params.junction_depth

    # Mesh resolution follows the dimensions being swept.
    y_channel_spacing = min(1.0e-8, params.oxide_thickness * 0.25)
    y_junction_spacing = min(2.0e-7, params.junction_depth * 0.2)
    y_gate_top_spacing = 1.0e-8
    y_gate_mid_spacing = params.gate_thickness * 0.25
    y_gate_bottom_spacing = 1.0e-8
    y_oxide_mid_spacing = params.oxide_thickness * 0.25
    x_channel_spacing = min(2.0e-7, params.gate_length * 0.05)
    x_halo_spacing = min(x_channel_spacing, params.halo_lateral_sigma * 0.25)
    x_diffusion_spacing = min(1.0e-5, (params.device_width - params.gate_length) * 0.25)
    max_y_spacing = params.device_thickness
    max_x_spacing = max(params.device_width, 1.0e-2)
    y_bulk_mid_spacing = params.device_thickness * 0.25
    y_bulk_bottom_spacing = 1.0e-8

    devsim.create_2d_mesh(mesh=mesh)
    devsim.add_2d_mesh_line(mesh=mesh, dir="y", pos=y_device_top, ps=max_y_spacing)
    devsim.add_2d_mesh_line(
        mesh=mesh, dir="y", pos=y_gate_top, ps=y_gate_top_spacing
    )
    devsim.add_2d_mesh_line(
        mesh=mesh, dir="y", pos=y_gate_mid, ps=y_gate_mid_spacing
    )
    devsim.add_2d_mesh_line(
        mesh=mesh,
        dir="y",
        pos=y_oxide_top,
        ns=y_oxide_mid_spacing,
        ps=y_gate_bottom_spacing,
    )
    devsim.add_2d_mesh_line(
        mesh=mesh, dir="y", pos=y_oxide_mid, ps=y_oxide_mid_spacing
    )
    devsim.add_2d_mesh_line(
        mesh=mesh,
        dir="y",
        pos=y_bulk_top,
        ns=y_oxide_mid_spacing,
        ps=y_channel_spacing,
    )
    devsim.add_2d_mesh_line(
        mesh=mesh, dir="y", pos=y_junction, ps=y_junction_spacing
    )
    devsim.add_2d_mesh_line(
        mesh=mesh, dir="y", pos=y_bulk_mid, ps=y_bulk_mid_spacing
    )
    devsim.add_2d_mesh_line(
        mesh=mesh,
        dir="y",
        pos=y_bulk_bottom,
        ns=y_bulk_bottom_spacing,
        ps=max_y_spacing,
    )
    devsim.add_2d_mesh_line(
        mesh=mesh, dir="y", pos=y_device_bottom, ps=max_y_spacing
    )

    # Geometric ordering and two-sided constraints preserve left/right symmetry.
    devsim.add_2d_mesh_line(
        mesh=mesh, dir="x", pos=x_device_left, ps=max_x_spacing
    )
    devsim.add_2d_mesh_line(
        mesh=mesh,
        dir="x",
        pos=x_bulk_left,
        ns=max_x_spacing,
        ps=x_diffusion_spacing,
    )
    devsim.add_2d_mesh_line(
        mesh=mesh,
        dir="x",
        pos=x_gate_left,
        ns=x_halo_spacing,
        ps=x_channel_spacing,
    )
    devsim.add_2d_mesh_line(
        mesh=mesh,
        dir="x",
        pos=x_center,
        ns=x_channel_spacing,
        ps=x_channel_spacing,
    )
    devsim.add_2d_mesh_line(
        mesh=mesh,
        dir="x",
        pos=x_gate_right,
        ns=x_channel_spacing,
        ps=x_halo_spacing,
    )
    devsim.add_2d_mesh_line(
        mesh=mesh,
        dir="x",
        pos=x_bulk_right,
        ns=x_diffusion_spacing,
        ps=max_x_spacing,
    )
    devsim.add_2d_mesh_line(
        mesh=mesh,
        dir="x",
        pos=x_device_right,
        ns=max_x_spacing,
        ps=max_x_spacing,
    )

    devsim.add_2d_region(mesh=mesh, material="Air", region="air")
    devsim.add_2d_region(
        mesh=mesh,
        material="Silicon",
        region="bulk",
        xl=x_bulk_left,
        xh=x_bulk_right,
        yl=y_bulk_bottom,
        yh=y_bulk_top,
    )
    devsim.add_2d_region(
        mesh=mesh,
        material="Silicon",
        region="gate",
        xl=x_gate_left,
        xh=x_gate_right,
        yl=y_oxide_top,
        yh=y_gate_top,
    )
    devsim.add_2d_region(
        mesh=mesh,
        material="Oxide",
        region="oxide",
        xl=x_gate_left,
        xh=x_gate_right,
        yl=y_bulk_top,
        yh=y_oxide_top,
    )

    devsim.add_2d_contact(
        mesh=mesh,
        name="gate",
        region="gate",
        yl=y_gate_top,
        yh=y_gate_top,
        material="metal",
    )
    devsim.add_2d_contact(
        mesh=mesh,
        name="body",
        region="bulk",
        yl=y_bulk_bottom,
        yh=y_bulk_bottom,
        material="metal",
    )
    devsim.add_2d_contact(
        mesh=mesh,
        name="source",
        region="bulk",
        yl=y_bulk_top,
        yh=y_bulk_top,
        xl=x_device_left,
        xh=x_gate_left,
        material="metal",
    )
    devsim.add_2d_contact(
        mesh=mesh,
        name="drain",
        region="bulk",
        yl=y_bulk_top,
        yh=y_bulk_top,
        xl=x_gate_right,
        xh=x_device_right,
        material="metal",
    )

    devsim.add_2d_interface(
        mesh=mesh, name="gate_oxide", region0="gate", region1="oxide"
    )
    devsim.add_2d_interface(
        mesh=mesh, name="bulk_oxide", region0="bulk", region1="oxide"
    )
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=device)

    values = {
        "gate_doping": params.gate_doping,
        "source_doping": params.source_doping,
        "drain_doping": params.drain_doping,
        "body_doping": params.body_doping,
        "bulk_doping": params.bulk_doping,
        "halo_peak_doping": params.halo_peak_doping,
        "x_gate_left": x_gate_left,
        "x_gate_right": x_gate_right,
        "x_halo_left": x_gate_left + params.halo_lateral_offset,
        "x_halo_right": x_gate_right - params.halo_lateral_offset,
        "x_diffusion_decay": params.lateral_diffusion_decay,
        "y_junction": y_junction,
        "y_diffusion_decay": params.vertical_diffusion_decay,
        "y_bulk_bottom": y_bulk_bottom,
        "y_halo": params.halo_depth,
        "halo_lateral_sigma": params.halo_lateral_sigma,
        "halo_vertical_sigma": params.halo_vertical_sigma,
    }
    devsim.node_model(
        name="NetDoping",
        device=device,
        region="gate",
        equation="%(gate_doping)s" % values,
    )
    devsim.node_model(
        name="SourceDoping",
        device=device,
        region="bulk",
        equation=(
            "0.25*%(source_doping)s"
            "*erfc((x-%(x_gate_left)s)/%(x_diffusion_decay)s)"
            "*erfc((y-%(y_junction)s)/%(y_diffusion_decay)s)"
        )
        % values,
    )
    devsim.node_model(
        name="DrainDoping",
        device=device,
        region="bulk",
        equation=(
            "0.25*%(drain_doping)s"
            "*erfc(-(x-%(x_gate_right)s)/%(x_diffusion_decay)s)"
            "*erfc((y-%(y_junction)s)/%(y_diffusion_decay)s)"
        )
        % values,
    )
    devsim.node_model(
        name="BodyDoping",
        device=device,
        region="bulk",
        equation=(
            "0.5*%(body_doping)s"
            "*erfc(-(y-%(y_bulk_bottom)s)/%(y_diffusion_decay)s)"
        )
        % values,
    )
    devsim.node_model(
        name="HaloDoping",
        device=device,
        region="bulk",
        equation=(
            "-%(halo_peak_doping)s*("
            "exp(-0.5*((x-%(x_halo_left)s)/%(halo_lateral_sigma)s)^2"
            "-0.5*((y-%(y_halo)s)/%(halo_vertical_sigma)s)^2)"
            "+exp(-0.5*((x-%(x_halo_right)s)/%(halo_lateral_sigma)s)^2"
            "-0.5*((y-%(y_halo)s)/%(halo_vertical_sigma)s)^2)"
            ")"
        )
        % values,
    )
    devsim.node_model(
        name="NetDoping",
        device=device,
        region="bulk",
        equation=(
            "SourceDoping + DrainDoping + %(bulk_doping)s"
            " + BodyDoping + HaloDoping"
        )
        % values,
    )

    if write_structure:
        devsim.write_devices(file=str(output_dir / "mos_2d"), type="vtk")
        devsim.write_devices(file=str(output_dir / "mos_2d.tec"), type="tecplot")

    return {
        "x_gate_left": x_gate_left,
        "x_gate_right": x_gate_right,
        "y_gate_top": y_gate_top,
        "y_oxide_top": y_oxide_top,
        "y_junction": y_junction,
    }
