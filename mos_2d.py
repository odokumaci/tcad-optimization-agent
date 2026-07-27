"""Run a parameterized 2D MOSFET Poisson + drift-diffusion simulation."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import devsim
from devsim.python_packages.model_create import CreateSolution
from devsim.python_packages.ramp import rampbias
from devsim.python_packages.simple_physics import (
    CreateOxidePotentialOnly,
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconOxideInterface,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    GetContactBiasName,
    SetOxideParameters,
    SetSiliconParameters,
)

from mos_2d_model import MOSParameters, create_mos_device, parameters_to_dict


DEFAULT_OUT_DIR = Path(__file__).parent / "output"
SILICON_REGIONS = ("gate", "bulk")
OXIDE_REGIONS = ("oxide",)
REGIONS = ("gate", "bulk", "oxide")
INTERFACES = ("bulk_oxide", "gate_oxide")


@dataclass(frozen=True)
class Biases:
    gate: float = 0.0
    drain: float = 0.0
    source: float = 0.0
    body: float = 0.0


def _write_parameter_restore(device: str, output_dir: Path) -> None:
    with (output_dir / "mos_2d_params.py").open("w", encoding="utf-8") as ofh:
        ofh.write("import devsim\n")
        for parameter in devsim.get_parameter_list():
            if parameter in ("solver_callback", "direct_solver", "info"):
                continue
            value = repr(devsim.get_parameter(name=parameter))
            ofh.write(
                f'devsim.set_parameter(name="{parameter}", value={value})\n'
            )
        for parameter in devsim.get_parameter_list(device=device):
            value = repr(devsim.get_parameter(device=device, name=parameter))
            ofh.write(
                f'devsim.set_parameter(device="{device}", '
                f'name="{parameter}", value={value})\n'
            )
        for region in devsim.get_region_list(device=device):
            for parameter in devsim.get_parameter_list(device=device, region=region):
                value = repr(
                    devsim.get_parameter(
                        device=device, region=region, name=parameter
                    )
                )
                ofh.write(
                    f'devsim.set_parameter(device="{device}", region="{region}", '
                    f'name="{parameter}", value={value})\n'
                )


def _ramp_biases(
    device: str,
    biases: Biases,
    *,
    step_size: float,
) -> None:
    if step_size <= 0:
        raise ValueError("bias step must be positive")

    # Establish terminal voltages one at a time from the converged equilibrium
    # solution. Adaptive halving in rampbias improves high-field convergence.
    for contact in ("source", "body", "gate", "drain"):
        target = getattr(biases, contact)
        if target == 0.0:
            continue
        rampbias(
            device=device,
            contact=contact,
            end_bias=target,
            step_size=min(step_size, abs(target)),
            min_step=1.0e-4,
            max_iter=50,
            rel_error=1.0e-4,
            abs_error=1.0e30,
            callback=lambda _device: None,
        )


def _contact_currents(device: str) -> dict[str, dict[str, float]]:
    currents: dict[str, dict[str, float]] = {}
    for contact in devsim.get_contact_list(device=device):
        electron = devsim.get_contact_current(
            device=device,
            contact=contact,
            equation="ElectronContinuityEquation",
        )
        hole = devsim.get_contact_current(
            device=device,
            contact=contact,
            equation="HoleContinuityEquation",
        )
        currents[contact] = {
            "electron": electron,
            "hole": hole,
            "total": electron + hole,
        }
    return currents


def run_simulation(
    params: MOSParameters | None = None,
    biases: Biases | None = None,
    output_dir: Path | str = DEFAULT_OUT_DIR,
    *,
    bias_step: float = 0.05,
) -> dict:
    """Build, solve, bias, and save one MOSFET simulation."""
    params = params or MOSParameters()
    biases = biases or Biases()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = create_mos_device(params, output_dir)
    device = params.device

    for region in REGIONS:
        CreateSolution(device, region, "Potential")

    for region in SILICON_REGIONS:
        SetSiliconParameters(device, region, params.temperature)
        for mobility in ("mu_n", "mu_p"):
            value = devsim.get_parameter(
                device=device, region=region, name=mobility
            )
            devsim.set_parameter(
                device=device,
                region=region,
                name=mobility,
                value=value * params.mobility_scale,
            )
        CreateSiliconPotentialOnly(device, region)

    for region in OXIDE_REGIONS:
        SetOxideParameters(device, region, params.temperature)
        CreateOxidePotentialOnly(device, region, "log_damp")

    contacts = devsim.get_contact_list(device=device)
    for contact in contacts:
        region = devsim.get_region_list(device=device, contact=contact)[0]
        devsim.set_parameter(
            device=device, name=GetContactBiasName(contact), value=0.0
        )
        CreateSiliconPotentialOnlyContact(device, region, contact)

    for interface in INTERFACES:
        CreateSiliconOxideInterface(device, interface)

    devsim.solve(
        type="dc",
        absolute_error=1.0e-13,
        relative_error=1.0e-12,
        maximum_iterations=50,
    )
    devsim.solve(
        type="dc",
        absolute_error=1.0e-13,
        relative_error=1.0e-12,
        maximum_iterations=50,
    )
    devsim.write_devices(
        file=str(output_dir / "gmsh_mos2d_potentialonly"), type="vtk"
    )

    for region in SILICON_REGIONS:
        CreateSolution(device, region, "Electrons")
        CreateSolution(device, region, "Holes")
        devsim.set_node_values(
            device=device,
            region=region,
            name="Electrons",
            init_from="IntrinsicElectrons",
        )
        devsim.set_node_values(
            device=device,
            region=region,
            name="Holes",
            init_from="IntrinsicHoles",
        )
        CreateSiliconDriftDiffusion(device, region, "mu_n", "mu_p")

    for contact in contacts:
        region = devsim.get_region_list(device=device, contact=contact)[0]
        CreateSiliconDriftDiffusionAtContact(device, region, contact)

    devsim.solve(
        type="dc",
        absolute_error=1.0e30,
        relative_error=1.0e-5,
        maximum_iterations=50,
    )
    _ramp_biases(device, biases, step_size=bias_step)

    for region in SILICON_REGIONS:
        devsim.node_model(
            device=device,
            region=region,
            name="logElectrons",
            equation="log(Electrons)/log(10)",
        )

    devsim.write_devices(file=str(output_dir / "mos_2d_dd.msh"), type="devsim")
    _write_parameter_restore(device, output_dir)

    result = {
        "parameters": parameters_to_dict(params),
        "biases": asdict(biases),
        "geometry_cm": geometry,
        "currents": _contact_currents(device),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "mos_2d_run.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _nm(value: float) -> float:
    return value * 1.0e-7


def _um(value: float) -> float:
    return value * 1.0e-4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-length-nm", type=float, default=50.0)
    parser.add_argument("--oxide-thickness-nm", type=float, default=1.2)
    parser.add_argument("--junction-depth-nm", type=float, default=30.0)
    parser.add_argument("--gate-thickness-nm", type=float, default=100.0)
    parser.add_argument("--device-width-um", type=float, default=1.0)
    parser.add_argument("--device-depth-um", type=float, default=1.0)
    parser.add_argument("--bulk-doping", type=float, default=-1.0e16)
    parser.add_argument("--body-doping", type=float, default=-1.0e19)
    parser.add_argument("--source-doping", type=float, default=1.0e20)
    parser.add_argument("--drain-doping", type=float, default=1.0e20)
    parser.add_argument("--gate-doping", type=float, default=1.0e20)
    parser.add_argument("--halo-doping", type=float, default=3.0e19)
    parser.add_argument("--halo-depth-nm", type=float, default=30.0)
    parser.add_argument("--halo-offset-nm", type=float, default=10.0)
    parser.add_argument("--halo-lateral-sigma-nm", type=float, default=15.0)
    parser.add_argument("--halo-vertical-sigma-nm", type=float, default=10.0)
    parser.add_argument("--mobility-scale", type=float, default=0.1)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--vg", type=float, default=0.0)
    parser.add_argument("--vd", type=float, default=0.0)
    parser.add_argument("--vs", type=float, default=0.0)
    parser.add_argument("--vb", type=float, default=0.0)
    parser.add_argument("--bias-step", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = MOSParameters(
        device_width=_um(args.device_width_um),
        gate_length=_nm(args.gate_length_nm),
        oxide_thickness=_nm(args.oxide_thickness_nm),
        gate_thickness=_nm(args.gate_thickness_nm),
        device_thickness=_um(args.device_depth_um),
        junction_depth=_nm(args.junction_depth_nm),
        bulk_doping=args.bulk_doping,
        body_doping=args.body_doping,
        source_doping=args.source_doping,
        drain_doping=args.drain_doping,
        gate_doping=args.gate_doping,
        halo_peak_doping=args.halo_doping,
        halo_depth=_nm(args.halo_depth_nm),
        halo_lateral_offset=_nm(args.halo_offset_nm),
        halo_lateral_sigma=_nm(args.halo_lateral_sigma_nm),
        halo_vertical_sigma=_nm(args.halo_vertical_sigma_nm),
        mobility_scale=args.mobility_scale,
        temperature=args.temperature_k,
    )
    biases = Biases(
        gate=args.vg,
        drain=args.vd,
        source=args.vs,
        body=args.vb,
    )
    result = run_simulation(
        params=params,
        biases=biases,
        output_dir=args.output_dir,
        bias_step=args.bias_step,
    )
    print(json.dumps({"biases": result["biases"], "currents": result["currents"]}, indent=2))


if __name__ == "__main__":
    main()
