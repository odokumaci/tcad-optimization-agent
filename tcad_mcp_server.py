"""MCP adapter for the independently deployed MOSFET surrogate API."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


API_URL = os.getenv("SURROGATE_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 15.0
ROOT = Path(__file__).parent

mcp = FastMCP(
    name="MOSFET TCAD Surrogate",
    instructions=(
        "Use these tools to predict MOSFET transfer curves and engineering "
        "metrics inside the validated design domain. Check valid ranges before "
        "proposing a design. Surrogate predictions should be verified with "
        "full TCAD before final engineering decisions."
    ),
)


async def api_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(
            base_url=API_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as client:
            response = await client.request(method, path, json=json_body)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as error:
        detail = error.response.text
        raise RuntimeError(
            f"Surrogate API rejected the request ({error.response.status_code}): "
            f"{detail}"
        ) from error
    except httpx.RequestError as error:
        raise RuntimeError(
            f"Surrogate API is unavailable at {API_URL}: {error}"
        ) from error


@mcp.tool()
async def check_api_health() -> dict[str, Any]:
    """Check whether the surrogate API and trained model are available."""
    return await api_request("GET", "/health")


@mcp.tool()
async def get_valid_design_ranges() -> dict[str, Any]:
    """Return validated parameter ranges and supported drain voltages."""
    health = await api_request("GET", "/health")
    model = health["model"]
    return {
        "training_domain": model["training_domain"],
        "trained_drain_voltages_v": model["trained_drain_voltages_v"],
        "extrapolation_policy": (
            "Requests outside these ranges are rejected unless "
            "allow_extrapolation is explicitly enabled."
        ),
    }


@mcp.tool()
async def predict_device_metrics(
    gate_length_nm: float,
    oxide_thickness_nm: float,
    halo_peak_doping_1e19: float,
    junction_depth_nm: float,
    allow_extrapolation: bool = False,
) -> dict[str, Any]:
    """Predict device metrics; halo doping is expressed in units of 1e19 cm^-3."""
    return await api_request(
        "POST",
        "/predict/metrics",
        json_body={
            "gate_length_nm": gate_length_nm,
            "oxide_thickness_nm": oxide_thickness_nm,
            "halo_peak_doping_cm3": halo_peak_doping_1e19 * 1.0e19,
            "junction_depth_nm": junction_depth_nm,
            "allow_extrapolation": allow_extrapolation,
        },
    )


@mcp.tool()
async def predict_idvg_curves(
    gate_length_nm: float,
    oxide_thickness_nm: float,
    halo_peak_doping_1e19: float,
    junction_depth_nm: float,
    gate_voltages_v: list[float] | None = None,
    drain_voltages_v: list[float] | None = None,
    allow_extrapolation: bool = False,
) -> dict[str, Any]:
    """Predict ID-VG curves; halo doping is in units of 1e19 cm^-3."""
    payload: dict[str, Any] = {
        "gate_length_nm": gate_length_nm,
        "oxide_thickness_nm": oxide_thickness_nm,
        "halo_peak_doping_cm3": halo_peak_doping_1e19 * 1.0e19,
        "junction_depth_nm": junction_depth_nm,
        "allow_extrapolation": allow_extrapolation,
    }
    if gate_voltages_v is not None:
        payload["gate_voltages_v"] = gate_voltages_v
    if drain_voltages_v is not None:
        payload["drain_voltages_v"] = drain_voltages_v
    return await api_request(
        "POST",
        "/predict/curve",
        json_body=payload,
    )


@mcp.tool()
async def optimize_device_design(
    max_ioff_ua_per_um: float = 0.001,
    max_ss_mv_per_dec: float = 85.0,
    max_dibl_mv_per_v: float = 50.0,
    min_threshold_voltage_v: float | None = None,
    samples: int = 4096,
    seed: int = 90,
    top_k: int = 5,
) -> dict[str, Any]:
    """Maximize ION subject to IOFF, SS, DIBL, and optional VTH constraints."""
    return await api_request(
        "POST",
        "/optimize/design",
        json_body={
            "max_ioff_ua_per_um": max_ioff_ua_per_um,
            "max_ss_mv_per_dec": max_ss_mv_per_dec,
            "max_dibl_mv_per_v": max_dibl_mv_per_v,
            "min_threshold_voltage_v": min_threshold_voltage_v,
            "samples": samples,
            "seed": seed,
            "top_k": top_k,
        },
    )


@mcp.tool()
async def verify_design_with_tcad(
    gate_length_nm: float,
    oxide_thickness_nm: float,
    halo_peak_doping_1e19: float,
    junction_depth_nm: float,
    max_ioff_ua_per_um: float = 0.001,
    max_ss_mv_per_dec: float = 85.0,
    max_dibl_mv_per_v: float = 50.0,
) -> dict[str, Any]:
    """Run full DEVSIM verification; this long-running tool may take one minute."""
    run_id = f"mcp_{uuid.uuid4().hex[:10]}"
    output_dir = ROOT / "verification" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "verification.log"
    environment = os.environ.copy()
    environment["SURROGATE_API_URL"] = API_URL
    environment["DEVSIM_MATH_LIBS"] = "mkl_rt.3.dll"
    environment["PATH"] = (
        str(ROOT / ".venv" / "Library" / "bin")
        + os.pathsep
        + environment.get("PATH", "")
    )
    command = [
        sys.executable,
        "-u",
        str(ROOT / "verify_design_with_tcad.py"),
        "--gate-length-nm",
        str(gate_length_nm),
        "--oxide-thickness-nm",
        str(oxide_thickness_nm),
        "--halo-peak-doping-cm3",
        str(halo_peak_doping_1e19 * 1.0e19),
        "--junction-depth-nm",
        str(junction_depth_nm),
        "--max-ioff-ua-per-um",
        str(max_ioff_ua_per_um),
        "--max-ss-mv-per-dec",
        str(max_ss_mv_per_dec),
        "--max-dibl-mv-per-v",
        str(max_dibl_mv_per_v),
        "--api-url",
        API_URL,
        "--output-dir",
        str(output_dir),
    ]

    def run_verification() -> None:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                close_fds=True,
            )
            assert process.stdout is not None

            def copy_output() -> None:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(
                        f"[verification {run_id}] {line}",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

            reader = threading.Thread(target=copy_output, daemon=True)
            reader.start()
            try:
                return_code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                reader.join(timeout=5)
                raise
            reader.join()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

    try:
        await asyncio.to_thread(run_verification)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"DEVSIM verification exceeded 180 seconds; log={log_path}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"DEVSIM verification failed; log={log_path}"
        ) from error

    report_path = output_dir / "verification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "report_path": str(report_path),
        "curve_plot_path": str(output_dir / "verification_curves.png"),
        "report": report,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
