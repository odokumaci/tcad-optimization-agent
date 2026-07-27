"""Start Uvicorn, test live HTTP endpoints, then stop the server."""

import subprocess
import sys
import time

import httpx


BASE_URL = "http://127.0.0.1:8000"
DESIGN = {
    "gate_length_nm": 50.0,
    "oxide_thickness_nm": 1.2,
    "halo_peak_doping_cm3": 3.0e19,
    "junction_depth_nm": 30.0,
}


def wait_until_ready(client: httpx.Client, process: subprocess.Popen) -> None:
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError(f"Uvicorn exited with code {process.returncode}")
        try:
            response = client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.1)
    raise TimeoutError("API server did not become ready within 6 seconds")


def main() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "surrogate_api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
        ]
    )
    try:
        with httpx.Client(timeout=10.0) as client:
            wait_until_ready(client, process)

            health = client.get(f"{BASE_URL}/health")
            health.raise_for_status()

            metrics = client.post(f"{BASE_URL}/predict/metrics", json=DESIGN)
            metrics.raise_for_status()

            curves = client.post(f"{BASE_URL}/predict/curve", json=DESIGN)
            curves.raise_for_status()

            outside_domain = client.post(
                f"{BASE_URL}/predict/metrics",
                json={**DESIGN, "gate_length_nm": 30.0},
            )
            if outside_domain.status_code != 422:
                raise AssertionError("Out-of-domain request was not rejected")

            metric_payload = metrics.json()
            curve_payload = curves.json()
            print("health=passed")
            print(
                f"metrics=passed ion={metric_payload['ion_ua_per_um']:.3f} "
                f"ioff={metric_payload['ioff_ua_per_um']:.6g}"
            )
            print(
                f"curves=passed count={len(curve_payload['curves'])} "
                f"points={len(curve_payload['curves'][0]['points'])}"
            )
            print("domain_guard=passed")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
