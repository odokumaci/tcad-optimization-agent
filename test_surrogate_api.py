"""API integration tests for the MOSFET surrogate."""

from fastapi.testclient import TestClient

from surrogate_api import app


client = TestClient(app)
BASE_DESIGN = {
    "gate_length_nm": 50.0,
    "oxide_thickness_nm": 1.2,
    "halo_peak_doping_cm3": 3.0e19,
    "junction_depth_nm": 30.0,
}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_metrics() -> None:
    response = client.post("/predict/metrics", json=BASE_DESIGN)
    assert response.status_code == 200
    payload = response.json()
    assert payload["ion_ua_per_um"] > payload["ioff_ua_per_um"] > 0
    assert payload["warnings"] == []


def test_predict_curve_shape() -> None:
    response = client.post("/predict/curve", json=BASE_DESIGN)
    assert response.status_code == 200
    curves = response.json()["curves"]
    assert len(curves) == 2
    assert all(len(curve["points"]) == 13 for curve in curves)


def test_out_of_domain_rejected() -> None:
    request = {**BASE_DESIGN, "gate_length_nm": 30.0}
    response = client.post("/predict/metrics", json=request)
    assert response.status_code == 422
    assert "outside the validated training domain" in response.json()["detail"]["message"]


def test_extrapolation_override_returns_warning() -> None:
    request = {
        **BASE_DESIGN,
        "gate_length_nm": 30.0,
        "allow_extrapolation": True,
    }
    response = client.post("/predict/metrics", json=request)
    assert response.status_code == 200
    assert response.json()["warnings"]


def test_constrained_design_optimization() -> None:
    request = {
        "max_ioff_ua_per_um": 0.001,
        "max_ss_mv_per_dec": 85.0,
        "max_dibl_mv_per_v": 50.0,
        "samples": 256,
        "top_k": 3,
    }
    response = client.post("/optimize/design", json=request)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    metrics = payload["best"]["predicted_metrics"]
    assert metrics["ioff_ua_per_um"] <= request["max_ioff_ua_per_um"]
    assert metrics["subthreshold_slope_mv_per_dec"] <= request["max_ss_mv_per_dec"]
    assert metrics["dibl_mv_per_v"] <= request["max_dibl_mv_per_v"]
