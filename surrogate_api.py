"""FastAPI service for MOSFET surrogate inference."""

from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from design_optimizer import optimize_design
from surrogate_inference import (
    DEFAULT_MODEL_PATH,
    TRAINED_DRAIN_VOLTAGES,
    SurrogatePredictor,
)


app = FastAPI(
    title="MOSFET TCAD Surrogate API",
    version="1.0.0",
    description="Predict ID-VG curves and extracted device metrics.",
)
predictor = SurrogatePredictor(DEFAULT_MODEL_PATH)

PositiveFloat = Annotated[float, Field(gt=0)]


class DeviceDesign(BaseModel):
    gate_length_nm: PositiveFloat
    oxide_thickness_nm: PositiveFloat
    halo_peak_doping_cm3: PositiveFloat
    junction_depth_nm: PositiveFloat
    allow_extrapolation: bool = False


class CurveRequest(DeviceDesign):
    gate_voltages_v: list[float] = Field(
        default_factory=lambda: np.round(np.arange(0.0, 1.21, 0.1), 10).tolist(),
        min_length=2,
    )
    drain_voltages_v: list[float] = Field(
        default_factory=lambda: list(TRAINED_DRAIN_VOLTAGES),
        min_length=1,
    )

    @field_validator("gate_voltages_v")
    @classmethod
    def gate_voltages_must_increase(cls, values: list[float]) -> list[float]:
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("gate_voltages_v must be strictly increasing")
        return values

    @field_validator("drain_voltages_v")
    @classmethod
    def drain_voltages_must_be_unique(cls, values: list[float]) -> list[float]:
        if len(set(values)) != len(values):
            raise ValueError("drain_voltages_v must contain unique values")
        return values


class CurvePoint(BaseModel):
    gate_voltage_v: float
    drain_current_ua_per_um: float


class DrainCurve(BaseModel):
    drain_voltage_v: float
    points: list[CurvePoint]


class CurveResponse(BaseModel):
    curves: list[DrainCurve]
    warnings: list[str]
    model_inference_latency_ms: float


class MetricResponse(BaseModel):
    threshold_voltage_v: float
    ion_ua_per_um: float
    ioff_ua_per_um: float
    ion_ioff_ratio: float
    subthreshold_slope_mv_per_dec: float
    dibl_mv_per_v: float
    warnings: list[str]
    model_inference_latency_ms: float


class OptimizationRequest(BaseModel):
    max_ioff_ua_per_um: PositiveFloat = 0.001
    max_ss_mv_per_dec: PositiveFloat = 85.0
    max_dibl_mv_per_v: PositiveFloat = 50.0
    min_threshold_voltage_v: float | None = None
    samples: int = Field(default=4096, ge=256, le=65536)
    seed: int = 90
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("samples")
    @classmethod
    def samples_must_be_power_of_two(cls, value: int) -> int:
        if value & (value - 1):
            raise ValueError("samples must be a power of two")
        return value


def validate_domain(
    request: DeviceDesign,
    gate_voltages_v: list[float],
    drain_voltages_v: list[float],
) -> list[str]:
    warnings = predictor.domain_warnings(
        gate_length_nm=request.gate_length_nm,
        oxide_thickness_nm=request.oxide_thickness_nm,
        halo_peak_doping_cm3=request.halo_peak_doping_cm3,
        junction_depth_nm=request.junction_depth_nm,
        gate_voltages_v=gate_voltages_v,
        drain_voltages_v=drain_voltages_v,
    )
    if warnings and not request.allow_extrapolation:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Request is outside the validated training domain",
                "warnings": warnings,
                "override": "Set allow_extrapolation=true to continue",
            },
        )
    return warnings


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model": predictor.metadata()}


@app.post("/predict/curve", response_model=CurveResponse)
def predict_curve(request: CurveRequest) -> CurveResponse:
    warnings = validate_domain(
        request, request.gate_voltages_v, request.drain_voltages_v
    )
    curves, latency_ms = predictor.predict_curves(
        gate_length_nm=request.gate_length_nm,
        oxide_thickness_nm=request.oxide_thickness_nm,
        halo_peak_doping_cm3=request.halo_peak_doping_cm3,
        junction_depth_nm=request.junction_depth_nm,
        gate_voltages_v=request.gate_voltages_v,
        drain_voltages_v=request.drain_voltages_v,
    )
    return CurveResponse(
        curves=[
            DrainCurve(
                drain_voltage_v=drain_voltage,
                points=[
                    CurvePoint(
                        gate_voltage_v=gate_voltage,
                        drain_current_ua_per_um=current,
                    )
                    for gate_voltage, current in zip(
                        request.gate_voltages_v, currents, strict=True
                    )
                ],
            )
            for drain_voltage, currents in curves.items()
        ],
        warnings=warnings,
        model_inference_latency_ms=latency_ms,
    )


@app.post("/predict/metrics", response_model=MetricResponse)
def predict_metrics(request: DeviceDesign) -> MetricResponse:
    gate_voltages = np.round(np.arange(0.0, 1.21, 0.1), 10).tolist()
    drain_voltages = list(TRAINED_DRAIN_VOLTAGES)
    warnings = validate_domain(request, gate_voltages, drain_voltages)
    curves, latency_ms = predictor.predict_curves(
        gate_length_nm=request.gate_length_nm,
        oxide_thickness_nm=request.oxide_thickness_nm,
        halo_peak_doping_cm3=request.halo_peak_doping_cm3,
        junction_depth_nm=request.junction_depth_nm,
        gate_voltages_v=gate_voltages,
        drain_voltages_v=drain_voltages,
    )
    metrics = predictor.extract_metrics(
        gate_voltages,
        curves[drain_voltages[0]],
        curves[drain_voltages[1]],
    )
    return MetricResponse(
        **metrics,
        warnings=warnings,
        model_inference_latency_ms=latency_ms,
    )


@app.post("/optimize/design")
def optimize_device_design(request: OptimizationRequest) -> dict[str, Any]:
    return optimize_design(
        predictor,
        max_ioff_ua_per_um=request.max_ioff_ua_per_um,
        max_ss_mv_per_dec=request.max_ss_mv_per_dec,
        max_dibl_mv_per_v=request.max_dibl_mv_per_v,
        min_threshold_voltage_v=request.min_threshold_voltage_v,
        samples=request.samples,
        seed=request.seed,
        top_k=request.top_k,
    )
