"""Train a PyTorch surrogate for complete MOSFET ID-VG curves."""

import argparse
import copy
import csv
import json
import math
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).parent
DEFAULT_DATASET = ROOT / "dataset" / "sobol_256"
DEFAULT_MODEL = ROOT / "models" / "idvg_surrogate.pt"
FEATURE_NAMES = (
    "gate_length_nm",
    "oxide_thickness_nm",
    "log10_halo_peak_doping_cm3",
    "junction_depth_nm",
    "gate_voltage_v",
    "drain_voltage_v",
)
CURRENT_COLUMN = re.compile(
    r"drain_current_vd_(?P<vd>[0-9.]+)_ua_per_um"
)


class IdVgSurrogate(nn.Module):
    def __init__(self, input_size: int = len(FEATURE_NAMES)) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_case_ids(
    case_ids: list[str], seed: int
) -> tuple[list[str], list[str], list[str]]:
    shuffled = case_ids.copy()
    random.Random(seed).shuffle(shuffled)
    train_end = int(0.70 * len(shuffled))
    validation_end = train_end + int(0.15 * len(shuffled))
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )


def load_completed_cases(dataset_dir: Path) -> dict[str, dict[str, str]]:
    with (dataset_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = {
        row["case_id"]: row
        for row in rows
        if row["status"] == "completed"
        and (dataset_dir / row["case_id"] / "transfer_curves.csv").exists()
    }
    if not completed:
        raise ValueError(f"No completed cases found in {dataset_dir}")
    return completed


def load_samples(
    dataset_dir: Path,
    case_rows: dict[str, dict[str, str]],
    case_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    targets: list[float] = []
    sample_cases: list[str] = []

    for case_id in case_ids:
        device = case_rows[case_id]
        fixed_features = [
            float(device["gate_length_nm"]),
            float(device["oxide_thickness_nm"]),
            math.log10(float(device["halo_peak_doping_cm3"])),
            float(device["junction_depth_nm"]),
        ]
        curve_path = dataset_dir / case_id / "transfer_curves.csv"
        with curve_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            current_columns: list[tuple[str, float]] = []
            for column in reader.fieldnames or []:
                match = CURRENT_COLUMN.fullmatch(column)
                if match:
                    current_columns.append((column, float(match.group("vd"))))
            if not current_columns:
                raise ValueError(f"No drain-current columns found in {curve_path}")

            for row in reader:
                gate_voltage = float(row["gate_voltage_v"])
                for column, drain_voltage in current_columns:
                    current = max(abs(float(row[column])), 1.0e-18)
                    features.append(
                        [*fixed_features, gate_voltage, drain_voltage]
                    )
                    targets.append(math.log10(current))
                    sample_cases.append(case_id)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(sample_cases),
    )


def standardize(
    train_x: np.ndarray,
    train_y: np.ndarray,
    arrays_x: list[np.ndarray],
    arrays_y: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, np.ndarray]]:
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std[x_std < 1.0e-12] = 1.0
    y_mean = np.asarray(train_y.mean(), dtype=np.float32)
    y_std = np.asarray(train_y.std(), dtype=np.float32)
    if float(y_std) < 1.0e-12:
        y_std = np.asarray(1.0, dtype=np.float32)

    normalized_x = [(values - x_mean) / x_std for values in arrays_x]
    normalized_y = [(values - y_mean) / y_std for values in arrays_y]
    normalization = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return normalized_x, normalized_y, normalization


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_function: nn.Module,
) -> float:
    model.eval()
    return float(loss_function(model(inputs), targets).item())


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    train_dataset = TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y)
    )
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_inputs = torch.from_numpy(validation_x).to(device)
    validation_targets = torch.from_numpy(validation_y).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-5
    )

    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.shape[0]
            total_samples += batch_x.shape[0]

        train_loss = total_loss / total_samples
        val_loss = validation_loss(
            model, validation_inputs, validation_targets, loss_function
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss}
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"validation_loss={val_loss:.6f}"
            )

        if val_loss < best_loss - 1.0e-7:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_log_current(
    model: nn.Module,
    normalized_x: np.ndarray,
    normalization: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = model(torch.from_numpy(normalized_x).to(device)).cpu().numpy()
    return (
        predictions * float(normalization["y_std"])
        + float(normalization["y_mean"])
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - actual
    ss_residual = float(np.sum(residual**2))
    ss_total = float(np.sum((actual - actual.mean()) ** 2))
    actual_linear = 10.0**actual
    predicted_linear = 10.0**predicted
    return {
        "log10_mae_decades": float(np.mean(np.abs(residual))),
        "log10_rmse_decades": float(np.sqrt(np.mean(residual**2))),
        "log10_r2": 1.0 - ss_residual / ss_total,
        "median_absolute_percentage_error": float(
            np.median(np.abs(predicted_linear - actual_linear) / actual_linear)
            * 100.0
        ),
    }


def plot_parity(
    actual: np.ndarray, predicted: np.ndarray, output_path: Path
) -> None:
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(actual, predicted, s=14, alpha=0.55)
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("TCAD log₁₀|ID| (µA/µm)")
    ax.set_ylabel("Predicted log₁₀|ID| (µA/µm)")
    ax.set_title("Held-out device parity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_test_curves(
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_predictions: np.ndarray,
    test_sample_cases: np.ndarray,
    output_path: Path,
) -> None:
    selected_cases = list(dict.fromkeys(test_sample_cases.tolist()))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, case_id in zip(axes.flat, selected_cases, strict=True):
        mask = test_sample_cases == case_id
        case_x = test_x[mask]
        actual = 10.0 ** test_y[mask]
        predicted = 10.0 ** test_predictions[mask]
        for drain_voltage in sorted(set(case_x[:, 5])):
            vd_mask = np.isclose(case_x[:, 5], drain_voltage)
            order = np.argsort(case_x[vd_mask, 4])
            gate_voltage = case_x[vd_mask, 4][order]
            ax.semilogy(
                gate_voltage,
                actual[vd_mask][order],
                marker="o",
                label=f"TCAD VD={drain_voltage:g} V",
            )
            ax.semilogy(
                gate_voltage,
                predicted[vd_mask][order],
                linestyle="--",
                label=f"ML VD={drain_voltage:g} V",
            )
        ax.set_title(case_id)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("VG (V)")
    for ax in axes[:, 0]:
        ax.set_ylabel("|ID| (µA/µm)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Surrogate predictions on held-out devices", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=90)
    args = parser.parse_args()
    set_seed(args.seed)

    dataset_dir = args.dataset_dir.resolve()
    case_rows = load_completed_cases(dataset_dir)
    train_ids, validation_ids, test_ids = split_case_ids(
        sorted(case_rows), args.seed
    )
    train_x, train_y, _ = load_samples(dataset_dir, case_rows, train_ids)
    validation_x, validation_y, _ = load_samples(
        dataset_dir, case_rows, validation_ids
    )
    test_x, test_y, test_sample_cases = load_samples(
        dataset_dir, case_rows, test_ids
    )
    normalized_x, normalized_y, normalization = standardize(
        train_x,
        train_y,
        [train_x, validation_x, test_x],
        [train_y, validation_y, test_y],
    )
    train_x_norm, validation_x_norm, test_x_norm = normalized_x
    train_y_norm, validation_y_norm, _ = normalized_y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} cases={len(case_rows)} "
        f"split={len(train_ids)}/{len(validation_ids)}/{len(test_ids)}"
    )
    model, history = train_model(
        IdVgSurrogate(),
        train_x_norm,
        train_y_norm,
        validation_x_norm,
        validation_y_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=device,
    )
    test_predictions = predict_log_current(
        model, test_x_norm, normalization, device
    )
    metrics = regression_metrics(test_y, test_predictions)
    print(json.dumps(metrics, indent=2))

    model_path = args.model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "feature_names": FEATURE_NAMES,
            "normalization": {
                name: np.asarray(value).tolist()
                for name, value in normalization.items()
            },
            "split_case_ids": {
                "train": train_ids,
                "validation": validation_ids,
                "test": test_ids,
            },
            "metrics": metrics,
            "seed": args.seed,
        },
        model_path,
    )
    metrics_path = model_path.with_name(f"{model_path.stem}_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "dataset_directory": str(dataset_dir),
                "completed_devices": len(case_rows),
                "sample_counts": {
                    "train": len(train_x),
                    "validation": len(validation_x),
                    "test": len(test_x),
                },
                "split_case_counts": {
                    "train": len(train_ids),
                    "validation": len(validation_ids),
                    "test": len(test_ids),
                },
                "test_metrics": metrics,
                "training_history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assets = args.assets_dir.resolve()
    assets.mkdir(exist_ok=True)
    plot_parity(test_y, test_predictions, assets / "surrogate_idvg_parity.png")
    plot_test_curves(
        test_x,
        test_y,
        test_predictions,
        test_sample_cases,
        assets / "surrogate_idvg_test_curves.png",
    )


if __name__ == "__main__":
    main()
