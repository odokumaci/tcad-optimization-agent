"""Load the surrogate and perform one prediction for cold-start benchmarking."""

import argparse
from pathlib import Path

import torch

from train_idvg_surrogate import IdVgSurrogate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.model_path.resolve(), map_location="cpu", weights_only=True
    )
    model = IdVgSurrogate()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(torch.zeros((26, 6), dtype=torch.float32))
    print(float(prediction[0]))


if __name__ == "__main__":
    main()
