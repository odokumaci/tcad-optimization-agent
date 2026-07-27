"""Create the default parameterized 2D MOSFET mesh and doping structure."""

from pathlib import Path

from mos_2d_model import MOSParameters, create_mos_device


OUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    create_mos_device(MOSParameters(), OUT_DIR)


if __name__ == "__main__":
    main()
