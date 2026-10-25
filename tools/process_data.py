#!/usr/bin/env python3
"""Consolida somente os CSVs de propriedades, sem alterar os dados brutos."""
import argparse
from pathlib import Path

from src.processing_data import processing_all_data


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "Data")
    parser.add_argument("--output", type=Path, default=root / "processed")
    args = parser.parse_args()
    processing_all_data(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
