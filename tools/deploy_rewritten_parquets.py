#!/usr/bin/env python3
"""Compatibilidade: parquets processados agora são gerados diretamente em processed/."""

import argparse
from pathlib import Path

from rewrite_main_parquets import rewrite_parquets


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "Data")
    parser.add_argument("--output", type=Path, default=root / "processed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rewrite_parquets(args.input, args.output, args.overwrite)


if __name__ == "__main__":
    main()
