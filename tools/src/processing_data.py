"""Processamento dos CSVs ``prop`` sem escrever na árvore bruta."""
from __future__ import annotations

import csv
import re
from math import nan
from pathlib import Path

import numpy as np
import pandas as pd

VALUE_COLUMNS = ["filename", "ShortestCost", "ShortestBFS", "ShortestBFSSampled1000", "R_Spearman", "R_Spearman_err", "R_Newman", "R_Newman_err", "ClusterCoefficient"]
HEADER_MAP = {
    "meanshortestpathdijstrika": "ShortestCost", "meanshortestpathbfs": "ShortestBFS",
    "meanshortestpathbfssampled1000": "ShortestBFSSampled1000", "assspearman": "R_Spearman",
    "assspearmanerror": "R_Spearman_err", "assnewman": "R_Newman", "assnewmanerror": "R_Newman_err",
    "clustercoefficient": "ClusterCoefficient",
}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _prop_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("N_*/m0_*/dim_*/alpha_a_*_alpha_g_*/prop"))


def _params(prop: Path) -> tuple[int, int, int, float, float]:
    text = prop.as_posix()
    get = lambda pattern: re.search(pattern, text).group(1)
    alpha = re.search(r"alpha_a_([+-]?\d+(?:\.\d+)?)_alpha_g_([+-]?\d+(?:\.\d+)?)", text)
    if alpha is None:
        raise ValueError(f"Parâmetros alpha ausentes: {prop}")
    return int(get(r"(?:^|/)N_(\d+)(?:/|$)")), int(get(r"(?:^|/)m0_(\d+)(?:/|$)")), int(get(r"(?:^|/)dim_(\d+)(?:/|$)")), float(alpha.group(1)), float(alpha.group(2))


def _read_csv(path: Path) -> dict[str, float | str]:
    row: dict[str, float | str] = {column: nan for column in VALUE_COLUMNS}
    row["filename"] = path.name
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        reader = (values for values in csv.reader(stream, delimiter=",", quotechar="|") if values and any(value.strip() for value in values))
        header, data = next(reader, None), next(reader, None)
    if not header or not data:
        return row
    positions = {_norm(name.strip().lstrip("#")): i for i, name in enumerate(header)}
    for source, destination in HEADER_MAP.items():
        index = positions.get(source)
        if index is not None and index < len(data) and data[index].strip():
            try:
                row[destination] = float(data[index])
            except ValueError:
                pass
    return row


def _sem(values: pd.Series) -> float:
    values = values.dropna()
    return np.nan if len(values) <= 1 else values.std(ddof=1) / np.sqrt(len(values))


def _error(values: pd.Series, errors: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return np.sqrt(_sem(values) ** 2 + np.square(errors.reindex(values.index).dropna()).sum() / len(values) ** 2)


def _aggregate(rows: pd.DataFrame, params: tuple[int, int, int, float, float]) -> dict:
    N, m0, dim, alpha_a, alpha_g = params
    num = lambda name: pd.to_numeric(rows[name], errors="coerce")
    cost, bfs, sampled = num("ShortestCost"), num("ShortestBFS"), num("ShortestBFSSampled1000")
    spearman, newman, cluster = num("R_Spearman"), num("R_Newman"), num("ClusterCoefficient")
    return {"N": N, "m0": m0, "dim": dim, "alpha_a": alpha_a, "alpha_g": alpha_g, "N_samples": len(rows),
            "Short_Cost_mean": cost.mean(), "Short_Cost_error": _sem(cost), "Short_BFS_mean": bfs.mean(), "Short_BFS_error": _sem(bfs),
            "Short_BFS_sampled_1000_mean": sampled.mean(), "Short_BFS_sampled_1000_error": _sem(sampled),
            "R_Spearman_mean": spearman.mean(), "R_Spearman_error": _error(spearman, num("R_Spearman_err")),
            "R_Newman_mean": newman.mean(), "R_Newman_error": _error(newman, num("R_Newman_err")),
            "ClusterCoefficient_mean": cluster.mean(), "ClusterCoefficient_error": _sem(cluster)}


def _build_from_processed(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstrói manifest global e agregado a partir dos manifests processados."""
    manifests, aggregates = [], []
    pattern = "N_*/m0_*/dim_*/alpha_a_*_alpha_g_*/prop_manifest.csv"
    for manifest_path in sorted(output.glob(pattern)):
        rows = pd.read_csv(manifest_path)
        if rows.empty:
            continue
        params = _params(manifest_path.parent)
        N, m0, dim, alpha_a, alpha_g = params
        relative_combo = manifest_path.parent.relative_to(output)
        global_manifest = rows.copy()
        for name, value in reversed(list(zip(["N", "m0", "dim", "alpha_a", "alpha_g"], params))):
            global_manifest.insert(0, name, value)
        global_manifest.insert(5, "source_path", [str(relative_combo / "prop" / filename) for filename in rows["filename"]])
        manifests.append(global_manifest)
        aggregates.append(_aggregate(rows, (N, m0, dim, alpha_a, alpha_g)))

    columns = ["N", "m0", "dim", "alpha_a", "alpha_g", "source_path", *VALUE_COLUMNS]
    manifest = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame(columns=columns)
    return manifest, pd.DataFrame(aggregates)


def processing_all_data(raw_root: str | Path, processed_root: str | Path | None = None):
    """Atualiza manifests de ``prop`` e agrega exclusivamente a partir de processed.

    A árvore bruta é usada somente para criar/atualizar ``prop_manifest.csv``.
    ``all_data.csv`` é sempre calculado dos manifests em ``processed/properties``;
    portanto pode ser regenerado mesmo depois de apagar os dados brutos.
    """
    raw_root = Path(raw_root).resolve()
    processed_root = Path(processed_root or raw_root.parent / "processed").resolve()
    output = processed_root / "properties"
    for prop in _prop_dirs(raw_root):
        relative = prop.relative_to(raw_root)
        per_combo = output / relative.parent / "prop_manifest.csv"
        existing = pd.read_csv(per_combo) if per_combo.exists() else pd.DataFrame(columns=VALUE_COLUMNS)
        done = set(existing.get("filename", pd.Series(dtype=str)).dropna().astype(str))
        new_paths = [path for path in sorted(prop.glob("*.csv")) if path.name not in done]
        if not new_paths:
            print(f"[SKIP] prop: {relative} (todos os CSVs já estão no manifesto)")
            continue
        rows = pd.DataFrame([_read_csv(path) for path in new_paths], columns=VALUE_COLUMNS)
        per_combo.parent.mkdir(parents=True, exist_ok=True)
        combined = rows if existing.empty else pd.concat([existing, rows], ignore_index=True)
        combined.to_csv(per_combo, index=False, na_rep="NaN")
        print(f"[OK] prop: {relative} ({len(rows)} novos) -> {per_combo.relative_to(processed_root)}")
    output.mkdir(parents=True, exist_ok=True)
    manifest, aggregate = _build_from_processed(output)
    manifest.to_csv(output / "manifest.csv", index=False, na_rep="NaN")
    aggregate.to_csv(output / "all_data.csv", index=False, na_rep="NaN")
    print(f"[OK] manifesto global: {output / 'manifest.csv'}")
    print(f"[OK] agregado: {output / 'all_data.csv'}")
    return aggregate.to_dict("list"), aggregate
