#!/usr/bin/env python3
"""Converte incrementalmente GMLs brutos em parquets em ``processed/parquets``."""
from __future__ import annotations

import argparse
import gzip
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ALPHA_RE = re.compile(r"^alpha_a_([+-]?\d+(?:\.\d+)?)_alpha_g_([+-]?\d+(?:\.\d+)?)$")
KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(.*)$")
SEED_RE = re.compile(r"gml_(\d+)(?:\.gml)?(?:\.gz)?$")
NODES_SCHEMA = pa.schema([("Id", pa.int64()), ("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64()), ("deg", pa.int64()), ("seed", pa.int64())])
EDGES_SCHEMA = pa.schema([("source", pa.int64()), ("target", pa.int64()), ("len", pa.float64()), ("seed", pa.int64())])
MANIFEST_COLUMNS = ["source_path", "nodes_file", "edges_file", "N", "m0", "dim", "alpha_a", "alpha_g", "seed", "nodes_rows", "edges_rows"]


def _value(raw: str):
    raw = raw.strip().strip('"')
    try:
        return int(raw) if "." not in raw and "e" not in raw.lower() else float(raw)
    except ValueError:
        return raw


def extract_gml(path: Path):
    """Lê um GML sem construir um grafo, retornando linhas de nodes e edges."""
    match = SEED_RE.search(path.name)
    seed = int(match.group(1)) if match else -1
    nodes, edges, stack = [], [], []
    node = edge = graphics = None
    pending = None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            if line in {"node", "edge", "graphics"}:
                pending = line
            elif line == "[" and pending:
                stack.append(pending)
                if pending == "node": node = {"Id": None, "x": None, "y": None, "z": None, "deg": None, "seed": seed}
                elif pending == "edge": edge = {"source": None, "target": None, "len": None, "seed": seed}
                else: graphics = {"x": None, "y": None, "z": None}
                pending = None
            elif line.endswith("["):
                stack.append(line[:-1].strip())
                pending = None
            elif line == "]":
                if not stack:
                    continue
                context = stack.pop()
                if context == "graphics" and node is not None and graphics is not None:
                    node.update(graphics); graphics = None
                elif context == "node" and node is not None:
                    nodes.append(node); node = None
                elif context == "edge" and edge is not None:
                    edges.append(edge); edge = None
            else:
                item = KEY_RE.match(line)
                if not item or not stack:
                    continue
                key, value, context = item.group(1), _value(item.group(2)), stack[-1]
                if context == "node" and node is not None:
                    if key == "id": node["Id"] = value
                    elif key == "degree": node["deg"] = value
                elif context == "graphics" and graphics is not None and key in graphics:
                    graphics[key] = value
                elif context == "edge" and edge is not None:
                    if key in ("source", "target"): edge[key] = value
                    elif key == "distance": edge["len"] = value
    return nodes, edges


def _combo(gml_dir: Path):
    alpha = ALPHA_RE.fullmatch(gml_dir.parent.name)
    if alpha is None:
        raise ValueError(f"diretório alpha inválido: {gml_dir.parent}")
    get = lambda prefix: int(next(part[len(prefix):] for part in gml_dir.parts if part.startswith(prefix)))
    return get("N_"), get("m0_"), get("dim_"), float(alpha.group(1)), float(alpha.group(2))


def _name(kind: str, N: int, m0: int, dim: int, alpha_a: float, alpha_g: float, samples: int) -> str:
    return f"N{N}_d{dim}_m{m0}_G{alpha_g:.1f}_A{alpha_a:.1f}_seed001to{samples:03d}_{kind}.parquet"


def _manifest_file(output: Path, record: dict, key: str, N: int, dim: int) -> Path:
    value = Path(str(record[key]))
    candidates = (output / value, output / f"N_{N}" / f"dim_{dim}" / value.name)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _copy_parquet(source: Path, writer: pq.ParquetWriter) -> None:
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=200_000):
        table = pa.Table.from_batches([batch])
        writer.write_table(table if table.schema == writer.schema else table.cast(writer.schema))


def _write_combo(raw_root: Path, gml_dir: Path, output: Path, existing: list[dict], force: bool) -> list[dict]:
    N, m0, dim, alpha_a, alpha_g = _combo(gml_dir)
    sources = sorted((*gml_dir.glob("*.gml.gz"), *gml_dir.glob("*.gml")))
    done = {record["source_path"] for record in existing}
    new_sources = sources if force else [source for source in sources if str(source.relative_to(raw_root)) not in done]
    if not new_sources and not force:
        print(f"[SKIP] gml: {gml_dir.relative_to(raw_root)} (todos no manifesto)", flush=True)
        return existing

    output_dir = output / f"N_{N}" / f"dim_{dim}"
    output_dir.mkdir(parents=True, exist_ok=True)
    old_nodes = old_edges = None
    if existing and not force:
        old_nodes = _manifest_file(output, existing[0], "nodes_file", N, dim)
        old_edges = _manifest_file(output, existing[0], "edges_file", N, dim)
        if not old_nodes.exists() or not old_edges.exists():
            raise RuntimeError(f"Manifesto indica GMLs processados, mas parquet ausente em {gml_dir.parent}")

    total = len(sources) if force else len(existing) + len(new_sources)
    node_target = output_dir / _name("nodes", N, m0, dim, alpha_a, alpha_g, total)
    edge_target = output_dir / _name("edges", N, m0, dim, alpha_a, alpha_g, total)
    node_tmp, edge_tmp = node_target.with_suffix(".parquet.tmp"), edge_target.with_suffix(".parquet.tmp")
    records = [] if force else list(existing)
    print(f"[PROCESSANDO] {gml_dir.relative_to(raw_root)}: {len(new_sources)} novos de {len(sources)} GMLs", flush=True)
    try:
        with pq.ParquetWriter(node_tmp, NODES_SCHEMA, compression="snappy") as nodes_writer, pq.ParquetWriter(edge_tmp, EDGES_SCHEMA, compression="snappy") as edges_writer:
            if old_nodes is not None:
                _copy_parquet(old_nodes, nodes_writer)
                _copy_parquet(old_edges, edges_writer)
            for source in new_sources:
                nodes, edges = extract_gml(source)
                if nodes: nodes_writer.write_table(pa.Table.from_pylist(nodes, schema=NODES_SCHEMA))
                if edges: edges_writer.write_table(pa.Table.from_pylist(edges, schema=EDGES_SCHEMA))
                records.append({"source_path": str(source.relative_to(raw_root)), "nodes_file": "", "edges_file": "", "N": N, "m0": m0, "dim": dim,
                                "alpha_a": alpha_a, "alpha_g": alpha_g, "seed": nodes[0]["seed"] if nodes else None,
                                "nodes_rows": len(nodes), "edges_rows": len(edges)})
        os.replace(node_tmp, node_target)
        os.replace(edge_tmp, edge_target)
    except Exception:
        node_tmp.unlink(missing_ok=True); edge_tmp.unlink(missing_ok=True)
        raise
    nodes_file, edges_file = str(node_target.relative_to(output)), str(edge_target.relative_to(output))
    for record in records:
        record["nodes_file"], record["edges_file"] = nodes_file, edges_file
    for old in {old_nodes, old_edges} - {None, node_target, edge_target}:
        old.unlink(missing_ok=True)
    print(f"[OK] gml: {gml_dir.relative_to(raw_root)} -> {nodes_file}, {edges_file}", flush=True)
    return records


def _write_combo_worker(raw_root: str, gml_dir: str, output: str, existing: list[dict], force: bool) -> list[dict]:
    return _write_combo(Path(raw_root), Path(gml_dir), Path(output), existing, force)


def _default_workers() -> int:
    return min(4, os.cpu_count() or 1)


def rewrite_parquets(raw_root: str | Path, processed_root: str | Path, force: bool = False, workers: int | None = None):
    """Processa apenas GMLs ausentes do manifesto, salvo com ``force=True``."""
    raw_root, output = Path(raw_root).resolve(), Path(processed_root).resolve() / "parquets"
    workers = _default_workers() if workers is None else workers
    if workers < 1:
        raise ValueError("workers deve ser pelo menos 1")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    old_manifest = pd.read_csv(manifest_path).fillna("") if manifest_path.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
    gml_dirs = sorted(raw_root.glob("N_*/m0_*/dim_*/alpha_a_*_alpha_g_*/gml"))
    active = {str(directory.relative_to(raw_root)) for directory in gml_dirs}
    retained = [row for row in old_manifest.to_dict("records") if not any(str(row.get("source_path", "")).startswith(prefix + "/") for prefix in active)]
    print(f"[INFO] {len(gml_dirs)} pastas gml; force={force}; workers={workers}", flush=True)

    jobs = []
    for directory in gml_dirs:
        prefix = str(directory.relative_to(raw_root)) + "/"
        existing = [row for row in old_manifest.to_dict("records") if str(row.get("source_path", "")).startswith(prefix)]
        jobs.append((directory, existing))
    records = list(retained)
    if workers == 1:
        for directory, existing in jobs:
            records.extend(_write_combo(raw_root, directory, output, existing, force))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(_write_combo_worker, str(raw_root), str(directory), str(output), existing, force): directory for directory, existing in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                records.extend(future.result())
                print(f"[CONCLUÍDO {completed}/{len(jobs)}] {futures[future].relative_to(raw_root)}", flush=True)
    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS).sort_values("source_path")
    manifest.to_csv(manifest_path, index=False)
    print(f"[OK] manifesto de parquets: {manifest_path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="reprocessa todos os GMLs")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    rewrite_parquets(args.input, args.output, args.force or args.overwrite, args.workers)


if __name__ == "__main__":
    main()
