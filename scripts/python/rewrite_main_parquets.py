#!/usr/bin/env python3
import re
import ast
import argparse
from pathlib import Path
import pandas as pd

SEED_RE = re.compile(r"gml_(\d+)\.gml\.gz$")
NS_RE   = re.compile(r"_Ns_(\d+)\.parquet$")
ALPHA_DIR_RE  = re.compile(r"alpha_a_(\d+(?:\.\d+)?)_alpha_g_(\d+(?:\.\d+)?)")
ALPHA_NAME_RE = re.compile(r"alpha_a_(\d+(?:\.\d+)?)_alpha_g_(\d+(?:\.\d+)?)")


def parse_seed(value):
    if pd.isna(value):
        return value
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value)
    m = SEED_RE.search(s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", s)
    return int(m2.group(1)) if m2 else value


def ensure_listlike(v):
    if isinstance(v, (list, tuple)):
        return v
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, (list, tuple)):
                return parsed
        except Exception:
            return None
    return None


def transform_nodes(df):
    rename = {
        "ID_vertex": "Id",
        "degree": "deg",      # <-- aqui corrigido
        "deg": "deg",         # segurança
        "ID_sample": "seed",
        "ID_samples": "seed",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Garantir apenas uma coluna seed
    if "seed" in df.columns:
        df["seed"] = (
            df["seed"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype("int64")
        )

    # pos → x,y,z
    if "pos" in df.columns:
        df[["x", "y", "z"]] = pd.DataFrame(
            df["pos"].tolist(),
            index=df.index
        )
        df = df.drop(columns=["pos"])

    # Ordem final exata
    df = df[["Id", "x", "y", "z", "deg", "seed"]]

    return df

def transform_edges(df):
    # renomeia colunas
    rename = {
        "distance": "len",
        "ID_sample": "seed",
        "ID_samples": "seed",   # <-- correção importante
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # extrair valor numérico do seed
    if "seed" in df.columns:
        df["seed"] = (
            df["seed"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(int)
        )

    # dividir edge em source/target
    if "edge" in df.columns:
        df[["source", "target"]] = pd.DataFrame(
            df["edge"].tolist(),
            index=df.index
        )
        df = df.drop(columns=["edge"])

    # garantir ordem exata
    cols = ["source", "target", "len", "seed"]
    df = df[cols]

    return df


def extract_metadata(path: Path):
    parts = path.parts

    # N_100000 / m0_2 / dim_2
    N_part   = next(p for p in parts if p.startswith("N_"))
    m_part   = next(p for p in parts if p.startswith("m0_"))
    dim_part = next(p for p in parts if p.startswith("dim_"))

    N = N_part.split("_", 1)[1]
    m = m_part.split("_", 1)[1]
    dim = dim_part.split("_", 1)[1]

    # alpha_a_3.00_alpha_g_2.00 (pasta)
    alpha_a = None
    alpha_g = None
    for p in parts:
        mm = ALPHA_DIR_RE.fullmatch(p)
        if mm:
            alpha_a, alpha_g = mm.group(1), mm.group(2)
            break

    # fallback: tentar no nome do arquivo
    if alpha_a is None or alpha_g is None:
        mm = ALPHA_NAME_RE.search(path.name)
        if mm:
            alpha_a, alpha_g = mm.group(1), mm.group(2)

    if alpha_a is None or alpha_g is None:
        raise ValueError(f"Não consegui extrair alpha_a/alpha_g de: {path}")

    # Ns no filename
    ns_m = NS_RE.search(path.name)
    if not ns_m:
        raise ValueError(f"Não encontrei _Ns_XX no filename: {path.name}")
    Ns = int(ns_m.group(1))

    # kind pelo prefixo do arquivo
    if path.name.startswith("nodes_"):
        kind = "nodes"
    elif path.name.startswith("edges_"):
        kind = "edges"
    else:
        # se não começa com nodes_/edges_, tenta achar dentro do nome
        if "_nodes_" in path.name:
            kind = "nodes"
        elif "_edges_" in path.name:
            kind = "edges"
        else:
            raise ValueError(f"Não consegui inferir kind (nodes/edges): {path.name}")

    return N, m, dim, alpha_g, alpha_a, Ns, kind


def build_new_name(meta):
    N, m, dim, G, A, Ns, kind = meta

    G = float(G)
    A = float(A)

    return (
        f"N{N}_d{dim}_m{m}"
        f"_G{G:.1f}_A{A:.1f}"
        f"_seed001to{Ns:03d}_{kind}.parquet"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ex: ../../data")
    ap.add_argument("--output", required=True, help="ex: ../../data_rewrite")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(in_root.rglob("*.parquet"))
    if not files:
        print(f"Nenhum parquet em {in_root}")
        return

    ok = 0
    for f in files:
        try:
            meta = extract_metadata(f)
        except Exception:
            continue

        new_name = build_new_name(meta)
        dst = out_root / new_name
        if dst.exists() and not args.overwrite:
            continue

        df = pd.read_parquet(f)
        if meta[-1] == "nodes":
            df2 = transform_nodes(df)
        else:
            df2 = transform_edges(df)

        df2.to_parquet(dst, index=False)
        print(f"[OK] {f.name} -> {new_name}")
        ok += 1

    print(f"\nDone. Processed: {ok}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()