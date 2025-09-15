#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Varre:
  network/N_500/dim_2/alpha_a_*_alpha_g_*/gml/gml_*.gml(.gz)

Recria a estrutura em --out-dir e, para cada pasta de parâmetros (alpha_a_*_alpha_g_*),
salva em  network/N_500/dim_2/alpha_a_*_alpha_g_*/extract/ :

  - nodes_all.csv       (id, x, y, z)
  - edges_all.csv       (u, v, distance)

Opcional (--per-seed):
  - extract/seeds/nodes_{seed}.csv  (id, x, y, z)
  - extract/seeds/edges_{seed}.csv  (u, v, distance)
"""
from __future__ import annotations
import argparse
import gzip
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import networkx as nx


def load_gml_any(path: Path) -> nx.Graph:
    """Lê .gml ou .gml.gz e retorna um nx.Graph."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as gz:
            data = gz.read().decode("utf-8", errors="replace")
        return nx.parse_gml(data)
    data = path.read_text(encoding="utf-8", errors="replace")
    return nx.parse_gml(data)


def find_gml_files(base: Path) -> List[Path]:
    """Encontra .gml e .gml.gz no padrão desejado a partir de --base."""
    patt1 = "network/N_500/dim_2/alpha_a_*_alpha_g_*/gml/gml_*.gml"
    patt2 = "network/N_500/dim_2/alpha_a_*_alpha_g_*/gml/gml_*.gml.gz"
    return sorted(base.glob(patt1)) + sorted(base.glob(patt2))


def infer_seed(path: Path) -> str:
    """Extrai seed de gml_{seed}.gml(.gz)."""
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".gml"):
        name = name[:-4]
    return name.split("gml_", 1)[1] if "gml_" in name else name


def nodes_rows(G: nx.Graph):
    """Gera linhas (id,x,y,z) para nodes_all."""
    for n, attrs in G.nodes(data=True):
        g = attrs.get("graphics", {}) or {}
        x = g.get("x", attrs.get("x"))
        y = g.get("y", attrs.get("y"))
        z = g.get("z", attrs.get("z"))
        yield {"id": n, "x": x, "y": y, "z": z}


def edges_rows(G: nx.Graph):
    """Gera linhas (u,v,distance) para edges_all."""
    for u, v, attrs in G.edges(data=True):
        dist = attrs.get("distance")
        yield {"u": u, "v": v, "distance": dist}


def main():
    ap = argparse.ArgumentParser(description="Extrai nós/arestas em CSV, recriando a estrutura por parâmetros.")
    ap.add_argument("--base", type=Path, required=True,
                    help="Diretório que contém 'network/'. Ex.: /home/junior/Documents/SpatialPreferential")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Raiz onde a estrutura será recriada (network/…/alpha_a_*_alpha_g_*/extract/).")
    ap.add_argument("--per-seed", action="store_true",
                    help="Além dos agregados por pasta, salva arquivos individuais por seed.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Processar só os primeiros N arquivos (teste).")
    args = ap.parse_args()

    files = find_gml_files(args.base)
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f"[WARN] Nenhum arquivo GML encontrado em: {args.base}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Encontrados {len(files)} arquivo(s).")

    # Acumuladores por diretório de parâmetros
    nodes_by_param: Dict[Path, List[dict]] = {}
    edges_by_param: Dict[Path, List[dict]] = {}

    for i, src in enumerate(files, 1):
        seed = infer_seed(src)

        # Caminho relativo ao base: network/N_500/dim_2/alpha_dir/gml/gml_*.gml.gz
        rel = src.relative_to(args.base)
        # Diretório de parâmetros (sem o 'gml/'): network/N_500/dim_2/alpha_dir
        param_dir_rel = rel.parent.parent
        # Diretório de saída para este conjunto de parâmetros
        out_param_dir = args.out_dir / param_dir_rel / "extract"
        out_param_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(files)}] {src}  ->  {out_param_dir}  (seed={seed})")

        try:
            G = load_gml_any(src)
        except Exception as e:
            print(f"[ERRO] Falha ao ler {src}: {e}", file=sys.stderr)
            continue

        # Acumula linhas agregadas (sem seed/alphas)
        nodes_by_param.setdefault(out_param_dir, []).extend(nodes_rows(G))
        edges_by_param.setdefault(out_param_dir, []).extend(edges_rows(G))

        # (Opcional) por seed, sem seed na saída também
        if args.per_seed:
            seeds_dir = out_param_dir / "seeds"
            seeds_dir.mkdir(parents=True, exist_ok=True)
            dfN = pd.DataFrame(list(nodes_rows(G)))
            dfE = pd.DataFrame(list(edges_rows(G)))
            # Sem deduplicação; se quiser, habilite:
            # dfN = dfN.drop_duplicates(subset=["id","x","y","z"]).reset_index(drop=True)
            # dfE = dfE.drop_duplicates(subset=["u","v","distance"]).reset_index(drop=True)
            dfN.to_csv(seeds_dir / f"nodes_{seed}.csv", index=False)
            dfE.to_csv(seeds_dir / f"edges_{seed}.csv", index=False)

    # Grava agregados por pasta de parâmetros
    for out_param_dir, rows in nodes_by_param.items():
        dfN = pd.DataFrame(rows, columns=["id","x","y","z"])
        # Sem deduplicação entre seeds (IDs podem se repetir entre seeds)
        dfN.to_csv(out_param_dir / "nodes_all.csv", index=False)

    for out_param_dir, rows in edges_by_param.items():
        dfE = pd.DataFrame(rows, columns=["u","v","distance"])
        # Sem deduplicação entre seeds (arestas podem repetir entre seeds)
        dfE.to_csv(out_param_dir / "edges_all.csv", index=False)

    print(f"[OK] Extração finalizada em: {args.out_dir}")


if __name__ == "__main__":
    main()
