import re
import shutil
from pathlib import Path
from collections import defaultdict

PATTERN = re.compile(
    r"N(?P<N>\d+)_d(?P<dim>\d+)_m(?P<m>\d+)_G(?P<G>[\d\.]+)_A(?P<A>[\d\.]+)_seed\d+to(?P<Ns>\d+)_(?P<type>nodes|edges)\.parquet"
)


def deploy_parquets(
    rewrite_root="../../Data_Tsallis",
    data_root="../../data",
):
    rewrite_root = Path(rewrite_root).resolve()
    data_root = Path(data_root).resolve()

    files = list(rewrite_root.glob("*.parquet"))

    print(f"{len(files)} arquivos encontrados\n")

    # ------------------------------------------------------------
    # agrupar nodes + edges
    # ------------------------------------------------------------

    groups = defaultdict(dict)

    for f in files:

        m = PATTERN.match(f.name)
        if not m:
            print(f"SKIP (nome inesperado): {f.name}")
            continue

        key = (
            m.group("N"),
            m.group("dim"),
            m.group("m"),
            m.group("G"),
            m.group("A"),
            m.group("Ns"),
        )

        groups[key][m.group("type")] = f

    # ------------------------------------------------------------
    # transferir apenas pares completos
    # ------------------------------------------------------------

    for key, pair in groups.items():

        if "nodes" not in pair or "edges" not in pair:
            print("PAR INCOMPLETO, ignorando:", pair)
            continue

        N, dim, m0, G, A, Ns = key

        N = int(N)
        dim = int(dim)
        m0 = int(m0)
        alpha_g = float(G)
        alpha_a = float(A)

        target_dir = (
            data_root
            / f"N_{N}"
            / f"m0_{m0}"
            / f"dim_{dim}"
            / f"alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}"
        )

        if not target_dir.exists():
            print("PASTA NÃO EXISTE:", target_dir)
            continue

        # ------------------------------------------------------------
        # apagar parquets antigos
        # ------------------------------------------------------------

        for old in target_dir.glob("*.parquet"):
            old.unlink()

        # ------------------------------------------------------------
        # copiar par
        # ------------------------------------------------------------

        dst_nodes = target_dir / pair["nodes"].name
        dst_edges = target_dir / pair["edges"].name

        shutil.copy2(pair["nodes"], dst_nodes)
        shutil.copy2(pair["edges"], dst_edges)

        print("OK ->", dst_nodes)
        print("OK ->", dst_edges)

    print("\nConcluído.")


if __name__ == "__main__":
    deploy_parquets()