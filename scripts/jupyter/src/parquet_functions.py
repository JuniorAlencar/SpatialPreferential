# gml_parquet_extractor.py
# -*- coding: utf-8 -*-

import gzip
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow.parquet as pq
import pyarrow as pa
import gc
import os
import shutil
import pandas as pd


# -------------------------
# Regex / parsing helpers
# -------------------------
ALPHA_DIR_RE = re.compile(
    r"^alpha_a_([0-9]+(?:\.[0-9]+)?)_alpha_g_([0-9]+(?:\.[0-9]+)?)$"
)

KV_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s+(.*)\s*$')


def fmt2(x: float) -> str:
    return f"{x:.2f}"


def parse_combo_from_path(alpha_dir: Path):
    """
    alpha_dir: .../N_{N}/m0_{m0}/dim_{dim}/alpha_a_{a}_alpha_g_{g}
    Retorna: (m0, dim, alpha_a, alpha_g) ou (None,...) se não achar.
    """
    m0 = dim = alpha_a = alpha_g = None

    for part in alpha_dir.parts:
        if part.startswith("m0_"):
            try:
                m0 = int(part.split("m0_")[1])
            except Exception:
                pass
        elif part.startswith("dim_"):
            try:
                dim = int(part.split("dim_")[1])
            except Exception:
                pass
        else:
            m = ALPHA_DIR_RE.match(part)
            if m:
                alpha_a = float(m.group(1))
                alpha_g = float(m.group(2))

    return m0, dim, alpha_a, alpha_g


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    # int
    try:
        if "." not in raw and "e" not in raw.lower():
            return int(raw)
    except Exception:
        pass
    # float
    try:
        return float(raw)
    except Exception:
        return raw


def extract_gml_gz_light(fp: Path):
    """
    Extrai nodes/edges de um .gml.gz (ou .gml) sem construir grafo.

    OUTPUT NOVO:
      nodes: Id, x, y, z, deg, seed(int)
      edges: source, target, len, seed(int)

    OBS: manifest continua trabalhando com fp.name (ex: gml_123.gml.gz)
    """
    sample_name = fp.name  # ex: gml_123.gml.gz

    # seed numérica a partir do nome
    mseed = re.search(r"gml_(\d+)\.gml\.gz$", sample_name)
    if mseed:
        seed_num = int(mseed.group(1))
    else:
        # fallback: primeiro bloco numérico
        m2 = re.search(r"(\d+)", sample_name)
        seed_num = int(m2.group(1)) if m2 else -1

    node_rows = []
    edge_rows = []

    stack = []
    cur_node = None
    cur_edge = None
    cur_graphics = None
    pending_key = None

    if fp.suffix == ".gz":
        opener = lambda: gzip.open(fp, "rt", encoding="utf-8", errors="replace")
    else:
        opener = lambda: fp.open("rt", encoding="utf-8", errors="replace")

    def open_block(key: str):
        nonlocal cur_node, cur_edge, cur_graphics
        stack.append(key)

        if key == "node":
            cur_node = {
                "Id": None,
                "x": None,
                "y": None,
                "z": None,
                "deg": None,
                "seed": seed_num,
            }
        elif key == "edge":
            cur_edge = {
                "source": None,
                "target": None,
                "len": None,
                "seed": seed_num,
            }
        elif key == "graphics":
            cur_graphics = {"x": None, "y": None, "z": None}

    with opener() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s in ("node", "edge", "graphics"):
                pending_key = s
                continue

            if s == "[":
                if pending_key is not None:
                    open_block(pending_key)
                    pending_key = None
                continue

            if s.endswith("["):
                key = s[:-1].strip()
                if key in ("node", "edge", "graphics"):
                    open_block(key)
                    pending_key = None
                    continue
                stack.append(key)
                pending_key = None
                continue

            if s.startswith("]"):
                if not stack:
                    pending_key = None
                    continue

                key = stack.pop()

                if key == "graphics" and cur_node is not None and cur_graphics is not None:
                    cur_node["x"] = cur_graphics.get("x")
                    cur_node["y"] = cur_graphics.get("y")
                    cur_node["z"] = cur_graphics.get("z")
                    cur_graphics = None

                elif key == "node" and cur_node is not None:
                    node_rows.append(cur_node)
                    cur_node = None

                elif key == "edge" and cur_edge is not None:
                    edge_rows.append(cur_edge)
                    cur_edge = None

                pending_key = None
                continue

            m = KV_RE.match(s)
            if not m or not stack:
                continue

            k = m.group(1)
            v = _parse_value(m.group(2))
            ctx = stack[-1]

            if ctx == "node" and cur_node is not None:
                if k == "id":
                    cur_node["Id"] = v
                elif k == "degree":
                    cur_node["deg"] = v

            elif ctx == "graphics" and cur_graphics is not None:
                if k in ("x", "y", "z"):
                    cur_graphics[k] = v

            elif ctx == "edge" and cur_edge is not None:
                if k == "source":
                    cur_edge["source"] = v
                elif k == "target":
                    cur_edge["target"] = v
                elif k == "distance":
                    cur_edge["len"] = v

    return node_rows, edge_rows, sample_name


# -------------------------
# Parquet helpers
# -------------------------
def find_existing_parquet(save_dir: Path, kind: str, N: int, dim: int, m0: int, alpha_a: float, alpha_g: float):
    pat = f"N{N}_d{dim}_m{m0}_G{fmt1(alpha_g)}_A{fmt1(alpha_a)}_seed001to*_{kind}.parquet"
    candidates = list(save_dir.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def remove_old_parquets(save_dir: Path, kind: str, N: int, m0: int, alpha_a_str: str, alpha_g_str: str):
    pat = f"{kind}_N_{N}_m0_{m0}_alpha_a_{alpha_a_str}_alpha_g_{alpha_g_str}_Ns_*.parquet"
    for p in save_dir.glob(pat):
        try:
            p.unlink()
        except Exception:
            pass


def _worker_extract(fp_str: str):
    fp = Path(fp_str)
    return extract_gml_gz_light(fp)


def extract_new_files(
    new_files,
    use_parallel: bool = True,
    max_workers: int | None = None,
    progress=None,
):
    """
    Extrai rows somente para new_files.
    progress: função opcional para atualizar barra (ex.: tqdm), recebe iterável de futures.
    """
    node_rows_all = []
    edge_rows_all = []

    if not new_files:
        return node_rows_all, edge_rows_all

    if use_parallel and len(new_files) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_worker_extract, str(fp)) for fp in new_files]
            it = futures
            if progress is not None:
                it = progress(as_completed(futures), total=len(futures))
            else:
                it = as_completed(futures)

            for fut in it:
                node_rows, edge_rows, _ = fut.result()
                node_rows_all.extend(node_rows)
                edge_rows_all.extend(edge_rows)
    else:
        it = new_files
        if progress is not None:
            it = progress(new_files, total=len(new_files))
        for fp in it:
            node_rows, edge_rows, _ = extract_gml_gz_light(fp)
            node_rows_all.extend(node_rows)
            edge_rows_all.extend(edge_rows)

    return node_rows_all, edge_rows_all


def _write_table_append(writer: pq.ParquetWriter, table: pa.Table) -> None:
    # garante schema compatível
    if table.schema != writer.schema:
        table = table.cast(writer.schema)
    writer.write_table(table)

def _safe_remove_old_parquets_except(
    save_dir: Path,
    kind: str,
    N_value: int,
    dim: int,
    m0: int,
    alpha_a: float,
    alpha_g: float,
    keep: Path | None,
):
    pat = f"N{N_value}_d{dim}_m{m0}_G{fmt1(alpha_g)}_A{fmt1(alpha_a)}_seed001to*_{kind}.parquet"
    for p in save_dir.glob(pat):
        if keep is not None and p.resolve() == keep.resolve():
            continue
        try:
            p.unlink()
        except Exception:
            pass

def rewrite_parquet_streaming(old_path: Path | None, new_df: pd.DataFrame, out_path: Path) -> None:
    """
    Reescreve parquet monolítico SEM carregar tudo em RAM.
    - Se old_path existir: copia em batches para out_path
    - Escreve new_df ao final
    - out_path é o novo arquivo (escreve primeiro como .tmp e depois renomeia)
    """
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    # schema baseado no new_df se não houver old
    writer = None
    try:
        if old_path is not None and old_path.exists():
            pf = pq.ParquetFile(old_path)
            schema = pf.schema_arrow
            writer = pq.ParquetWriter(tmp_path, schema=schema, compression="snappy")

            # copia o antigo em batches
            for batch in pf.iter_batches(batch_size=200_000):  # ajuste se quiser
                table = pa.Table.from_batches([batch], schema=schema)
                _write_table_append(writer, table)
        else:
            # cria schema a partir do new_df
            table_new = pa.Table.from_pandas(new_df, preserve_index=False)
            writer = pq.ParquetWriter(tmp_path, schema=table_new.schema, compression="snappy")

        # escreve o novo
        table_new = pa.Table.from_pandas(new_df, preserve_index=False)
        _write_table_append(writer, table_new)

    finally:
        if writer is not None:
            writer.close()

    # troca atômica: remove antigo e renomeia tmp
    if old_path is not None and old_path.exists():
        old_path.unlink()
    os.replace(tmp_path, out_path)

    # ajuda a liberar memória cedo
    del new_df
    gc.collect()

def _rewrite_parquet_streaming_append_delta(
    old_path: Path | None,
    delta_df: pd.DataFrame,
    out_path: Path,
    batch_size: int = 200_000,
):
    """
    Cria out_path (novo parquet monolítico) copiando old_path em batches e
    escrevendo delta_df ao final. NÃO carrega o old inteiro em RAM.

    Escreve primeiro em .tmp e depois faz replace atômico.
    """
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    writer = None
    try:
        if old_path is not None and old_path.exists():
            pf = pq.ParquetFile(old_path)
            schema = pf.schema_arrow
            writer = pq.ParquetWriter(tmp_path, schema=schema, compression="snappy")

            # copia antigo em batches
            for batch in pf.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch], schema=schema)
                if table.schema != writer.schema:
                    table = table.cast(writer.schema)
                writer.write_table(table)

            # escreve delta
            if delta_df is not None and len(delta_df) > 0:
                table_new = pa.Table.from_pandas(delta_df, preserve_index=False)
                if table_new.schema != writer.schema:
                    table_new = table_new.cast(writer.schema)
                writer.write_table(table_new)

        else:
            # sem old -> cria schema do delta e escreve só delta
            if delta_df is None or len(delta_df) == 0:
                raise RuntimeError("delta_df vazio e old_path inexistente; nada a escrever.")
            table_new = pa.Table.from_pandas(delta_df, preserve_index=False)
            writer = pq.ParquetWriter(tmp_path, schema=table_new.schema, compression="snappy")
            writer.write_table(table_new)

    finally:
        if writer is not None:
            writer.close()

    os.replace(tmp_path, out_path)

    # ajuda GC
    gc.collect()

def fmt1(x: float) -> str:
    return f"{x:.1f}"

def parquet_name_new(kind: str, N: int, dim: int, m0: int, alpha_a: float, alpha_g: float, Ns: int) -> str:
    # kind: "nodes" ou "edges"
    return f"N{N}_d{dim}_m{m0}_G{fmt1(alpha_g)}_A{fmt1(alpha_a)}_seed001to{Ns:03d}_{kind}.parquet"

# -------------------------
# Main entrypoint for notebook
# -------------------------
def update_parquets_for_N(
    N_value: int,
    base_data_dir: str | Path = "../../data",
    use_parallel: bool = True,
    max_workers: int | None = None,
    tqdm_outer=None,
    tqdm_inner=None,
    debug: bool = False,
    batch_size: int = 200_000,  # streaming copy batch
):
    """
    Pipeline por combo:
      - usa manifest .txt para saber samples já processadas
      - extrai apenas samples novas (parser leve)
      - atualiza parquets monolíticos via streaming (sem explodir RAM)
      - atualiza manifest

    Mantém:
      nodes parquet cols: ID_vertex, pos, degree, ID_sample
      edges parquet cols: edge, distance, ID_samples
    """
    base = (Path(base_data_dir) / f"N_{N_value}").resolve()
    if not base.exists():
        raise RuntimeError(f"Pasta base não existe: {base}")

    # pega SOMENTE o padrão correto
    combo_gml_dirs = sorted(base.glob("m0_*/dim_*/alpha_a_*_alpha_g_*/gml"))
    if not combo_gml_dirs:
        raise RuntimeError(f"Nenhuma pasta 'gml' encontrada dentro de {base}")

    outer_it = combo_gml_dirs if tqdm_outer is None else tqdm_outer(combo_gml_dirs, desc=f"Combinações para N={N_value}", unit="combo")

    for gml_dir in outer_it:
        save_dir = gml_dir.parent  # .../alpha_a_*_alpha_g_*

        m0, dim, alpha_a, alpha_g = parse_combo_from_path(save_dir)
        print(
            f"[PROCESSANDO] "
            f"N={N_value} | m0={m0} | dim={dim} | "
            f"alpha_a={alpha_a:.2f} | alpha_g={alpha_g:.2f}"
        )
        if (m0 is None) or (dim is None) or (alpha_a is None) or (alpha_g is None):
            if debug:
                print(f"[SKIP] combo inválida: {save_dir}")
            continue

        alpha_a_str = fmt2(alpha_a)
        alpha_g_str = fmt2(alpha_g)

        if tqdm_outer is not None:
            outer_it.set_postfix_str(f"m0={m0}, dim={dim}, aa={alpha_a_str}, ag={alpha_g_str}")

        gml_files = sorted(gml_dir.glob("*.gml.gz"))
        if not gml_files:
            if debug:
                print(f"[SKIP] sem .gml.gz: {gml_dir}")
            continue

        # ---------------------------
        # Check via manifest
        # ---------------------------
        mf = manifest_path(save_dir, N_value, m0, alpha_a_str, alpha_g_str)
        done = read_manifest(mf)

        # Se manifest não existe, cria a partir do parquet existente (uma vez)
        if not mf.exists():
            nodes_pq0 = find_existing_parquet(save_dir, "nodes", N_value, dim, m0, alpha_a, alpha_g)
            edges_pq0 = find_existing_parquet(save_dir, "edges", N_value, dim, m0, alpha_a, alpha_g)

            samples = []

            if nodes_pq0 is not None:
                try:
                    # novo: seed numérica
                    df = pd.read_parquet(nodes_pq0, columns=["seed"])
                    seeds = df["seed"].dropna().astype(int).unique().tolist()
                    samples = [f"gml_{s}.gml.gz" for s in seeds]
                except Exception:
                    # velho: ID_sample com string gml_*.gml.gz
                    try:
                        df = pd.read_parquet(nodes_pq0, columns=["ID_sample"])
                        samples = df["ID_sample"].astype(str).unique().tolist()
                    except Exception:
                        samples = []

            elif edges_pq0 is not None:
                try:
                    df = pd.read_parquet(edges_pq0, columns=["seed"])
                    seeds = df["seed"].dropna().astype(int).unique().tolist()
                    samples = [f"gml_{s}.gml.gz" for s in seeds]
                except Exception:
                    try:
                        df = pd.read_parquet(edges_pq0, columns=["ID_samples"])
                        samples = df["ID_samples"].astype(str).unique().tolist()
                    except Exception:
                        samples = []

            if samples:
                write_manifest(mf, samples)
                done = set(samples)
                if debug:
                    print(f"[INFO] manifest criado via parquet: {mf}")

        new_files = [fp for fp in gml_files if fp.name not in done]
        if not new_files:
            continue

        # ---------------------------
        # Extrai somente novos
        # ---------------------------
        def _progress_iter(it, total):
            if tqdm_inner is None:
                return it
            return tqdm_inner(
                it,
                total=total,
                desc=f"Extraindo (m0={m0}, dim={dim}, aa={alpha_a_str}, ag={alpha_g_str})",
                unit="arquivo",
                leave=False
            )

        node_rows, edge_rows = extract_new_files(
            new_files,
            use_parallel=use_parallel,
            max_workers=max_workers,
            progress=lambda it, total: _progress_iter(it, total),
        )

        if len(node_rows) == 0 and len(edge_rows) == 0:
            if debug:
                print(f"[ERRO] rows vazias em {save_dir} (isso não deveria acontecer)")
            continue

        # DataFrames delta (só novos)
        # DataFrames delta (só novos) - FORMATO NOVO + ORDEM
        df_nodes_new = (
            pd.DataFrame(node_rows, columns=["Id", "x", "y", "z", "deg", "seed"])
            if node_rows else
            pd.DataFrame(columns=["Id", "x", "y", "z", "deg", "seed"])
        )

        df_edges_new = (
            pd.DataFrame(edge_rows, columns=["source", "target", "len", "seed"])
            if edge_rows else
            pd.DataFrame(columns=["source", "target", "len", "seed"])
        )

        # ---------------------------
        # Descobre Ns antigo sem carregar tudo
        # ---------------------------
        nodes_pq = find_existing_parquet(save_dir, "nodes", N_value, dim, m0, alpha_a, alpha_g)
        edges_pq = find_existing_parquet(save_dir, "edges", N_value, dim, m0, alpha_a, alpha_g)

        Ns_old = len(done)  # manifest é a fonte de verdade
        Ns = Ns_old + len(new_files)

        nodes_out = save_dir / parquet_name_new("nodes", N_value, dim, m0, alpha_a, alpha_g, Ns)
        edges_out = save_dir / parquet_name_new("edges", N_value, dim, m0, alpha_a, alpha_g, Ns)

        # ---------------------------
        # STREAMING rewrite (old + delta)
        # ---------------------------
        _rewrite_parquet_streaming_append_delta(nodes_pq, df_nodes_new, nodes_out, batch_size=batch_size)
        _rewrite_parquet_streaming_append_delta(edges_pq, df_edges_new, edges_out, batch_size=batch_size)

        # remove antigos, preservando o recém-criado
        _safe_remove_old_parquets_except(save_dir, "nodes", N_value, dim, m0, alpha_a, alpha_g, keep=nodes_out)
        _safe_remove_old_parquets_except(save_dir, "edges", N_value, dim, m0, alpha_a, alpha_g, keep=edges_out)

        # ---------------------------
        # Atualiza manifest
        # ---------------------------
        done_updated = set(done)
        done_updated.update([fp.name for fp in new_files])
        write_manifest(mf, done_updated)

        # limpeza agressiva por combo (evita acumular)
        del node_rows, edge_rows, df_nodes_new, df_edges_new
        gc.collect()

    return True

from typing import Iterable

def manifest_path(save_dir: Path, N_value: int, m0: int, alpha_a_str: str, alpha_g_str: str) -> Path:
    """
    Manifesto por combo, salvo na pasta alpha_a_*_alpha_g_* (save_dir).
    Nome fixo (sem Ns) para não precisar renomear.
    """
    return save_dir / f"manifest_N_{N_value}_m0_{m0}_alpha_a_{alpha_a_str}_alpha_g_{alpha_g_str}.txt"


def read_manifest(manifest_fp: Path) -> set[str]:
    """
    Lê manifest (1 sample por linha). Retorna set de nomes.
    """
    if not manifest_fp.exists():
        return set()
    # lê rápido e remove vazios
    lines = manifest_fp.read_text(encoding="utf-8", errors="replace").splitlines()
    return {ln.strip() for ln in lines if ln.strip()}


def write_manifest(manifest_fp: Path, samples: Iterable[str]) -> None:
    """
    Escreve manifest ordenado, 1 por linha.
    """
    samples_sorted = sorted(set(samples))
    manifest_fp.write_text("\n".join(samples_sorted) + ("\n" if samples_sorted else ""), encoding="utf-8")


def create_manifest_from_existing_parquets(
    N_value: int,
    base_data_dir: str | Path = "../../data",
    tqdm_outer=None,
    debug: bool = False,
) -> int:
    """
    RODE UMA VEZ (bootstrap):
    Para cada combo, se existir parquet de nodes/edges, cria manifest .txt contendo os nomes
    (preferencialmente de nodes; se não existir, usa edges).

    Retorna: número de manifests criados/atualizados.
    """
    base = (Path(base_data_dir) / f"N_{N_value}").resolve()
    if not base.exists():
        raise RuntimeError(f"Pasta base não existe: {base}")

    combo_gml_dirs = sorted(base.glob("m0_*/dim_*/alpha_a_*_alpha_g_*/gml"))
    if not combo_gml_dirs:
        raise RuntimeError(f"Nenhuma pasta 'gml' encontrada dentro de {base}")

    it = combo_gml_dirs if tqdm_outer is None else tqdm_outer(combo_gml_dirs, desc=f"Manifests para N={N_value}", unit="combo")
    n_written = 0

    for gml_dir in it:
        save_dir = gml_dir.parent
        m0, dim, alpha_a, alpha_g = parse_combo_from_path(save_dir)
        if (m0 is None) or (dim is None) or (alpha_a is None) or (alpha_g is None):
            if debug:
                print(f"[SKIP] combo inválida: {save_dir}")
            continue

        alpha_a_str = fmt2(alpha_a)
        alpha_g_str = fmt2(alpha_g)

        nodes_pq = find_existing_parquet(save_dir, "nodes", N_value, m0, alpha_a_str, alpha_g_str)
        edges_pq = find_existing_parquet(save_dir, "edges", N_value, m0, alpha_a_str, alpha_g_str)

        samples = None

        if nodes_pq is not None:
            df = pd.read_parquet(nodes_pq, columns=["ID_sample"])
            samples = df["ID_sample"].astype(str).unique().tolist()
        elif edges_pq is not None:
            df = pd.read_parquet(edges_pq, columns=["ID_samples"])
            samples = df["ID_samples"].astype(str).unique().tolist()

        if not samples:
            if debug:
                print(f"[SKIP] sem parquet em {save_dir}")
            continue

        mf = manifest_path(save_dir, N_value, m0, alpha_a_str, alpha_g_str)
        write_manifest(mf, samples)
        n_written += 1

    return n_written



def _is_manifest_txt(fname: str) -> bool:
    return fname.startswith("manifest_") and fname.endswith(".txt")


def _safe_prefix_from_relpath(relpath: Path) -> str:
    """
    Junta os diretórios usando apenas um '_' como separador,
    sem criar '__' extras.
    """
    parts = [p for p in relpath.parts if p not in ("", ".")]
    return "_".join(parts) if parts else "ROOT"

# aceita seu novo padrão (nodes|edges)
NEW_PQ_RE = re.compile(
    r"^N(?P<N>\d+)_d(?P<dim>\d+)_m(?P<m>\d+)_G(?P<G>\d+(?:\.\d+)?)_A(?P<A>\d+(?:\.\d+)?)_seed001to(?P<Ns>\d+)_(?P<kind>nodes|edges)\.parquet$"
)

def _extract_dim_from_new_parquet_name(fname: str) -> int | None:
    m = NEW_PQ_RE.match(fname)
    if not m:
        return None
    return int(m.group("dim"))


def copy_all_parquet_flat_by_dim(root_dir, out_dir="../../Data_Tsallis"):
    """
    NOVA ESTRUTURA:
      - Copia apenas *.parquet no padrão novo:
          N{N}_d{dim}_m{m}_G{G}_A{A}_seed001to{Ns}_{nodes|edges}.parquet
      - Organiza em:
          out_dir/Dim_<dim>/
      - NÃO copia manifest.
      - NÃO prefixa nome com caminho relativo (não precisa mais).
    """
    root_dir = Path(root_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_pattern = 0
    created_dirs = 0

    for dirpath, _, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)

        for fname in filenames:
            if not fname.endswith(".parquet"):
                continue

            dim = _extract_dim_from_new_parquet_name(fname)
            if dim is None:
                skipped_pattern += 1
                continue

            dst_dim_dir = out_dir / f"Dim_{dim}"
            if not dst_dim_dir.exists():
                dst_dim_dir.mkdir(parents=True, exist_ok=True)
                created_dirs += 1

            src = dirpath / fname
            dst = dst_dim_dir / fname

            shutil.copy2(src, dst)
            copied += 1

    print(f"[OK] Flat copy by Dim -> {out_dir}")
    print(f"     dirs_created={created_dirs} files_copied={copied} skipped_pattern={skipped_pattern}")

def copy_tree_only_new_parquet(root_dir, out_root="../../Data_Network"):
    """
    NOVA ESTRUTURA (mirrored):
      - Espelha a estrutura inteira de root_dir -> out_root
      - Copia APENAS *.parquet que batem o padrão novo (NEW_PQ_RE)
      - NÃO copia manifest
    """
    root_dir = Path(root_dir).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    created_dirs = 0
    skipped_pattern = 0

    for dirpath, _, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)
        rel = dirpath.relative_to(root_dir)
        dst_dir = out_root / rel

        selected = []
        for f in filenames:
            if not f.endswith(".parquet"):
                continue
            if _extract_dim_from_new_parquet_name(f) is None:
                skipped_pattern += 1
                continue
            selected.append(f)

        if not selected:
            continue

        if not dst_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            created_dirs += 1

        for fname in selected:
            shutil.copy2(dirpath / fname, dst_dir / fname)
            copied += 1

    print(f"[OK] Mirrored tree (new parquet only) -> {out_root}")
    print(f"     dirs_created={created_dirs} files_copied={copied} skipped_pattern={skipped_pattern}")

def copy_tree_new_parquet_grouped_by_dim(root_dir, out_root="../../Data_Network"):
    """
    NOVA ESTRUTURA (Dim + mirrored):
      - Copia APENAS *.parquet padrão novo
      - Organiza como:
          out_root/Dim_<dim>/<estrutura_relativa>/<arquivo.parquet>
      - NÃO copia manifest
    """
    root_dir = Path(root_dir).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    created_dirs = 0
    skipped_pattern = 0

    for dirpath, _, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)
        rel = dirpath.relative_to(root_dir)

        for fname in filenames:
            if not fname.endswith(".parquet"):
                continue

            dim = _extract_dim_from_new_parquet_name(fname)
            if dim is None:
                skipped_pattern += 1
                continue

            dst_dir = out_root / f"Dim_{dim}" / rel
            if not dst_dir.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                created_dirs += 1

            shutil.copy2(dirpath / fname, dst_dir / fname)
            copied += 1

    print(f"[OK] Tree copy grouped by Dim -> {out_root}")
    print(f"     dirs_created={created_dirs} files_copied={copied} skipped_pattern={skipped_pattern}")

def remove_all_gml_gz(root_dir):
    """
    Percorre a estrutura de diretórios a partir de root_dir,
    entra em cada pasta chamada 'gml' e remove todos os arquivos *.gml.gz.
    """

    root_dir = Path(root_dir).resolve()

    removed = 0
    scanned_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)

        # só agir se a pasta se chama 'gml'
        if dirpath.name != "gml":
            continue

        scanned_dirs += 1

        for fname in filenames:
            if fname.endswith(".gml.gz"):
                fpath = dirpath / fname
                try:
                    fpath.unlink()
                    removed += 1
                except Exception as e:
                    print(f"ERRO ao remover {fpath}: {e}")

    print(f"[OK] Limpeza concluída")
    print(f"     pastas_gml_verificadas={scanned_dirs}")
    print(f"     arquivos_removidos={removed}")