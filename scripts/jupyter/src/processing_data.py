import os
import csv
import glob
import re
from math import nan
import pandas as pd
import numpy as np

# =========================
# Auxiliares já usadas antes
# =========================
def create_folders(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    print(f"Directory '{folder_path}' ensured to exist.")

dict_keys = ["filename","ShortestCost","ShortestBFS",
             "R_Spearman","R_Spearman_err","R_Newman","R_Newman_err",
             "ClusterCoefficient"]

def _new_dict_props():
    return {k: [] for k in dict_keys}

def _norm(s: str) -> str:
    """Normaliza nomes do cabeçalho para casar de forma robusta."""
    return re.sub(r'[^a-z0-9]+', '', s.lower())

NORM_MAP = {
    _norm('MeanShortestPathDijstrika'): 'ShortestCost',
    _norm('MeanShortestPathBFS'):       'ShortestBFS',
    _norm('Ass_Spearman'):              'R_Spearman',
    _norm('Ass_Spearman_error'):        'R_Spearman_err',
    _norm('Ass_Newman'):                'R_Newman',
    _norm('Ass_Newman_error'):          'R_Newman_err',
    _norm('ClusterCoefficient'):        'ClusterCoefficient',
}

# --- helper: carrega all_data.dat existente (se houver) ---
def _load_existing_all_data(out_path: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            df = pd.read_csv(out_path)  # sep=',' padrão, como salvamos
            # garante que a coluna 'filename' exista
            if "filename" not in df.columns:
                return None
            return df
        except Exception:
            return None
    return None

# --- helper: lista apenas CSVs "novos" (não presentes no all_data.dat) ---
def _list_new_csvs(prop_dir: str, existing_df: pd.DataFrame | None):
    all_csvs = sorted(glob.glob(os.path.join(prop_dir, "*.csv")))
    if existing_df is None or existing_df.empty:
        return all_csvs  # tudo é novo

    already = set(existing_df["filename"].astype(str).str.strip())
    new_files = [fp for fp in all_csvs if os.path.basename(fp) not in already]
    return new_files

def _read_header_and_first_data(csv_path):
    """Retorna (header:list[str], data:list[str]) pegando:
       - primeira linha não-vazia como header (removendo '#')
       - primeira linha de dados não-vazia em seguida
    """
    header = None
    data = None
    with open(csv_path, 'rt', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for row in reader:
            if row and any(c.strip() for c in row):
                header = [c.strip().lstrip('#') for c in row]
                break
        for row in reader:
            if row and any(c.strip() for c in row):
                data = row
                break
    return header, data

def add_csv_to_props(csv_path, out_dict):
    header, data = _read_header_and_first_data(csv_path)

    out_dict["filename"].append(os.path.basename(csv_path))

    if not header or not data:
        # arquivo vazio/sem dados
        for k in dict_keys:
            if k != "filename":
                out_dict[k].append(nan)
        return

    idx_by_norm = {_norm(h): i for i, h in enumerate(header)}
    for norm_name, key in NORM_MAP.items():
        i = idx_by_norm.get(norm_name)
        if i is None or i >= len(data) or data[i].strip() == '':
            out_dict[key].append(nan)
        else:
            try:
                out_dict[key].append(float(data[i]))
            except ValueError:
                out_dict[key].append(nan)

def process_folder(path, pattern="*.csv", out_dict=None):
    """Processa todos os CSVs de 'path' e preenche out_dict."""
    if out_dict is None:
        out_dict = _new_dict_props()
    files = sorted(glob.glob(os.path.join(path, pattern)))
    for fp in files:
        add_csv_to_props(fp, out_dict)
    return out_dict

# =========================
# Nova lógica pedida
# =========================

def _find_prop_dirs(root_path):
    """Encontra todas as pastas 'prop' no padrão:
       N_*/m0_*/dim_*/alpha_a_*_alpha_g_*/prop
    """
    pattern = os.path.join(
        root_path,
        "N_*", "m0_*", "dim_*", "alpha_a_*_alpha_g_*", "prop"
    )
    return sorted(glob.glob(pattern))

def _is_prop_dir(path):
    """Retorna True se 'path' parece ser a pasta 'prop' do padrão."""
    # checa se termina com /prop e se o pai atende ao alpha_a_..._alpha_g_...
    base = os.path.basename(os.path.normpath(path))
    parent = os.path.basename(os.path.dirname(os.path.normpath(path)))
    return (base == "prop") and bool(re.match(r"^alpha_a_.+_alpha_g_.+$", parent))

# === SUBSTITUA sua create_data_folder por esta versão incremental ===
def create_data_folder(path_root_or_prop):
    """
    Se receber o root (ex.: '../../data'), percorre toda a árvore e
    cria/atualiza 'all_data.dat' em cada pasta 'prop' **apenas** se houver CSV novo.
    Se receber diretamente uma pasta 'prop', processa só aquela.
    (Não retorna nada; apenas imprime [SKIP] ou [UPDATE].)
    """
    import os
    import pandas as pd

    path_root_or_prop = os.path.normpath(path_root_or_prop)

    def _process_one_prop(prop_dir: str):
        out_path = os.path.join(prop_dir, "all_data.dat")

        # 1) carrega all_data.dat existente (se houver)
        existing_df = _load_existing_all_data(out_path)

        # 2) descobre CSVs novos
        new_csvs = _list_new_csvs(prop_dir, existing_df)

        if not new_csvs:
            print(f"[SKIP] Sem novos arquivos em: {prop_dir} — nada a fazer.")
            return  # <- não faz nada e não retorna valor (None)

        # 3) processa apenas os novos
        dict_props = _new_dict_props()
        for fp in new_csvs:
            add_csv_to_props(fp, dict_props)

        new_df = pd.DataFrame(dict_props)

        # 4) concatena com existente (se houver) e alinha colunas
        if existing_df is not None:
            for col in set(existing_df.columns) - set(new_df.columns):
                new_df[col] = pd.NA
            for col in set(new_df.columns) - set(existing_df.columns):
                existing_df[col] = pd.NA
            df_out = pd.concat([existing_df, new_df], ignore_index=True)
            if "filename" in df_out.columns:
                df_out = df_out.drop_duplicates(subset=["filename"], keep="first")
        else:
            df_out = new_df

        # 5) salva atualização
        os.makedirs(prop_dir, exist_ok=True)
        df_out.to_csv(out_path, index=False, sep=',', na_rep='NaN')
        print("[UPDATE] Save in:", out_path)
        return  # não retorna nada

    # Caso: usuário passou diretamente uma pasta 'prop'
    if _is_prop_dir(path_root_or_prop):
        _process_one_prop(path_root_or_prop)
        return  # nada a retornar

    # Caso root: percorre todas as 'prop'
    prop_dirs = _find_prop_dirs(path_root_or_prop)
    if not prop_dirs:
        print("Nenhuma pasta 'prop' encontrada sob:", path_root_or_prop)
        return

    for prop_dir in prop_dirs:
        _process_one_prop(prop_dir)
    # sem retorno -> não imprime [] no notebook

# ==== parser dos parâmetros a partir do caminho ====
_param_re = {
    "N":       re.compile(r"(?:^|/)N_(\d+)(?:/|$)"),
    "m0":      re.compile(r"(?:^|/)m0_(\d+)(?:/|$)"),
    "dim":     re.compile(r"(?:^|/)dim_(\d+)(?:/|$)"),
    "alphas":  re.compile(r"(?:^|/)alpha_a_([+-]?\d+(?:\.\d+)?)_alpha_g_([+-]?\d+(?:\.\d+)?)(?:/|$)")
}

def _parse_params_from_prop_dir(prop_dir: str):
    path = os.path.normpath(prop_dir)
    N   = int(_param_re["N"].search(path).group(1))
    m0  = int(_param_re["m0"].search(path).group(1))
    dim = int(_param_re["dim"].search(path).group(1))
    a_match = _param_re["alphas"].search(path)
    alpha_a = float(a_match.group(1))
    alpha_g = float(a_match.group(2))
    return N, m0, dim, alpha_a, alpha_g

# ==== helpers numéricos ====
def _sem(series: pd.Series):
    s = series.dropna()
    n = len(s)
    if n <= 1:
        return np.nan
    return s.std(ddof=1) / np.sqrt(n)

def _combined_error_with_per_sample(values: pd.Series, per_sample_errs: pd.Series):
    """sqrt( SEM^2 + (sum(err_i^2))/n^2 ), com n = número de valores válidos."""
    v = values.dropna()
    n = len(v)
    if n == 0:
        return np.nan
    sem = _sem(v)
    if per_sample_errs is None or per_sample_errs.empty:
        return sem
    e = per_sample_errs.dropna()
    # alinhar índices para evitar confusão:
    e = e.reindex(v.index)
    sum_sigma2 = np.nansum(np.square(e.to_numpy()))
    return np.sqrt((sem ** 2) + (sum_sigma2 / (n ** 2)))

# ==== função principal pedida ====
def build_all_means(root_or_prop_path: str):
    """
    Percorre cada pasta 'prop' que tiver um 'all_data.dat' e monta o dict_all_data
    com médias e erros.

    Retorna (dict_all_data, df_all).
    """
    root_or_prop_path = os.path.normpath(root_or_prop_path)

    # onde procurar:
    if _is_prop_dir(root_or_prop_path):
        prop_dirs = [root_or_prop_path]
    else:
        prop_dirs = _find_prop_dirs(root_or_prop_path)

    dict_all_data = {
        "N":[], "m0":[], "dim":[], "alpha_a":[], "alpha_g":[],
        "N_samples":[], "Short_Cost_mean":[], "Short_Cost_error":[],
        "Short_BFS_mean":[], "Short_BFS_error":[],
        "R_Spearman_mean":[], "R_Spearman_error":[],
        "R_Newman_mean":[],  "R_Newman_error":[],
        "ClusterCoefficient_mean":[], "ClusterCoefficient_error":[]
    }

    for prop_dir in prop_dirs:
        all_data_path = os.path.join(prop_dir, "all_data.dat")
        if not os.path.exists(all_data_path):
            # não há resumo nessa pasta ainda
            continue

        try:
            df = pd.read_csv(all_data_path)
        except Exception:
            # arquivo corrompido ou separador diferente
            try:
                df = pd.read_csv(all_data_path, sep='\t')
            except Exception:
                print(f"[WARN] Não consegui ler {all_data_path}. Pulando.")
                continue

        # extrai parâmetros do caminho
        N, m0, dim, alpha_a, alpha_g = _parse_params_from_prop_dir(prop_dir)

        # N_samples = número de linhas (arquivos) válidas
        n_samples = len(df.index)

        # pega colunas (se não existirem, vira série vazia p/ dar NaN)
        col = lambda name: df[name] if name in df.columns else pd.Series(dtype=float)

        shortest_cost = col("ShortestCost")
        shortest_bfs  = col("ShortestBFS")
        r_spear       = col("R_Spearman")
        r_spear_err   = col("R_Spearman_err")  # erros por amostra
        r_new         = col("R_Newman")
        r_new_err     = col("R_Newman_err")
        cluster       = col("ClusterCoefficient")

        # MÉDIAS (ignorando NaN)
        sc_mean  = shortest_cost.mean(skipna=True)
        sb_mean  = shortest_bfs.mean(skipna=True)
        rs_mean  = r_spear.mean(skipna=True)
        rn_mean  = r_new.mean(skipna=True)
        cc_mean  = cluster.mean(skipna=True)

        # ERROS
        sc_err = _sem(shortest_cost)
        sb_err = _sem(shortest_bfs)
        # combina SEM com erros individuais reportados
        rs_err = _combined_error_with_per_sample(r_spear, r_spear_err)
        rn_err = _combined_error_with_per_sample(r_new,   r_new_err)
        cc_err = _sem(cluster)

        # preenche dicionário
        dict_all_data["N"].append(N)
        dict_all_data["m0"].append(m0)
        dict_all_data["dim"].append(dim)
        dict_all_data["alpha_a"].append(alpha_a)
        dict_all_data["alpha_g"].append(alpha_g)

        dict_all_data["N_samples"].append(int(n_samples))

        dict_all_data["Short_Cost_mean"].append(sc_mean)
        dict_all_data["Short_Cost_error"].append(sc_err)

        dict_all_data["Short_BFS_mean"].append(sb_mean)
        dict_all_data["Short_BFS_error"].append(sb_err)

        dict_all_data["R_Spearman_mean"].append(rs_mean)
        dict_all_data["R_Spearman_error"].append(rs_err)

        dict_all_data["R_Newman_mean"].append(rn_mean)
        dict_all_data["R_Newman_error"].append(rn_err)

        dict_all_data["ClusterCoefficient_mean"].append(cc_mean)
        dict_all_data["ClusterCoefficient_error"].append(cc_err)

    df_all = pd.DataFrame(dict_all_data)
    df_all.to_csv("../../data/all_data.csv", sep=',', index=False)

def processing_all_data(root):
    create_data_folder(root)   # percorre toda a árvore e cria um all_data.dat em cada 'prop/'
    build_all_means(root)