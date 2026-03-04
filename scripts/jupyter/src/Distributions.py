import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


# create folder to results
def make_results_folders():
    path = "../../results"
    # If file in all_files, create check_folder to move files
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + "/alpha_a")
        os.makedirs(path + "/alpha_g")
        os.makedirs(path + "/N")
        os.makedirs(path + "/distributions")
        os.makedirs(path + "/network")
        os.makedirs(path + "/parameters")
    else:
        pass

# Função para apagar todos os arquivos dentro de pastas gml, mantendo a estrutura de pastas
def delete_files_in_gml_folders(folder_path):
    for root, dirs, files in os.walk(folder_path):
        # Verifica se a pasta atual é uma pasta "gml"
        if os.path.basename(root) == 'gml':
            print(f"Removendo arquivos dentro da pasta: {root}")
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)  # Apaga o arquivo
                print(f"Arquivo removido: {file_path}")
    print("Todos os arquivos dentro das pastas 'gml' foram removidos.")


# ============================================================
# I/O
# ============================================================

def read_parquet(N, dim, m, alpha_a, alpha_g):
    folder = Path(
        f"../../data/N_{N}/m0_{m}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}"
    )

    files = list(folder.glob(
        f"N{N}_d{dim}_m{m}_G{alpha_g}_A{alpha_a}_seed001to*_nodes.parquet"
    ))

    df = pq.read_table(files).to_pandas()
    return df


def pooled_sigma_pk(pk, Ntot, eps=1e-15):
    """
    sigma_{P(k)} ~ sqrt( P(k)*(1-P(k)) / Ntot )
    """
    pk = np.asarray(pk, float)
    pk = np.clip(pk, eps, 1.0)
    return np.sqrt(pk * (1.0 - pk) / float(Ntot))

def pooled_sigma_lnq(pk, q, Ntot, eps=1e-15):
    """
    sigma_{ln_q} ≈ pk^{-q} * sigma_pk
    """
    pk = np.asarray(pk, float)
    pk = np.clip(pk, eps, 1.0)
    sig_pk = pooled_sigma_pk(pk, Ntot, eps=eps)
    return sig_pk / (pk ** float(q))


def chi2_red_lnq(k, pk, best, Ntot, k_sat=None, eps=1e-12):
    """
    chi^2_red para pooled no espaço y = ln_q(pk).
    """
    k_fit, pk_fit = normalize_pk_on_support(k, pk, k_sat=k_sat)

    A, eta, q = best["A"], best["eta"], best["q"]

    y_obs = ln_q(pk_fit, q, eps=eps)
    y_pred = ln_q(pk_model(k_fit, A, eta, q), q, eps=eps)

    sig_y = pooled_sigma_lnq(pk_fit, q, Ntot, eps=eps)
    sig_y = np.maximum(sig_y, eps)

    resid = (y_obs - y_pred) / sig_y
    chi2 = np.sum(resid**2)

    n = len(k_fit)
    p = 2  # aqui você está ajustando eta e q; A é determinado por normalização
    dof = max(1, n - p)

    return chi2 / dof
def rmse_norm_lnq(k, pk, best, Ntot, k_sat=None, eps=1e-12):
    """
    RMSE normalizado pelo erro esperado (sigma_lnq).
    """
    k_fit, pk_fit = normalize_pk_on_support(k, pk, k_sat=k_sat)

    A, eta, q = best["A"], best["eta"], best["q"]
    y_obs = ln_q(pk_fit, q, eps=eps)
    y_pred = ln_q(pk_model(k_fit, A, eta, q), q, eps=eps)

    sig_y = pooled_sigma_lnq(pk_fit, q, Ntot, eps=eps)
    sig_y = np.maximum(sig_y, eps)

    z = (y_obs - y_pred) / sig_y
    return float(np.sqrt(np.mean(z**2)))
# ============================================================
# Degree distributions
# ============================================================
def pooled_pk_from_nodes(df_nodes: pd.DataFrame) -> pd.DataFrame:
    counts = df_nodes["deg"].value_counts().sort_index()
    Pk = counts / counts.sum()
    return pd.DataFrame({"k": counts.index.astype(int), "count": counts.values, "Pk": Pk.values})

def degree_distribution_pooled(df_nodes: pd.DataFrame) -> pd.DataFrame:
    counts = df_nodes["deg"].value_counts().sort_index()
    Pk = counts / counts.sum()
    return pd.DataFrame(
        {"k": counts.index.astype(int), "count": counts.values, "Pk": Pk.values}
    )


def degree_distribution_mean_over_seeds(df_nodes: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        df_nodes.groupby("seed")["deg"]
        .value_counts(normalize=True)
        .rename("Pk")
        .reset_index()
        .rename(columns={"deg": "k"})
    )

    agg = per_seed.groupby("k")["Pk"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.rename(columns={"mean": "Pk_mean", "std": "Pk_std", "count": "n_seeds"})
    return agg.sort_values("k")

def pooled_pk_bootstrap_by_seed(df_nodes: pd.DataFrame, n_boot=2000, rng=123):
    """
    Reamostra seeds com reposição, junta todos os nós das seeds sorteadas
    e calcula P(k). Retorna:
      - df_pooled (central)
      - df_err: k, Pk_std, Pk_sem (bootstrap)
    """
    gen = np.random.default_rng(rng)

    seeds = df_nodes["seed"].unique()
    nS = len(seeds)

    # central (pooled real)
    df_c = pooled_pk_from_nodes(df_nodes)
    ks = df_c["k"].to_numpy()

    # para acelerar, separa por seed
    by_seed = {s: df_nodes.loc[df_nodes["seed"] == s, "deg"].to_numpy() for s in seeds}

    boot_pk = np.zeros((n_boot, len(ks)), float)

    # mapeia k -> idx
    k_to_idx = {k: i for i, k in enumerate(ks)}

    for b in range(n_boot):
        draw = gen.choice(seeds, size=nS, replace=True)

        # agrega contagens
        counts = {}
        total = 0
        for s in draw:
            degs = by_seed[s]
            total += degs.size
            # conta graus nessa seed
            uniq, cts = np.unique(degs, return_counts=True)
            for kk, cc in zip(uniq, cts):
                counts[kk] = counts.get(kk, 0) + int(cc)

        # converte em Pk no mesmo suporte ks (zeros onde não apareceu)
        for kk, cc in counts.items():
            if kk in k_to_idx:
                boot_pk[b, k_to_idx[kk]] = cc / total

    Pk_std = boot_pk.std(axis=0, ddof=1)
    Pk_sem = Pk_std  # bootstrap std já é a dispersão entre réplicas bootstrap

    df_err = pd.DataFrame({"k": ks.astype(int), "Pk_boot_std": Pk_std, "Pk_boot_sem": Pk_sem})
    return df_c, df_err

def lnq_bootstrap_errors_from_boots(boot_pk, q, eps=1e-12):
    """
    Dado boot_pk (n_boot x n_k), retorna sigma_y em y=ln_q(Pk).
    """
    Y = ln_q(np.maximum(boot_pk, eps), q, eps=eps)
    return Y.std(axis=0, ddof=1)
# ============================================================
# k_sat (cutoff)
# ============================================================

def find_k_saturation_pooled(df_pooled: pd.DataFrame,
                             min_count: int = 10,
                             p_min: float | None = None) -> int:
    df = df_pooled.copy().sort_values("k")

    mask = df["count"].to_numpy() >= int(min_count)
    if p_min is not None:
        mask &= df["Pk"].to_numpy() >= float(p_min)

    if not np.any(mask):
        return int(df["k"].min())

    return int(df.loc[mask, "k"].max())


def find_k_saturation_mean_over_seeds(df_mean: pd.DataFrame,
                                      min_seeds: int = 5,
                                      p_min: float | None = None,
                                      snr_min: float | None = None) -> int:
    df = df_mean.copy().sort_values("k")

    mask = df["n_seeds"].to_numpy() >= int(min_seeds)

    if p_min is not None:
        mask &= df["Pk_mean"].to_numpy() >= float(p_min)

    if snr_min is not None:
        sem = df["sem"].to_numpy()
        snr = np.where(sem > 0, df["Pk_mean"].to_numpy() / sem, 0.0)
        mask &= snr >= float(snr_min)

    if not np.any(mask):
        return int(df["k"].min())

    return int(df.loc[mask, "k"].max())


# ============================================================
# q-exponential / q-log + model normalization
# ============================================================

def exp_q(x, q):
    x = np.asarray(x, float)
    q = float(q)

    if np.isclose(q, 1.0):
        return np.exp(x)

    base = 1.0 + (1.0 - q) * x
    return np.where(base > 0.0, base ** (1.0 / (1.0 - q)), 0.0)


def ln_q(x, q, eps=1e-12):
    x = np.asarray(x, float)
    q = float(q)
    x = np.maximum(x, eps)

    if np.isclose(q, 1.0):
        return np.log(x)

    return (x ** (1.0 - q) - 1.0) / (1.0 - q)


def compute_A_from_support(k_support, eta, q):
    k_support = np.asarray(k_support, float)
    eta = float(eta)
    q = float(q)

    w = exp_q(-k_support / eta, q)
    Z = np.sum(w)
    return (1.0 / Z) if Z > 0 else np.nan


def normalize_pk_on_support(k, pk, k_sat=None):
    k = np.asarray(k, float)
    pk = np.asarray(pk, float)

    if k_sat is not None:
        mask = k <= float(k_sat)
        k = k[mask]
        pk = pk[mask]

    s = pk.sum()
    if s <= 0:
        raise ValueError("pk soma <= 0 após truncamento/filtragem.")
    pk = pk / s

    return k, pk


def pk_model(k, A, eta, q):
    k = np.asarray(k, float)
    return float(A) * exp_q(-k / float(eta), float(q))


# ============================================================
# Fit (ln_q space)
# ============================================================

def evaluate_lnq_fit(k, pk_obs, A, eta, q, eps=1e-12):
    k = np.asarray(k, float)
    pk_obs = np.asarray(pk_obs, float)

    pk_pred = pk_model(k, A, eta, q)

    y_obs = ln_q(pk_obs, q, eps=eps)
    y_pred = ln_q(pk_pred, q, eps=eps)

    resid = y_obs - y_pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)

    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    RMSE = np.sqrt(np.mean(resid**2)) if len(resid) > 0 else np.nan

    return {
        "A": A, "eta": eta, "q": q,
        "R2": R2,
        "RMSE": RMSE,
        "y_obs": y_obs,
        "y_pred": y_pred,
        "pk_pred": pk_pred,
    }


def detect_best_parms(k, pk, k_sat=None, eta_grid=None, q_grid=None, eps=1e-12):
    k_fit, pk_fit = normalize_pk_on_support(k, pk, k_sat=k_sat)

    if eta_grid is None:
        eta_grid = np.linspace(0.1, 0.5, 80)
    if q_grid is None:
        q_grid = np.linspace(1.05, 1.8, 40)

    best = {"R2": -np.inf}

    for eta in eta_grid:
        for q in q_grid:
            A = compute_A_from_support(k_fit, eta, q)
            if not np.isfinite(A):
                continue

            out = evaluate_lnq_fit(k_fit, pk_fit, A, eta, q, eps=eps)
            if np.isfinite(out["R2"]) and out["R2"] > best["R2"]:
                best = {
                    "A": out["A"],
                    "eta": out["eta"],
                    "q": out["q"],
                    "R2": out["R2"],
                    "RMSE": out["RMSE"],
                }

    return best


def fit_from_pooled_df(df_pooled, k_sat=None, **kwargs):
    k = df_pooled["k"].to_numpy()
    pk = df_pooled["Pk"].to_numpy()
    return detect_best_parms(k, pk, k_sat=k_sat, **kwargs)


def fit_from_mean_df(df_mean, k_sat=None, **kwargs):
    k = df_mean["k"].to_numpy()
    pk = df_mean["Pk_mean"].to_numpy()
    return detect_best_parms(k, pk, k_sat=k_sat, **kwargs)


# ============================================================
# Plot
# ============================================================

def plot_lnq_pk_with_best_fit(k, pk, best, k_sat=None, eps=1e-12,
                             ms=6, lw=2.0, fs=16):
    k_fit, pk_fit = normalize_pk_on_support(k, pk, k_sat=k_sat)

    A = best["A"]
    eta = best["eta"]
    q = best["q"]

    y_obs = ln_q(pk_fit, q, eps=eps)
    pk_pred = pk_model(k_fit, A, eta, q)
    y_pred = ln_q(pk_pred, q, eps=eps)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(k_fit, y_obs, "o", ms=ms, mfc="none", label=r"Data: $\ln_q(p(k))$")
    ax.plot(k_fit, y_pred, "-", lw=lw, label=r"Fit: $\ln_q(A\,e_q(-k/\eta))$")

    title = rf"$q={q:.4f},\ \eta={eta:.4f},\ A={A:.4g}$"
    if "R2" in best:
        title += rf",  $R^2={best['R2']:.4f}$"
    if "RMSE" in best:
        title += rf",  RMSE={best['RMSE']:.4g}"

    ax.set_title(title, fontsize=fs)
    ax.set_xlabel(r"$k$", fontsize=fs)
    ax.set_ylabel(r"$\ln_q(p(k))$", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.legend(fontsize=fs-2, frameon=True)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig, ax