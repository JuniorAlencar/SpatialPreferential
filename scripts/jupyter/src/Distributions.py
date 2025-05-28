import numpy as np
import pandas as pd
import os
import glob
import gzip
import json
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.stats import kstest


# N: number of nodes
# d: dimension
# alpha_a:.2f: parameter for preferential attrachment
# alpha_g:.2f: parameter for probability of distances nodes

def process_distance_file(index_file_tuple):
    """Processa um único arquivo para extrair as distâncias."""
    i, file = index_file_tuple
    n_lines = 10**5  # Número fixo de linhas por arquivo
    distances = np.zeros(n_lines, dtype=np.float32)

    try:
        with gzip.open(file, 'rt') as gzip_file:
            count = 0
            for line in gzip_file:
                if line.startswith("distance "):
                    distances[count] = float(line[9:])
                    count += 1
                    if count == n_lines:
                        break
    except EOFError:
        print(f"Arquivo {file} está corrompido ou incompleto. Apagando arquivo.")
        os.remove(file)
        return None

    return i, distances, os.path.basename(file)


def distance_file(N):
    """Processa todos os arquivos dentro de N, independentemente de dim, alpha_a e alpha_g."""
    base_folder = f"../../data/N_{N}"

    if not os.path.exists(base_folder):
        print(f"Pasta {base_folder} não existe.")
        return

    # Buscar todos os diretórios com gml
    gml_dirs = glob.glob(os.path.join(base_folder, "dim_*", "alpha_a_*_alpha_g_*", "gml"))

    if not gml_dirs:
        print(f"Nenhum diretório com arquivos .gml encontrado em {base_folder}")
        return

    print(f"Encontrado {len(gml_dirs)} diretórios com arquivos para processar.")

    for gml_dir in gml_dirs:
        output_dir = os.path.dirname(gml_dir)

        all_files = sorted(glob.glob(os.path.join(gml_dir, "*.gml.gz")))

        if not all_files:
            print(f"Nenhum arquivo .gml.gz encontrado em {gml_dir}")
            continue

        n_files = len(all_files)
        n_lines = 10**5

        distances_npy_path = os.path.join(output_dir, "distances.npy")
        filenames_csv_path = os.path.join(output_dir, "filenames_distances.csv")

        if os.path.exists(filenames_csv_path):
            df_existing = pd.read_csv(filenames_csv_path)
            processed_files = set(df_existing["filenames"].tolist())
        else:
            df_existing = pd.DataFrame(columns=["filenames"])
            processed_files = set()

        new_files = [file for file in all_files if os.path.basename(file) not in processed_files]

        if not new_files:
            print(f"Todos os arquivos já foram processados em {gml_dir}.")
            continue

        print(f"Processando {len(new_files)} novos arquivos em {gml_dir}...")

        with Pool() as pool:
            results = pool.map(process_distance_file, enumerate(new_files))

        results = [res for res in results if res is not None]

        if not results:
            print(f"Nenhum dado válido processado em {gml_dir}.")
            continue

        new_filenames = []
        new_data = np.empty((n_lines, len(results)), dtype=np.float32)

        for i, (idx, distances, filename) in enumerate(results):
            new_data[:, i] = distances
            new_filenames.append(filename)

        if os.path.exists(distances_npy_path):
            existing_data = np.load(distances_npy_path)
            updated_data = np.hstack((existing_data, new_data))
        else:
            updated_data = new_data

        np.save(distances_npy_path, updated_data)

        df_new = pd.DataFrame({"filenames": new_filenames})
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        df_updated.to_csv(filenames_csv_path, index=False)

        print(f"Processamento concluído em {output_dir}. Dados salvos.")

    print(f"\nProcessamento total concluído para N = {N}.")




def process_file(index_file_tuple):
    """Processa um único arquivo para extrair os graus dos nós."""
    i, file = index_file_tuple
    n_lines = 10**5
    degrees = np.zeros(n_lines, dtype=np.int32)

    try:
        with gzip.open(file, 'rt') as gzip_file:
            count = 0
            for line in gzip_file:
                if line.startswith("degree "):
                    degrees[count] = int(line[7:])
                    count += 1
                    if count == n_lines:
                        break
    except EOFError:
        print(f"Arquivo {file} está corrompido ou incompleto. Apagando arquivo.")
        os.remove(file)
        return None

    return i, degrees, os.path.basename(file)


def degree_file(N):
    """Processa todos os arquivos dentro de N, extraindo os graus dos nós."""
    base_folder = f"../../data/N_{N}"

    if not os.path.exists(base_folder):
        print(f"Pasta {base_folder} não existe.")
        return

    # Buscar todos os diretórios com gml
    gml_dirs = glob.glob(os.path.join(base_folder, "dim_*", "alpha_a_*_alpha_g_*", "gml"))

    if not gml_dirs:
        print(f"Nenhum diretório com arquivos .gml encontrado em {base_folder}")
        return

    print(f"Encontrado {len(gml_dirs)} diretórios com arquivos para processar.")

    for gml_dir in gml_dirs:
        output_dir = os.path.dirname(gml_dir)

        all_files = sorted(glob.glob(os.path.join(gml_dir, "*.gml.gz")))

        if not all_files:
            print(f"Nenhum arquivo .gml.gz encontrado em {gml_dir}")
            continue

        n_files = len(all_files)
        n_lines = 10**5

        degree_npy_path = os.path.join(output_dir, "degree.npy")
        filenames_csv_path = os.path.join(output_dir, "filenames_degree.csv")

        if os.path.exists(filenames_csv_path):
            df_existing = pd.read_csv(filenames_csv_path)
            processed_files = set(df_existing["filenames"].tolist())
        else:
            df_existing = pd.DataFrame(columns=["filenames"])
            processed_files = set()

        new_files = [file for file in all_files if os.path.basename(file) not in processed_files]

        if not new_files:
            print(f"Todos os arquivos já foram processados em {gml_dir}.")
            continue

        print(f"Processando {len(new_files)} novos arquivos em {gml_dir}...")

        with Pool() as pool:
            results = pool.map(process_file, enumerate(new_files))

        results = [res for res in results if res is not None]

        if not results:
            print(f"Nenhum dado válido processado em {gml_dir}.")
            continue

        new_filenames = []
        new_data = np.empty((n_lines, len(results)), dtype=np.int32)

        for i, (idx, degrees, filename) in enumerate(results):
            new_data[:, i] = degrees
            new_filenames.append(filename)

        if os.path.exists(degree_npy_path):
            existing_data = np.load(degree_npy_path)
            updated_data = np.hstack((existing_data, new_data))
        else:
            updated_data = new_data

        np.save(degree_npy_path, updated_data)

        df_new = pd.DataFrame({"filenames": new_filenames})
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        df_updated.to_csv(filenames_csv_path, index=False)

        print(f"Processamento concluído em {output_dir}. Dados salvos.")

    print(f"\nProcessamento total concluído para N = {N}.")



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

def distribution(degree, save=False, **kwargs):
    """
    Calcula a distribuição de graus P(k) e remove valores nulos.
    
    Args:
        degree (array-like): Lista de graus.
        save (bool): Se True, salva os dados.
        **kwargs: Argumentos adicionais (N, dim, alpha_a, alpha_g) para salvar os dados.

    Returns:
        k_real (array): Valores de k após remoção de bins vazios.
        p_real (array): Distribuição normalizada de P(k).
    """
    
    hist, bins_edge = np.histogram(degree, bins=np.arange(0.5, 10**4 + 1.5, 1), density=True)
    P = hist * np.diff(bins_edge)  # P(k) = densidade * delta k
    K = bins_edge[:-1]  # Usamos o início dos bins como k representativo
    
    # Remover bins vazios (onde P(k) == 0)
    valid = P > 0
    k_real = K[valid]
    p_real = P[valid]

    # Normalização para garantir que ∑P(k) = 1
    P_sum = np.sum(p_real)
    if P_sum > 0:
        p_real /= P_sum

    # Remover NaNs ou infinitos
    valid_values = np.isfinite(k_real) & np.isfinite(p_real)
    k_real = k_real[valid_values]
    p_real = p_real[valid_values]

    # Se save=True, salvar os dados
    if save:
        required_keys = ["N", "dim", "alpha_a", "alpha_g"]
        if not all(arg in kwargs for arg in required_keys):
            raise ValueError(f"Se save=True, os argumentos {required_keys} devem ser fornecidos.")

        # Obtém os valores das variáveis
        N, dim, alpha_a, alpha_g = kwargs["N"], kwargs["dim"], kwargs["alpha_a"], kwargs["alpha_g"]

        # Define o caminho do arquivo
        save_path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/degree_distribution_linear.csv"

        # Salva os dados
        df = pd.DataFrame(data={"k": k_real, "pk": p_real})
        df.to_csv(save_path, sep=' ', index=False)
        print(f"Arquivo salvo em {save_path}")

    return k_real, p_real

# distribution: distribution (N_k/N)
# return(p_cum): cumulative distribution

def cumulative_distribution(distribution):
    p_cum = np.zeros(len(distribution))
    p_cum[0] = sum(distribution)
    for i in range(1,len(distribution)):
        p_cum[i] = p_cum[i-1] - distribution[i]
    return p_cum

def log_binning(counter_dict, bin_count, save=False, **kwargs):
    """Binagem logarítmica rápida."""

    keys = np.array(list(counter_dict.keys()), dtype=np.float64)
    values = np.array(list(counter_dict.values()), dtype=np.float64)

    # Filtrar k > 0
    mask = keys > 0
    keys = keys[mask]
    values = values[mask]

    min_x = np.log10(np.min(keys))
    max_x = np.log10(np.max(keys))
    bins = np.logspace(min_x, max_x, num=bin_count)

    # Histogramas
    hist_counts, _ = np.histogram(keys, bins=bins, weights=values)
    bin_counts, _ = np.histogram(keys, bins=bins)
    sum_k, _ = np.histogram(keys, bins=bins, weights=keys)

    with np.errstate(divide='ignore', invalid='ignore'):
        Pk = hist_counts / bin_counts
        k = sum_k / bin_counts

    # Remover bins inválidos
    valid = (bin_counts > 0) & (k > 0) & np.isfinite(k) & np.isfinite(Pk)
    k = k[valid]
    Pk = Pk[valid]

    # Normalizar P(k)
    Pk_sum = np.sum(Pk)
    if Pk_sum > 0:
        Pk /= Pk_sum

    # Salvar
    if save:
        required_keys = ["N", "dim", "alpha_a", "alpha_g", "propertie"]
        if not all(arg in kwargs for arg in required_keys):
            raise ValueError(f"Se save=True, precisa de {required_keys}.")

        N = kwargs["N"]
        dim = kwargs["dim"]
        alpha_a = kwargs["alpha_a"]
        alpha_g = kwargs["alpha_g"]
        propertie = kwargs["propertie"]

        save_path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/{propertie}_distribution_log.csv"
        columns = {"degree": ["k", "pk"], "distances": ["ds", "pds"]}
        df = pd.DataFrame({columns[propertie][0]: k, columns[propertie][1]: Pk})
        df.to_csv(save_path, sep=' ', index=False)
        print(f"Arquivo salvo em {save_path}")

    return k, Pk

def log_binning_distances(distances: np.ndarray[float], n_bins: int ,save: bool, **kwargs):
    """Binagem logarítmica rápida.
        distances[float]: list of all pair distances between nodes
        n_bins[int]: number of bins in log-bing
        save[bool]: if True, save in files, else, return ds, pds
        kwargs[list[str]]: if save True, kwargs = ["N", "dim", "alpha_a", "alpha_g"]
    """
    # filtering of distances (ds > 10⁻⁶)
    data_filt = distances[distances > 1e-6]
    
    bins = np.logspace(np.log10(min(data_filt)), np.log10(max(data_filt)), n_bins)
    pdf, bin_edges = np.histogram(data_filt, bins = bins, density = True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widhs = bin_edges[1:] - bin_edges[:-1]
    pdf = np.array(pdf) / sum(pdf)

    # Salvar
    if save:
        required_keys = ["N", "dim", "alpha_a", "alpha_g"]
        if not all(arg in kwargs for arg in required_keys):
            raise ValueError(f"Se save=True, precisa de {required_keys}.")

        N = kwargs["N"]
        dim = kwargs["dim"]
        alpha_a = kwargs["alpha_a"]
        alpha_g = kwargs["alpha_g"]

        save_path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/distances_distribution_log.csv"
        columns = {"distances": ["ds", "pds"]}
        df = pd.DataFrame({"ds": bin_widhs, "pds": pdf})
        df.to_csv(save_path, sep=' ', index=False)
        print(f"Arquivo salvo em {save_path}")

    return bin_widhs, pdf

def q(alpha_a, d):
    ration = alpha_a/d
    if(0<=ration<=1):
        return 4/3
    else:
        return (1/3)*np.exp(1-ration)+1

def eta(alpha_a, d):
    ration = alpha_a/d
    if(0<=ration<=1):
        return 0.3
    else:
        return -1.15*np.exp(1-ration) + 1.45

# Funções do modelo
def normalized_constant(x, q, b):
    distribution_list = (1 - (1 - q) * (x / b)) ** (1 / (1 - q))
    return 1 / np.sum(distribution_list)

def q_exp(x, q, b):
    A = normalized_constant(x, q, b)
    return A * (1 - (1 - q) * (x / b)) ** (1 / (1 - q))

def optimize_q_exp(k, pk, q_initial=1.333, b_initial=0.40, delta_q=0.01, delta_b=0.01):
    """
    Ajusta q e b inicialmente, depois fixa q e refina o ajuste de b para melhorar a precisão de b.

    Retorna:
        fitted_q (float): Melhor valor ajustado de q.
        refined_b (float): Valor ajustado de b com q fixo.
        perr_q (float): Erro padrão de q.
        perr_b (float): Erro padrão de b (refinado).
    """
    # --- Filtragem inicial ---
    k = np.array(k)
    pk = np.array(pk)
    mask = pk >= 1e-6
    k_filtered = k[mask]
    pk_filtered = pk[mask]

    if len(k_filtered) == 0:
        raise ValueError("Nenhum dado satisfaz a condição pk >= 1e-6.")

    # --- Normalização segura ---
    def normalized_constant_safe(q, b, k_vals):
        try:
            integral, _ = quad(lambda x: (1 - (1 - q) * (x / b)) ** (1 / (1 - q)),
                               np.min(k_vals), np.max(k_vals))
            return 1 / integral if integral > 0 else 1
        except:
            return 1

    # --- Etapa 1: ajustar q e b livremente ---
    def q_exp_full(x, q, b):
        A = normalized_constant_safe(q, b, k_filtered)
        inner = 1 - (1 - q) * (x / b)
        result = np.zeros_like(x)
        valid = inner > 0
        result[valid] = A * inner[valid] ** (1 / (1 - q))
        return result

    bounds_qb = (
        [q_initial - delta_q, b_initial - delta_b],
        [q_initial + delta_q, b_initial + delta_b]
    )

    popt_qb, pcov_qb = curve_fit(
        q_exp_full, k_filtered, pk_filtered,
        p0=[q_initial, b_initial], bounds=bounds_qb
    )

    fitted_q, prelim_b = popt_qb
    perr_q = np.sqrt(pcov_qb[0, 0])

    # --- Etapa 2: fixar q e refinar b ---
    def q_exp_fixed_q(x, b):
        A = normalized_constant_safe(fitted_q, b, k_filtered)
        inner = 1 - (1 - fitted_q) * (x / b)
        result = np.zeros_like(x)
        valid = inner > 0
        result[valid] = A * inner[valid] ** (1 / (1 - fitted_q))
        return result

    bounds_b = (b_initial - delta_b, b_initial + delta_b)

    popt_b, pcov_b = curve_fit(
        q_exp_fixed_q, k_filtered, pk_filtered,
        p0=[prelim_b], bounds=bounds_b
    )

    refined_b = popt_b[0]
    perr_b = np.sqrt(pcov_b[0, 0])

    return fitted_q, refined_b, perr_q, perr_b

def ln_q(k, pk, q, eta):
    k_values = np.zeros(len(k))
    for i in range(len(k)):
        k_values[i] = (1+(q-1)*(k/eta))**(1/(1-q))
    P0 = sum(k_values)
    return ((pk/P0)**(1-q)-1)/(1-q)

import os
import glob
import gzip
import numpy as np
import pandas as pd
from multiprocessing import Pool


def process_distance_file(index_file_tuple):
    """Processa um único arquivo para extrair as distâncias."""
    i, file = index_file_tuple
    n_lines = 10**5  # Número fixo de linhas por arquivo
    distances = np.zeros(n_lines, dtype=np.float32)

    try:
        with gzip.open(file, 'rt') as gzip_file:
            count = 0
            for line in gzip_file:
                if line.startswith("distance "):
                    distances[count] = float(line[9:])
                    count += 1
                    if count == n_lines:
                        break
    except EOFError:
        print(f"Arquivo {file} está corrompido ou incompleto. Apagando arquivo.")
        os.remove(file)
        return None

    return i, distances, os.path.basename(file)


def distance_file(N):
    """Processa todos os arquivos dentro de N, independentemente de dim, alpha_a e alpha_g."""
    base_folder = f"../../data/N_{N}"

    if not os.path.exists(base_folder):
        print(f"Pasta {base_folder} não existe.")
        return

    # Buscar todos os diretórios com gml
    gml_dirs = glob.glob(os.path.join(base_folder, "dim_*", "alpha_a_*_alpha_g_*", "gml"))

    if not gml_dirs:
        print(f"Nenhum diretório com arquivos .gml encontrado em {base_folder}")
        return

    print(f"Encontrado {len(gml_dirs)} diretórios com arquivos para processar.")

    for gml_dir in gml_dirs:
        output_dir = os.path.dirname(gml_dir)

        all_files = sorted(glob.glob(os.path.join(gml_dir, "*.gml.gz")))

        if not all_files:
            print(f"Nenhum arquivo .gml.gz encontrado em {gml_dir}")
            continue

        n_files = len(all_files)
        n_lines = 10**5

        distances_npy_path = os.path.join(output_dir, "distances.npy")
        filenames_csv_path = os.path.join(output_dir, "filenames_distances.csv")

        if os.path.exists(filenames_csv_path):
            df_existing = pd.read_csv(filenames_csv_path)
            processed_files = set(df_existing["filenames"].tolist())
        else:
            df_existing = pd.DataFrame(columns=["filenames"])
            processed_files = set()

        new_files = [file for file in all_files if os.path.basename(file) not in processed_files]

        if not new_files:
            print(f"Todos os arquivos já foram processados em {gml_dir}.")
            continue

        print(f"Processando {len(new_files)} novos arquivos em {gml_dir}...")

        with Pool() as pool:
            results = pool.map(process_distance_file, enumerate(new_files))

        results = [res for res in results if res is not None]

        if not results:
            print(f"Nenhum dado válido processado em {gml_dir}.")
            continue

        new_filenames = []
        new_data = np.empty((n_lines, len(results)), dtype=np.float32)

        for i, (idx, distances, filename) in enumerate(results):
            new_data[:, i] = distances
            new_filenames.append(filename)

        if os.path.exists(distances_npy_path):
            existing_data = np.load(distances_npy_path)
            updated_data = np.hstack((existing_data, new_data))
        else:
            updated_data = new_data

        np.save(distances_npy_path, updated_data)

        df_new = pd.DataFrame({"filenames": new_filenames})
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        df_updated.to_csv(filenames_csv_path, index=False)

        print(f"Processamento concluído em {output_dir}. Dados salvos.")

    print(f"\nProcessamento total concluído para N = {N}.")


# Exemplo de execução:
# distance_file(N=5000)


def combine_estimates_from_datasets(k_list, pk_list, dim, alpha_a=2.0,
                                     q_initial=1.333, b_initial=0.40,
                                     delta_q=0.01, delta_b=0.01):
    """
    Aplica optimize_q_exp a cada conjunto (k, pk) com filtragem e retorna os valores combinados de q e b com erro propagado.

    Parâmetros:
        k_list (list of lists): Lista de listas com valores de k (um por alpha_g).
        pk_list (list of lists): Lista de listas com valores de pk correspondentes.
        dim (int): Dimensão espacial do sistema.
        alpha_a (float): Valor de alpha_a.
        q_initial (float): Valor inicial para o ajuste de q.
        b_initial (float): Valor inicial para o ajuste de b.
        delta_q (float): Delta para busca em q.
        delta_b (float): Delta para busca em b.

    Retorna:
        mean_q, error_q, mean_b, error_b
    """
    qs, bs = [], []
    err_qs, err_bs = [], []

    # Inicialização baseada na teoria
    q_initial = q(alpha_a, dim)
    b_initial = eta(alpha_a, dim)
    
    for k, pk in zip(k_list, pk_list):
        # --- Filtragem de pk < 1e-6 mantendo correspondência com k ---
        filtered_data = [(ki, pki) for ki, pki in zip(k, pk) if pki >= 1e-6]
        if not filtered_data:
            continue
        k_filtered, pk_filtered = zip(*filtered_data)

        try:
            fitted_q, fitted_b, perr_q, perr_b = optimize_q_exp(
                np.array(k_filtered), np.array(pk_filtered),
                q_initial=q_initial,
                b_initial=b_initial,
                delta_q=delta_q,
                delta_b=delta_b
            )
            qs.append(fitted_q)
            bs.append(fitted_b)
            err_qs.append(perr_q)
            err_bs.append(perr_b)
        except Exception as e:
            print(f"Ajuste falhou para um conjunto: {e}")
            continue

    # Função auxiliar para média ponderada
    def combine(values, errors):
        values = np.array(values)
        errors = np.array(errors)
        weights = 1.0 / errors**2
        mean = np.sum(weights * values) / np.sum(weights)
        error = 1.0 / np.sqrt(np.sum(weights))
        return mean, error

    mean_q, error_q = combine(qs, err_qs)
    mean_b, error_b = combine(bs, err_bs)

    return mean_q, error_q, mean_b, error_b


def bootstrap_q_exp(k, pk, dim, alpha_a=2.0, n_bootstrap=1000):
    """
    Realiza um bootstrap para estimar os melhores valores de q e b.
    
    Parâmetros:
        k (array): Valores de k.
        pk (array): Valores de P(k).
        n_bootstrap (int): Número de reamostragens para o bootstrap.
    
    Retorna:
        mean_q (float): Média dos valores ajustados de q.
        std_q (float): Desvio padrão de q.
        mean_b (float): Média dos valores ajustados de b.
        std_b (float): Desvio padrão de b.
    """
    
    # Filtering data for pk <= 10⁻⁶, return k_filtering, pk_filtering
    k_filtered = []
    pk_filtered = []
    threshold = 1e-6  # Definição do limite mínimo
    
    for k_sublist, pk_sublist in zip(k, pk):
        # Filtra os valores mantendo a relação 1:1
        filtered_pairs = [(ki, pki) for ki, pki in zip(k_sublist, pk_sublist) if pki >= threshold]
        
        # Separa novamente em listas individuais
        if filtered_pairs:
            k_filtered.append([ki for ki, pki in filtered_pairs])
            pk_filtered.append([pki for ki, pki in filtered_pairs])
        else:
            k_filtered.append([])  # Mantém a estrutura original
            pk_filtered.append([])
    
    k_flat = [item for sublist in k_filtered for item in sublist]
    pk_flat = [item for sublist in pk_filtered for item in sublist]

    q_values = []
    b_values = []
    Q_init = q(alpha_a, dim)
    B_init = eta(alpha_a, dim)
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(k_flat), size=len(k_flat), replace=True)
        k_sample = np.array(k_flat)[indices]
        pk_sample = np.array(pk_flat)[indices]
        
        try:
            q_fit, b_fit, err_q, err_b = optimize_q_exp(k_sample, pk_sample, q_initial=Q_init, b_initial=B_init, delta_q=0.01, delta_b=0.01)
            q_values.append(q_fit)
            b_values.append(b_fit)
        except:
            continue  # Se a otimização falhar, ignora essa amostra
    
    mean_q = np.mean(q_values)
    std_q = np.std(q_values, ddof=1)
    mean_b = np.mean(b_values)
    std_b = np.std(b_values, ddof=1)
    
    return mean_q, std_q, mean_b, std_b


def remove_last_Y_entries(dim, alpha_a, alpha_g, Y):
    """
    Remove as Y últimas colunas do arquivo degree.npy e as Y últimas linhas do arquivo filenames_degree.csv.

    :param directory: Caminho do diretório onde estão os arquivos degree.npy e filenames_degree.csv
    :param Y: Número de colunas/linhas a serem removidas
    """
    directory = f"../../data/N_100000/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/"
    
    degree_path = os.path.join(directory, "degree.npy")
    filenames_path = os.path.join(directory, "filenames_degree.csv")

    # distances_path = os.path.join(directory, "distances.npy")
    # filenames_dist_path = os.path.join(directory, "filenames_distances.csv")

    # Verificar se os arquivos existem
    if not os.path.exists(degree_path):
        print(f"Arquivo {degree_path} não encontrado.")
        return
    if not os.path.exists(filenames_path):
        print(f"Arquivo {filenames_path} não encontrado.")
        return
    # if not os.path.exists(distances_path):
    #     print(f"Arquivo {distances_path} não encontrado.")
    #     return
    # if not os.path.exists(filenames_dist_path):
    #     print(f"Arquivo {filenames_dist_path} não encontrado.")
    #     return
    
    # Carregar degree.npy e remover as últimas Y colunas
    degree_data = np.load(degree_path)
    if degree_data.shape[1] <= Y:
        print("Erro: Tentativa de remover mais colunas do que as existentes.")
        return

    updated_degree_data = degree_data[:, :-Y]
    np.save(degree_path, updated_degree_data)
    print(f"Removidas {Y} colunas do arquivo {degree_path}.")    
    
    # Carregar distances.npy e remover as últimas Y colunas
    # distances_data = np.load(distances_path)
    # if distances_data.shape[1] <= Y:
    #     print("Erro: Tentativa de remover mais colunas do que as existentes.")
    #     return
    
    # updated_distances_data = distances_data[:, :-Y]
    # np.save(distances_path, updated_distances_data)
    # print(f"Removidas {Y} colunas do arquivo {distances_path}.") 

    # Carregar filenames_degree.csv e remover as últimas Y linhas
    df = pd.read_csv(filenames_path)
    if len(df) <= Y:
        print("Erro: Tentativa de remover mais linhas do que as existentes.")
        return
    df_updated = df.iloc[:-Y]
    df_updated.to_csv(filenames_path, index=False)
    print(f"Removidas {Y} linhas do arquivo {filenames_path}.")
    
    # Carregar filenames_degree.csv e remover as últimas Y linhas
    # df_dist = pd.read_csv(filenames_path)
    # if len(df_dist) <= Y:
    #     print("Erro: Tentativa de remover mais linhas do que as existentes.")
    #     return
    
    # df_dist_updated = df_dist.iloc[:-Y]
    # df_dist_updated.to_csv(filenames_dist_path, index=False)
    # print(f"Removidas {Y} linhas do arquivo {filenames_dist_path}.")

def round_measurement(value, error):
    """
    Arredonda o erro para a primeira casa decimal não nula
    e ajusta o valor da medida para ter o mesmo número de casas decimais.
    """
    if error == 0:
        return f"{value:.1f} ± 0.0"

    # Encontra a ordem de grandeza do erro
    order = -int(math.floor(math.log10(abs(error))))
    rounded_error = round(error, order)

    # Caso o erro arredondado tenha ficado 0, aumentar precisão
    while rounded_error == 0.0:
        order += 1
        rounded_error = round(error, order)

    # Arredondar a medida para o mesmo número de casas decimais do erro
    rounded_value = round(value, order)

    # Formatar com o mesmo número de casas decimais
    fmt = f"{{:.{order}f}}"
    return f"{fmt.format(rounded_value)} \\pm {fmt.format(rounded_error)}"



def save_json_distributions(N: int, dim: list[int], alpha_a_v: list[float], alpha_g_v: list[float]):
    """
        N (int): Number of nodes in network
        dim (list(int)): list of all values of dimensions used
        alpha_a_v (list(float)): list of all values alpha_a (with alpha_g = 2.00)
        alpha_g_v (list(float)): list of all values alpha_g (with alpha_a = 2.00)
    """

    alpha_ag_f = 2.0

    # Dicionários para armazenar dados separados
    data_distance = []
    data_degree = []

    types = [
        {
            "name": "distance",
            "csv_name": "distances_distribution_log.csv",
            "npy_name": "distances.npy",
            "csv_columns": ["ds", "pds"],
            "data_list": data_distance,
            "output_path": '../../data/distances_distributions.json'
        },
        {
            "name": "degree",
            "csv_name": "degree_distribution_linear.csv",
            "npy_name": "degree.npy",
            "csv_columns": ["k", "pk"],
            "data_list": data_degree,
            "output_path": '../../data/degree_distributions.json'
        }
    ]

    for d in dim:
        for t in types:
            # Loop variando alpha_g
            for alpha_g in alpha_g_v:
                folder = f"../../data/N_{N}/dim_{d}/alpha_a_{alpha_ag_f:.2f}_alpha_g_{alpha_g:.2f}"
                try:
                    df = pd.read_csv(os.path.join(folder, t["csv_name"]), delimiter=' ')
                    dados = np.load(os.path.join(folder, t["npy_name"]))
                except Exception as e:
                    print(f"Erro ao ler {folder} ({t['name']}): {e}")
                    continue

                N_s = int(dados.size / N)

                entry = {
                    "N": N,
                    "alpha_A": round(alpha_ag_f, 2),
                    "alpha_G": round(alpha_g, 2),
                    "dim": d,
                    "N_s": N_s,
                }

                if t["name"] == "distance":
                    entry.update({
                        "deltaS": df[t["csv_columns"][0]].tolist(),
                        "pdeltaS": df[t["csv_columns"][1]].tolist()
                    })
                else:
                    entry.update({
                        "k": df[t["csv_columns"][0]].tolist(),
                        "Pk": df[t["csv_columns"][1]].tolist()
                    })

                t["data_list"].append(entry)

            # Loop variando alpha_a
            for alpha_a in alpha_a_v:
                folder = f"../../data/N_{N}/dim_{d}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_ag_f:.2f}"
                try:
                    df = pd.read_csv(os.path.join(folder, t["csv_name"]), delimiter=' ')
                    dados = np.load(os.path.join(folder, t["npy_name"]))
                except Exception as e:
                    print(f"Erro ao ler {folder} ({t['name']}): {e}")
                    continue

                N_s = int(dados.size / N)

                entry = {
                    "N": N,
                    "alpha_A": round(alpha_a, 2),
                    "alpha_G": round(alpha_ag_f, 2),
                    "dim": d,
                    "N_s": N_s,
                }

                if t["name"] == "distance":
                    entry.update({
                        "deltaS": df[t["csv_columns"][0]].tolist(),
                        "pdeltaS": df[t["csv_columns"][1]].tolist()
                    })
                else:
                    entry.update({
                        "k": df[t["csv_columns"][0]].tolist(),
                        "Pk": df[t["csv_columns"][1]].tolist()
                    })

                t["data_list"].append(entry)

    # Salvar os arquivos separados
    for t in types:
        with open(t["output_path"], 'w') as f:
            json.dump(t["data_list"], f, indent=4)
        print(f"Arquivo '{t['output_path']}' salvo com sucesso.")


def json_to_dataframe_with_lists(json_path: str) -> pd.DataFrame:
    """
    Converte um arquivo JSON de distribuições em um DataFrame,
    mantendo 'k' e 'Pk' como listas em cada linha.

    Args:
        json_path (str): Caminho para o arquivo JSON.

    Returns:
        pd.DataFrame: DataFrame onde cada linha contém os parâmetros
                      (N, alpha_A, alpha_G, dim, N_s) e as listas 'k' e 'Pk'.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    return df

def get_k_pk_from_dataframe(
    df: pd.DataFrame,
    dim: int,
    fixed_param: str,
    fixed_value: float,
    varying_param_list: list[float]
):
    """
    Extrai listas de k e Pk a partir do DataFrame, para uma dimensão e um parâmetro fixo.

    Args:
        df (pd.DataFrame): DataFrame convertido do JSON.
        dim (int): Dimensão (1, 2, 3, 4).
        fixed_param (str): Qual parâmetro fica fixo: 'alpha_A' ou 'alpha_G'.
        fixed_value (float): Valor fixo de alpha_A ou alpha_G.
        varying_param_list (list[float]): Lista dos valores do parâmetro que varia.

    Returns:
        tuple: (list_k, list_pk) -> listas de listas com os dados de k e P(k).
    """
    assert fixed_param in ['alpha_A', 'alpha_G'], "fixed_param deve ser 'alpha_A' ou 'alpha_G'"

    varying_param = 'alpha_G' if fixed_param == 'alpha_A' else 'alpha_A'

    df_filtered = df[
        (df['dim'] == dim) &
        (df[fixed_param] == fixed_value) &
        (df[varying_param].isin(varying_param_list))
    ]

    k_list = df_filtered['k'].tolist()
    pk_list = df_filtered['Pk'].tolist()

    return k_list, pk_list

