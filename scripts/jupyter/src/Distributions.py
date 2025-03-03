import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  
import glob
import gzip
from multiprocessing import Pool
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.stats import kstest


# N: number of nodes
# d: dimension
# alpha_a:.2f: parameter for preferential attrachment
# alpha_g:.2f: parameter for probability of distances nodes

def process_file(index_file_tuple):
    """Processa um único arquivo para extrair os graus dos nós."""
    i, file = index_file_tuple
    n_lines = 10**5  # Número fixo de linhas por arquivo
    degrees = np.zeros(n_lines, dtype=np.int32)  # Criar array fixo (evita append)
    
    try:
        with gzip.open(file, 'rt') as gzip_file:  # 'rt' já faz a decodificação
            count = 0
            for line in gzip_file:
                if line.startswith("degree "):  # Mais rápido que strip() e [:6]
                    degrees[count] = int(line[7:])  # Extrai grau e adiciona no array
                    count += 1
                    if count == n_lines:  # Para evitar processamento extra
                        break
    except EOFError:
        print(f"Arquivo {file} está corrompido ou incompleto. Apagando arquivo.")
        os.remove(file)
        return None

    return i, degrees, os.path.basename(file)

def degree_file(N, dim, alpha_a, alpha_g):
    """Processa múltiplos arquivos .gml.gz e salva os graus extraídos."""
    # Pasta com os arquivos
    path_folder = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/gml"
    output_dir = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}"
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Listar todos os arquivos .gml.gz
    all_files = sorted(glob.glob(os.path.join(path_folder, "*.gml.gz")))  # Ordenação para consistência

    # Verificar se a pasta está vazia
    if not all_files:
        print("Folder empty")
        return

    n_files = len(all_files)  # Número total de arquivos
    n_lines = 10**5           # Número de linhas por arquivo

    # Caminhos dos arquivos de saída
    degree_npy_path = os.path.join(output_dir, "degree.npy")
    filenames_csv_path = os.path.join(output_dir, "filenames_degree.csv")

    # Verificar se já existem dados processados
    if os.path.exists(filenames_csv_path):
        df_existing = pd.read_csv(filenames_csv_path)
        processed_files = set(df_existing["filenames"].tolist())
    else:
        df_existing = pd.DataFrame(columns=["filenames"])
        processed_files = set()

    # Filtrar arquivos que ainda não foram processados
    new_files = [file for file in all_files if os.path.basename(file) not in processed_files]

    if not new_files:
        print("Todos os arquivos já foram processados. Nada a fazer.")
        return

    print(f"Processando {len(new_files)} novos arquivos...")

    # Processar apenas os arquivos novos em paralelo
    with Pool() as pool:
        results = pool.map(process_distance_file, enumerate(new_files))
    
    results = [res for res in results if res is not None]
    # Organizar os novos resultados
    new_filenames = []
    new_data = np.empty((n_lines, len(results)), dtype=np.int32)

    for i, (idx, degrees, filename) in enumerate(results):
        new_data[:, i] = degrees
        new_filenames.append(filename)

    # Se degree.npy já existe, carregar e adicionar os novos dados
    if os.path.exists(degree_npy_path):
        existing_data = np.load(degree_npy_path)
        updated_data = np.hstack((existing_data, new_data))
    else:
        updated_data = new_data

    # Atualizar e salvar degree.npy
    np.save(degree_npy_path, updated_data)

    # Atualizar e salvar filenames_degree.csv
    df_new = pd.DataFrame({"filenames": new_filenames})
    df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    df_updated.to_csv(filenames_csv_path, index=False)

    print(f"Processamento concluído. Dados salvos em {output_dir}")

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
        N, dim, alpha_a, alpha_g, propertie = kwargs["N"], kwargs["dim"], kwargs["alpha_a"], kwargs["alpha_g"], kwargs["propertie"]

        # Define o caminho do arquivo
        save_path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/{propertie}_distribution_linear.csv"

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

# Logbinnig_test

def drop_zeros(a_list):
    """Remove valores zero da lista."""
    return [i for i in a_list if i > 0]

def log_binning(counter_dict, bin_count, save=False, **kwargs):
    """Realiza binagem logarítmica e normaliza P(k). Se save=True, salva os dados."""
    
    keys = np.array(list(counter_dict.keys()), dtype=np.float64)
    values = np.array(list(counter_dict.values()), dtype=np.float64)

    # Definir os limites dos bins logarítmicos
    min_x = np.log10(min(drop_zeros(keys)))
    max_x = np.log10(max(keys))

    bins = np.logspace(min_x, max_x, num=bin_count)

    # Calcular os histogramas
    hist_counts, bin_edges = np.histogram(keys, bins=bins, weights=values, density=False)
    bin_counts, _ = np.histogram(keys, bins=bins)

    # Evitar divisões por zero
    valid_bins = bin_counts > 0
    Pk = np.zeros_like(hist_counts, dtype=np.float64)
    k = np.zeros_like(hist_counts, dtype=np.float64)

    # Calcular P(k) e k médio para cada bin válido
    Pk[valid_bins] = hist_counts[valid_bins] / bin_counts[valid_bins]
    k[valid_bins] = np.histogram(keys, bins=bins, weights=keys)[0][valid_bins] / bin_counts[valid_bins]

    # Normalização de P(k) para garantir que ∑P(k) = 1
    Pk_sum = np.sum(Pk)
    if Pk_sum > 0:
        Pk /= Pk_sum

    # Remover valores NaN ou infinitos
    valid_values = np.isfinite(k) & np.isfinite(Pk)
    k = k[valid_values]
    Pk = Pk[valid_values]

    # Salvar se save=True
    if save:
        # Verifica se os argumentos necessários foram passados
        required_keys = ["N", "dim", "alpha_a", "alpha_g"]
        if not all(arg in kwargs for arg in required_keys):
            raise ValueError(f"Se save=True, os argumentos {required_keys} devem ser fornecidos.")

        # Obtém os valores das variáveis
        N = kwargs["N"]
        dim = kwargs["dim"]
        alpha_a = kwargs["alpha_a"]
        alpha_g = kwargs["alpha_g"]
        propertie = kwargs["propertie"]

        # Define o caminho do arquivo
        save_path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/{propertie}_distribution_log.csv"

        # Salva os dados
        if(propertie=="degree"):
            df = pd.DataFrame(data={"k": k, "pk": Pk})
        elif(propertie=="distances"):
            df = pd.DataFrame(data={"ds": k, "pds": Pk})
        df.to_csv(save_path, sep=' ', index=False)
        print(f"Arquivo salvo em {save_path}")

    return k, Pk



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
    Otimiza os parâmetros q e b do ajuste q-exponencial dentro de uma faixa restrita.
    
    Parâmetros:
        k (array): Valores de k.
        pk (array): Valores de P(k).
        q_initial (float): Valor inicial de q baseado na relação teórica.
        b_initial (float): Valor inicial de b baseado na relação teórica.
        delta_q (float): Intervalo de variação permitido para q.
        delta_b (float): Intervalo de variação permitido para b.

    Retorna:
        fitted_q (float): Melhor valor ajustado de q.
        fitted_b (float): Melhor valor ajustado de b.
    """
    # Definir a função q-exponencial com normalização segura
    def normalized_constant_safe(q, b, k_vals):
        try:
            integral, _ = quad(lambda x: (1 - (1 - q) * (x / b)) ** (1 / (1 - q)), np.min(k_vals), np.max(k_vals))
            return 1 / integral if integral > 0 else 1
        except:
            return 1

    def q_exp_safe(x, q, b):
        A = normalized_constant_safe(q, b, k)
        return A * (1 - (1 - q) * (x / b)) ** (1 / (1 - q))

    # Ajustar corretamente os limites para garantir valores bem definidos
    bounds_ultra_fine = ([q_initial - delta_q, b_initial - delta_b], [q_initial + delta_q, b_initial + delta_b])

    # Ajuste ultra fino dentro dessa faixa extremamente reduzida
    popt_ultra_fine, _ = curve_fit(q_exp_safe, k, pk, p0=[q_initial, b_initial], bounds=bounds_ultra_fine)

    # Parâmetros finais ajustados
    fitted_q, fitted_b = popt_ultra_fine

    return fitted_q, fitted_b

def ln_q(k, pk, q, eta):
    k_values = np.zeros(len(k))
    for i in range(len(k)):
        k_values[i] = (1+(q-1)*(k/eta))**(1/(1-q))
    P0 = sum(k_values)
    return ((pk/P0)**(1-q)-1)/(1-q)

def process_distance_file(index_file_tuple):
    """Processa um único arquivo para extrair as distâncias."""
    i, file = index_file_tuple
    n_lines = 10**5  # Número fixo de linhas por arquivo
    distances = np.zeros(n_lines, dtype=np.float32)  # Criar array fixo para evitar append
    
    try:
        with gzip.open(file, 'rt') as gzip_file:  # 'rt' já faz a decodificação
            count = 0
            for line in gzip_file:
                if line.startswith("distance "):  # Mais rápido que strip() e [:9]
                    distances[count] = float(line[9:])  # Extrai distância e adiciona no array
                    count += 1
                    if count == n_lines:  # Para evitar processamento extra
                        break
    except EOFError:
        print(f"Arquivo {file} está corrompido ou incompleto. Apagando arquivo.")
        os.remove(file)
        return None

    return i, distances, os.path.basename(file)


def distance_file(N, dim, alpha_a, alpha_g):
    """Processa múltiplos arquivos .gml.gz e salva as distâncias extraídas."""
    # Pasta com os arquivos
    path_folder = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/gml"
    output_dir = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}"
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Listar todos os arquivos .gml.gz
    all_files = sorted(glob.glob(os.path.join(path_folder, "*.gml.gz")))  # Ordenação para consistência

    # Verificar se a pasta está vazia
    if not all_files:
        print("Folder empty")
        return

    n_files = len(all_files)  # Número total de arquivos
    n_lines = 10**5           # Número de linhas por arquivo

    # Caminhos dos arquivos de saída
    distances_npy_path = os.path.join(output_dir, "distances.npy")
    filenames_csv_path = os.path.join(output_dir, "filenames_distances.csv")

    # Verificar se já existem dados processados
    if os.path.exists(filenames_csv_path):
        df_existing = pd.read_csv(filenames_csv_path)
        processed_files = set(df_existing["filenames"].tolist())
    else:
        df_existing = pd.DataFrame(columns=["filenames"])
        processed_files = set()

    # Filtrar arquivos que ainda não foram processados
    new_files = [file for file in all_files if os.path.basename(file) not in processed_files]

    if not new_files:
        print("Todos os arquivos já foram processados. Nada a fazer.")
        return

    print(f"Processando {len(new_files)} novos arquivos...")

    # Processar apenas os arquivos novos em paralelo
    with Pool() as pool:
        results = pool.map(process_distance_file, enumerate(new_files))
    
    results = [res for res in results if res is not None]
    # Organizar os novos resultados
    new_filenames = []
    new_data = np.empty((n_lines, len(results)), dtype=np.float32)

    for i, (idx, distances, filename) in enumerate(results):
        new_data[:, i] = distances
        new_filenames.append(filename)

    # Se distances.npy já existe, carregar e adicionar os novos dados
    if os.path.exists(distances_npy_path):
        existing_data = np.load(distances_npy_path)
        updated_data = np.hstack((existing_data, new_data))
    else:
        updated_data = new_data

    # Atualizar e salvar distances.npy
    np.save(distances_npy_path, updated_data)

    # Atualizar e salvar filenames_distances.csv
    df_new = pd.DataFrame({"filenames": new_filenames})
    df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    df_updated.to_csv(filenames_csv_path, index=False)

    print(f"Processamento concluído. Dados salvos em {output_dir}")



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
            q_fit, b_fit = optimize_q_exp(k_sample, pk_sample, q_initial=Q_init, b_initial=B_init, delta_q=0.01, delta_b=0.01)
            q_values.append(q_fit)
            b_values.append(b_fit)
        except:
            continue  # Se a otimização falhar, ignora essa amostra
    
    mean_q = np.mean(q_values)
    std_q = np.std(q_values)
    mean_b = np.mean(b_values)
    std_b = np.std(b_values)
    
    return mean_q, std_q, mean_b, std_b