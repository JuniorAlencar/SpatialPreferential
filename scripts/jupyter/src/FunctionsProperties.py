import numpy as np
import pandas as pd
import os
import gzip
import glob
import networkx as nx

from io import StringIO
from decimal import Decimal, getcontext
from IPython.display import clear_output
import statsmodels.api as sm # Linear regression
import shutil
import re

from typing import Annotated

def remove_outliers(N, dim, alpha_a, alpha_g):
    files = f"../../data_2/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/properties_set.txt"
    if(os.path.exists(files)):
        df = pd.read_csv(files,delimiter=' ')        
        R = df["#ass_coeff"]
        q1 = R.quantile(0.25)
        q3 = R.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filtrar linhas que não são outliers
        df = df[(R >= lower_bound) & (R <= upper_bound)]

        # Salvar o DataFrame atualizado no mesmo arquivo
        df.to_csv(files, sep=' ', index=False)
        print(f"Outliers removidos e arquivo atualizado: {files}")

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

def move_to_data():
    path = "../../data"
    if not os.path.exists(path):
        os.makedirs(path)
    
    shutil.copy("../../all_data.txt", path)
    shutil.copy("../../coeff_linear.txt", path)

def data_to_move():
    path = "../../"
    shutil.copy("../../data/all_data.txt", path)
    shutil.copy("../../data/coeff_linear.txt", path)

# All combinations (N, dim, alpha_a, alpha_g) folders
def extract_alpha_values(folder_data):
    # Caminho inicial
#    base_path = "../../data_2"
    base_path = folder_data

    # Regex para capturar nvalue, dvalue, alpha_a (aavalue) e alpha_g (agvalue)
    pattern = r"N_(\d+)/m0_(\d+)/dim_(\d+)/alpha_a_([\d]+\.\d{2})_alpha_g_([\d]+\.\d{2})"

    # Estrutura para armazenar as combinações encontradas
    combinations = set()

    # Percorrer todas as subpastas a partir de base_path
    for root, dirs, files in os.walk(base_path):
        match = re.search(pattern, root)
        if match:
            nvalue = int(match.group(1))  # nvalue como inteiro
            m0value = int(match.group(2))  # nvalue como inteiro
            dvalue = int(match.group(3))  # dvalue como inteiro
            aavalue = float(match.group(4))  # alpha_a como float
            agvalue = float(match.group(5))  # alpha_g como float
            combinations.add((nvalue, m0value ,dvalue, round(aavalue, 2), round(agvalue, 2)))
    return combinations

def update_headers(folder_data):
    # Caminho inicial
    base_path = folder_data

    # Cabeçalhos
    old_header = "#short_path,#diamater,#ass_coeff"
    new_header = "#mean shortest path,# diamater,#assortativity coefficient"

    # Percorrer todas as pastas e subpastas
    for root, dirs, files in os.walk(base_path):
        if 'prop' in root:  # Processar apenas as pastas 'prop'
            for file_name in files:
                if file_name.endswith(".csv"):  # Verificar apenas arquivos .csv
                    file_path = os.path.join(root, file_name)
                    
                    # Ler e processar o arquivo
                    with open(file_path, "r") as file:
                        lines = file.readlines()
                    
                    # Verificar e substituir o cabeçalho, se necessário
                    if lines and lines[0].strip() == old_header:
                        print(f"Atualizando cabeçalho no arquivo: {file_path}")
                        lines[0] = new_header + "\n"  # Substituir o cabeçalho
                        
                        # Escrever as alterações de volta no arquivo
                        with open(file_path, "w") as file:
                            file.writelines(lines)
                    else:
                        print(f"Nenhuma atualização necessária para o arquivo: {file_path}")


import os, glob
import pandas as pd

def all_properties_file(N, m0, dim, alpha_a, alpha_g):
    # Diretórios
    path_d    = f"../../data/N_{N}/m0_{m0}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/prop"
    path_save = f"../../data/N_{N}/m0_{m0}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}"
    print(f"N = {N}, m0 = {m0}, dim = {dim}, alpha_a = {alpha_a:.2f}, alpha_g = {alpha_g:.2f}")

    properties_file = os.path.join(path_save, "properties_set.txt")
    filenames_file  = os.path.join(path_save, "filenames.txt")

    if not os.path.exists(path_d):
        print(f"O diretório {path_d} não existe. Nada a ser feito.")
        return

    all_files = [f for f in glob.glob(os.path.join(path_d, "*.csv")) if os.path.getsize(f) > 0]
    if not all_files:
        print(f"A pasta {path_d} está vazia ou contém apenas arquivos vazios. Nada a ser feito.")
        return

    # Arquivos já processados
    if os.path.exists(filenames_file):
        with open(filenames_file, "r") as f:
            filenames_set = set(f.read().splitlines())
    else:
        filenames_set = set()

    # Schema unificado do arquivo final (sem '#', nomes padronizados)
    cols_out = [
        "MeanShortestPathDijkstra",
        "MeanShortestPathBFS",
        "Ass_Spearman",
        "Ass_Spearman_error",
        "Ass_Newman",
        "Ass_Newman_error",
        "ClusterCoefficient",
    ]

    # DataFrame existente (se houver)
    if os.path.exists(properties_file):
        df = pd.read_csv(properties_file, sep=' ')
    else:
        df = pd.DataFrame(columns=cols_out)

    # Mapas de possíveis nomes de coluna no CSV de entrada → nome padronizado
    # (aceita o cabeçalho com ou sem '#', com/sem typo "Dijstrika")
    candidates = {
        "MeanShortestPathDijkstra": [
            "#MeanShortestPathDijstrika", "#MeanShortestPathDijkstra",
            "MeanShortestPathDijstrika", "MeanShortestPathDijkstra"
        ],
        "MeanShortestPathBFS": [
            "#MeanShortestPathBFS", "MeanShortestPathBFS"
        ],
        "Ass_Spearman": [
            "#Ass_Spearman", "Ass_Spearman"
        ],
        "Ass_Spearman_error": [
            "#Ass_Spearman_error", "Ass_Spearman_error"
        ],
        "Ass_Newman": [
            "#Ass_Newman", "Ass_Newman"
        ],
        "Ass_Newman_error": [
            "#Ass_Newman_error", "Ass_Newman_error"
        ],
        "ClusterCoefficient": [
            "#ClusterCoefficient", "ClusterCoefficient"
        ],
    }

    def pick_value(row, keys):
        for k in keys:
            if k in row:
                return row[k]
        missing = ", ".join(keys)
        raise KeyError(f"Coluna não encontrada no CSV: tentei {missing}")

    updated = False
    new_rows = []

    for file in all_files:
        filename = os.path.basename(file)

        if filename in filenames_set:
            # já processado — pode remover
            os.remove(file)
            continue

        # lê CSV com vírgula e ignora espaços depois da vírgula
        new_data = pd.read_csv(file, sep=',', skipinitialspace=True)
        if new_data.empty:
            os.remove(file)
            print(f"Arquivo vazio {filename} foi deletado.")
            continue

        row0 = new_data.iloc[0]

        try:
            new_row = {
                "MeanShortestPathDijkstra": pick_value(row0, candidates["MeanShortestPathDijkstra"]),
                "MeanShortestPathBFS"     : pick_value(row0, candidates["MeanShortestPathBFS"]),
                "Ass_Spearman"            : pick_value(row0, candidates["Ass_Spearman"]),
                "Ass_Spearman_error"      : pick_value(row0, candidates["Ass_Spearman_error"]),
                "Ass_Newman"              : pick_value(row0, candidates["Ass_Newman"]),
                "Ass_Newman_error"        : pick_value(row0, candidates["Ass_Newman_error"]),
                "ClusterCoefficient"      : pick_value(row0, candidates["ClusterCoefficient"]),
            }
        except KeyError as e:
            print(f"[WARN] {filename}: {e}. Pulando este arquivo.")
            continue

        new_rows.append(new_row)
        filenames_set.add(filename)
        updated = True
        os.remove(file)  # remove após consumir

    if updated:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        # salva com espaço para manter compatibilidade anterior
        df.to_csv(properties_file, sep=' ', index=False)
        with open(filenames_file, 'w') as f:
            f.write("\n".join(sorted(filenames_set)))
        print(f"Arquivos {properties_file} e {filenames_file} atualizados com sucesso.")
    else:
        print("Nenhuma atualização necessária. Todos os arquivos já estavam processados.")

import os, re, glob, shutil
import pandas as pd
import numpy as np

def all_data(folder_data):
    base_path = folder_data

    # Agora capturando também o m0
    pattern = r"N_(\d+)/m0_(\d+)/dim_(\d+)/alpha_a_([\d]+\.\d{2})_alpha_g_([\d]+\.\d{2})"

    out = {
        "N": [], "m0": [], "dim": [], "alpha_a": [], "alpha_g": [], "N_samples": [],
        # Dijkstra/BFS/Cluster → média e SEM
        "dijkstra_mean": [], "dijkstra_sem": [],
        "bfs_mean": [], "bfs_sem": [],
        "cluster_mean": [], "cluster_sem": [],
        # Assortatividades com erro por amostra → média ponderada e erro combinado
        "ass_spearman_mean": [], "ass_spearman_err": [],
        "ass_newman_mean": [], "ass_newman_err": [],
    }

    # mapeamento de nomes (flexível caso apareça com '#')
    cols_map = {
        "dijkstra": ["MeanShortestPathDijkstra", "#MeanShortestPathDijkstra", "#MeanShortestPathDijstrika"],
        "bfs":      ["MeanShortestPathBFS", "#MeanShortestPathBFS"],
        "spear":    ["Ass_Spearman", "#Ass_Spearman"],
        "spear_e":  ["Ass_Spearman_error", "#Ass_Spearman_error"],
        "newman":   ["Ass_Newman", "#Ass_Newman"],
        "newman_e": ["Ass_Newman_error", "#Ass_Newman_error"],
        "cluster":  ["ClusterCoefficient", "#ClusterCoefficient"],
    }

    def pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def wmean_and_err(values, errors):
        """Média ponderada por 1/err^2; retorna (mean, err_comb). Fallback p/ média simples."""
        vals = np.asarray(values, dtype=float)
        errs = np.asarray(errors, dtype=float)
        mask = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
        if mask.sum() >= 1:
            w = 1.0 / (errs[mask] ** 2)
            m = np.sum(w * vals[mask]) / np.sum(w)
            se = 1.0 / np.sqrt(np.sum(w))   # erro combinado da média ponderada
            return float(m), float(se)
        # fallback: média simples e SEM
        vals2 = vals[np.isfinite(vals)]
        if vals2.size == 0:
            return np.nan, np.nan
        return float(np.mean(vals2)), float(pd.Series(vals2).sem())

    for root, dirs, files in os.walk(base_path, topdown=False):
        m = re.search(pattern, root)
        if not m:
            continue

        nvalue = int(m.group(1))
        m0val  = int(m.group(2))
        dvalue = int(m.group(3))
        aav    = float(m.group(4))
        agv    = float(m.group(5))

        properties_file = os.path.join(root, "properties_set.txt")
        prop_folder     = os.path.join(root, "prop")

        # limpeza de pasta sem dados
        if not os.path.exists(properties_file) and (not os.path.exists(prop_folder) or not os.listdir(prop_folder)):
            print(f"❌ Removendo pasta vazia e sem dados: {root}")
            shutil.rmtree(root, ignore_errors=True)
            continue

        if not os.path.exists(properties_file):
            continue

        print(f"N = {nvalue}, m0 = {m0val}, dim = {dvalue}, alpha_a = {aav:.2f}, alpha_g = {agv:.2f}")
        df = pd.read_csv(properties_file, sep=' ')

        # localizar colunas
        c_dij   = pick_col(df, cols_map["dijkstra"])
        c_bfs   = pick_col(df, cols_map["bfs"])
        c_spear = pick_col(df, cols_map["spear"])
        c_spear_e = pick_col(df, cols_map["spear_e"])
        c_new   = pick_col(df, cols_map["newman"])
        c_new_e = pick_col(df, cols_map["newman_e"])
        c_clu   = pick_col(df, cols_map["cluster"])

        # número de amostras válidas (usa coluna Dijkstra como proxy; se faltar, usa qualquer)
        if c_dij and c_dij in df:
            n_samples = df[c_dij].dropna().shape[0]
        else:
            n_samples = len(df)

        # agrega métricas simples (média + SEM)
        def mean_sem(colname):
            if colname and colname in df:
                series = pd.to_numeric(df[colname], errors="coerce")
                return float(series.mean()), float(series.sem())
            return np.nan, np.nan

        dij_mean, dij_sem   = mean_sem(c_dij)
        bfs_mean, bfs_sem   = mean_sem(c_bfs)
        clu_mean, clu_sem   = mean_sem(c_clu)

        # assortatividades com erro por amostra → média ponderada
        if c_spear and c_spear_e and (c_spear in df) and (c_spear_e in df):
            spear_mean, spear_err = wmean_and_err(
                pd.to_numeric(df[c_spear], errors="coerce"),
                pd.to_numeric(df[c_spear_e], errors="coerce"),
            )
        else:
            spear_mean, spear_err = (np.nan, np.nan)

        if c_new and c_new_e and (c_new in df) and (c_new_e in df):
            new_mean, new_err = wmean_and_err(
                pd.to_numeric(df[c_new], errors="coerce"),
                pd.to_numeric(df[c_new_e], errors="coerce"),
            )
        else:
            new_mean, new_err = (np.nan, np.nan)

        # preencher saída
        out["N"].append(nvalue)
        out["m0"].append(m0val)
        out["dim"].append(dvalue)
        out["alpha_a"].append(round(aav, 2))
        out["alpha_g"].append(round(agv, 2))
        out["N_samples"].append(n_samples)

        out["dijkstra_mean"].append(dij_mean)
        out["dijkstra_sem"].append(dij_sem)
        out["bfs_mean"].append(bfs_mean)
        out["bfs_sem"].append(bfs_sem)
        out["cluster_mean"].append(clu_mean)
        out["cluster_sem"].append(clu_sem)

        out["ass_spearman_mean"].append(spear_mean)
        out["ass_spearman_err"].append(spear_err)
        out["ass_newman_mean"].append(new_mean)
        out["ass_newman_err"].append(new_err)

    # montar DF, filtrar e salvar
    df_all = pd.DataFrame(out)
    if not df_all.empty:
        df_all = df_all[df_all["alpha_g"] >= 1.0]
        df_all = df_all.sort_values("N_samples", ascending=False).drop_duplicates(
            subset=["N", "m0", "dim", "alpha_a", "alpha_g"], keep="first"
        ).reset_index(drop=True)

    df_all.to_csv("../../data/all_data.txt", sep=' ', index=False)
    print("✅ Processamento concluído!")


# Linear regression with errors in parameters
def linear_regression(X,Y,Erro_Y,Parameter):
    # Dados de exemplo
    x = X
    y = Y

    # Erros associados às medições no eixo y
    y_errors = Erro_Y

    # Calcular a regressão linear ponderada
    coefficients, cov_matrix = np.polyfit(x, y, deg=1, w=1/y_errors, cov=True)

    # Extrair os coeficientes e as incertezas
    slope = coefficients[0]
    intercept = coefficients[1]
    slope_error = np.sqrt(cov_matrix[0, 0])
    intercept_error = abs(np.sqrt(cov_matrix[1, 1]))
    
    # Return a, b, a_err, b_err
    if( Parameter == True):
        return slope, intercept, slope_error, intercept_error
    
    # Return y, where y = a*x + b
    else:
        return intercept + slope*x

# dataframe with all beta, xi parameters, with relative erros
# Prop = beta(alpha_a, alpha_g, dim)*ln(N) + xi(alpha_a, alpha_g, dim)
def parameters_calculate(df: pd.DataFrame, N: list, dimensions: list, alpha_filter: list):
    
    coeff_all = {"alpha_a":[], "alpha_g":[], "dim":[], 
             "A_ass":[], "A_ass_err":[], "B_ass":[], "B_ass_err":[],
             "A_diameter":[], "A_diameter_err":[], "B_diameter":[], "B_diameter_err":[],
             "A_short":[], "A_short_err":[], "B_short":[], "B_short_err":[],}

    properties = ["ass", "diameter", "short"]
    
    # Loop sobre cada dimensão e gera o gráfico correspondente
    for dim_idx, dim in enumerate(dimensions):
        for alpha in alpha_filter:
            for j, prop_name in enumerate(properties):
                # Inicializa listas de dados
                N_aux = []
                prop = []
                prop_err = []
                
                for n in N:
                    # Filtra o DataFrame para a dimensão, valor de N, alpha_g e alpha_a específicos
                    df_dim = df[(df['dim'] == dim) & (df['N'] == n)]
                    df_dim_alpha_a = df_dim[(df_dim["alpha_g"] == 2.00) & (df_dim["alpha_a"] == round(alpha, 2))]
                    
                    if not df_dim_alpha_a.empty:  # Verifica se o filtro retornou dados
                        N_aux.append(n)
                        
                        if prop_name == "ass":
                            value = df_dim_alpha_a["ass_coeff_mean"].values[0]
                            error = df_dim_alpha_a["ass_coeff_err"].values[0]
                        elif prop_name == "diameter":
                            value = df_dim_alpha_a["diameter_mean"].values[0]
                            error = df_dim_alpha_a["diameter_err"].values[0]
                        elif prop_name == "short":
                            value = df_dim_alpha_a["short_mean"].values[0]
                            error = df_dim_alpha_a["short_err"].values[0]
                        
                        prop.append(value)
                        prop_err.append(error)

                # Confere se as listas têm valores válidos antes da regressão
                if len(N_aux) > 1 and len(prop) == len(N_aux):
                    regression = linear_regression(np.log(N_aux), np.array(prop), np.array(prop_err), Parameter=True)
                    coeff_all[f"A_{prop_name}"].append(regression[0])
                    coeff_all[f"B_{prop_name}"].append(regression[1])
                    coeff_all[f"A_{prop_name}_err"].append(regression[2])
                    coeff_all[f"B_{prop_name}_err"].append(regression[3])
                else:
                    print(f"Dados insuficientes para regressão: dim={dim}, alpha={alpha}, prop={prop_name}")
                    coeff_all[f"A_{prop_name}"].append(None)
                    coeff_all[f"B_{prop_name}"].append(None)
                    coeff_all[f"A_{prop_name}_err"].append(None)
                    coeff_all[f"B_{prop_name}_err"].append(None)

            coeff_all["alpha_a"].append(alpha)
            coeff_all["dim"].append(dim)
            coeff_all["alpha_g"].append(2)

    # Criar DataFrame final
    df_coeff = pd.DataFrame(data=coeff_all)
    df_coeff.to_csv("../coeff_linear.txt", sep=' ', index=False)

def kappa(alpha_a,d):
    ration = alpha_a/d
    if(0 <= ration <= 1):
        return 0.3
    else:
        return -1.15*np.exp(1-ration)+1.45
    
def Lambda(alpha_a,d):
    ration = alpha_a/d
    if(0 <= ration <= 1):
        return 1/0.3
    else:
        return 1/(-1.15*np.exp(1-ration)+1.45)

def q(alpha_a,d):
    ration = alpha_a/d
    if(0 <= ration <= 1):
        return 4/3
    else:
        return (1/3)*np.exp(1-ration)+1
    
# def r_properties_dataframe(N, dim, alpha_a, alpha_g):
#     if(alpha_g==str(0.0)):
#         pass
#     else:
#         # Directory with all samples
#         path_d = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/gml/"
#         # dataframe with all samples
#         new_file = "/properties_set_r.txt"

#         # Check if directory exist
#         conditional_ = os.path.exists(path_d)
#         if(conditional_ == True):
#             pass
#         else:
#             print("data doesn't exist, run code in c++ to gen data")

#         # Check if file exist
#         check_file = os.path.isfile(path_d+new_file)

#         # Open all files path in directory .csv
#         all_files = glob.glob(os.path.join(path_d,"*.gz"))
#         # If file exist, open
#         if(check_file == True):
#             df = pd.read_csv(path_d+new_file,sep=" ")
#             #filter_list to check if files are in dataframe
#             filter_list = str(df["#cod_file"].values) 
#             num_samples = len(df)
#             for file in all_files:
#                 # Check if file are in dataframe
#                 conditional = os.path.basename(file)[4:-7] in filter_list
#                 # Make nothing if True conditional
#                 if(conditional==True):
#                     pass
#                 # Add new elements in dataframe
#                 else:
#                     # load node properties
#                     node = {"id": [],
#                     "position":[],
#                     "degree": []}
                    
#                     # load edge properties
#                     edge = {"connections": [],
#                             "distance": []}
                    
#                     with gzip.open(file) as file_in:
#                         String = file_in.readlines()
#                         Lines = [i.decode('utf-8') for i in String]
#                         for i in range(len(Lines)):
#                             if(Lines[i]=='node\n'):
#                                 node["id"].append(int(Lines[i+2][4:-2]))
#                                 node["position"].append([float(Lines[i+6][2:-1]),float(Lines[i+7][2:-1]),float(Lines[i+8][2:-1])])
#                                 if(Lines[i+9][0]=='q'):
#                                     node["degree"].append(int(Lines[i+10][7:-1]))
#                                 else:
#                                     node["degree"].append(int(Lines[i+9][7:-1]))
#                             elif(Lines[i]=="edge\n"):
#                                 edge["connections"].append([int(Lines[i+2][8:-2]),int(Lines[i+3][8:-2])])
#                                 edge["distance"].append(float(Lines[i+4][9:-1]))
                    
#                     D = np.array(node["degree"])
#                     getcontext().prec = 50  # Set precision to 50 decimal places
#                     Ter_1 = Decimal(int(sum(D)))
#                     Ter_3 = Decimal(int(np.dot(D,D)))
#                     Ter_4 = Decimal(int(sum(D**3)))
                    
#                     G = nx.from_edgelist(edge["connections"])
#                     Ter_2 = 0
                    
#                     for j in G.edges():
#                         d_s = G.degree[j[0]]
#                         d_t = G.degree[j[1]]
#                         Ter_2 += d_s*d_t 
                    
#                     Ter_2 = Decimal(Ter_2)
                    
#                     getcontext().prec = 10  # Set precision to 50 decimal places
                    
#                     r = Decimal((Ter_1*Ter_2-Ter_3*2)/(Ter_1*Ter_4-Ter_3**2))
#                     df.loc[num_samples,"#ass_coeff"] = r
#                     df.loc[num_samples,"#cod_file"] = os.path.basename(file)[4:-7]
#                     num_samples += 1
#             # Save new dataframe update
#             df["#cod_file"] = df["#cod_file"].astype(int)
#             df.to_csv(path_d+new_file,sep=' ',index=False)

#         # Else, create it
#         else:
#             ass_coeff = []
#             cod_file = []
#             # Open all files path in directory .csv
            
#             for file in all_files:
#                 node = {"id": [],
#                     "position":[],
#                     "degree": []}
#                 edge = {"connections": [],
#                         "distance": []}
#                 with gzip.open(file) as file_in:
#                     String = file_in.readlines()
#                     Lines = [i.decode('utf-8') for i in String]
#                     for i in range(len(Lines)):
#                         if(Lines[i]=='node\n'):
#                             node["id"].append(int(Lines[i+2][4:-2]))
#                             node["position"].append([float(Lines[i+6][2:-1]),float(Lines[i+7][2:-1]),float(Lines[i+8][2:-1])])
#                             if(Lines[i+9][0]=='q'):
#                                 node["degree"].append(int(Lines[i+10][7:-1]))
#                             else:
#                                 node["degree"].append(int(Lines[i+9][7:-1]))
#                         elif(Lines[i]=="edge\n"):
#                             edge["connections"].append([int(Lines[i+2][8:-2]),int(Lines[i+3][8:-2])])
#                             edge["distance"].append(float(Lines[i+4][9:-1]))
#                 D = np.array(node["degree"])
#                 getcontext().prec = 50  # Set precision to 50 decimal places
#                 Ter_1 = Decimal(int(sum(D)))
#                 Ter_3 = Decimal(int(np.dot(D,D)))
#                 Ter_4 = Decimal(int(sum(D**3)))
#                 G = nx.from_edgelist(edge["connections"])
#                 Ter_2 = 0
#                 for j in G.edges():
#                     d_s = G.degree[j[0]]
#                     d_t = G.degree[j[1]]
#                     Ter_2 += d_s*d_t 
#                 Ter_2 = Decimal(Ter_2)
#                 getcontext().prec = 10  # Set precision to 50 decimal places
#                 r = Decimal((Ter_1*Ter_2-Ter_3*2)/(Ter_1*Ter_4-Ter_3**2))
#                 ass_coeff.append(r)
#                 cod_file.append(os.path.basename(file)[4:-7])
#             df = pd.DataFrame(data={"#ass_coeff":ass_coeff,"#cod_file":cod_file})
#             df.to_csv(path_d+new_file,sep=' ',index=False)


def filter_N_properties(alpha_filter, properties):
    # All index where alpha_a in all_alpha_a dataframe
    all_alpha_a = [properties.iloc[i,0] in alpha_filter for i in range(len(properties))]
    # Select index with alpha_a values
    index = [i for i in range(len(all_alpha_a)) if all_alpha_a[i]==True]
    # Values of properties
    N_values = [int(i[2:]) for i in properties.columns.values.tolist()[1:8]]

    properties_values = []
    err_properties_path = []
    for j in range(len(index)):
        properties_values.append(properties.iloc[index[j]][1:8].values)
        err_properties_path.append(properties.iloc[index[j]][8:].values)
    return N_values, properties_values, err_properties_path

def filter_N_linear_regression(alpha_filter, properties):
    # All index where alpha_a in all_alpha_a dataframe
    all_alpha_a = [properties.iloc[i,0] in alpha_filter for i in range(len(properties))]
    # Select index with alpha_a values
    index = [i for i in range(len(all_alpha_a)) if all_alpha_a[i]==True]
    # Values of properties
    N_values = [int(i[2:]) for i in properties.columns.values.tolist()[1:8]]

    properties_values = []
    err_properties_path = []
    for j in range(len(index)):
        properties_values.append(properties.iloc[index[j]][1:8].values)
        err_properties_path.append(properties.iloc[index[j]][8:].values)
    return N_values, properties_values, err_properties_path


# Create datarframe with alpha_g fixed and alpha_a variable or the other way around (N fixed)
def create_all_surface(N,dim,alpha_a,alpha_g,alpha_g_variable):

    path = f"../../data/surface/N_{N}/dim_{dim}"  
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    if(alpha_g_variable==True):
        mean_values = {"#alpha_g":[],"#short_mean":[],"#diamater_mean":[],
                       "#ass_coeff_mean":[],"#short_err":[],"#diamater_err":[],
                       "#ass_coeff_err":[],"#n_samples":[]}
        for i in alpha_g:
            if(i==str(0.0)):
                pass
            else:
                df = pd.read_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}alpha_g{i:.2f}/prop/properties_set.txt", sep=' ')
                mean_values["#alpha_g"].append(round(i, 2))
                
                mean_values["#short_mean"].append(df["#short_path"].mean())
                mean_values["#diamater_mean"].append(df["#diamater"].mean())
                mean_values["#ass_coeff_mean"].append(df["#ass_coeff"].mean())
                
                mean_values["#short_err"].append(df["#short_path"].sem())
                mean_values["#diamater_err"].append(df["#diamater"].sem())
                mean_values["#ass_coeff_err"].append(df["#ass_coeff"].sem())

                mean_values["#n_samples"].append(len(df["#diamater"]))
        
        df_all = pd.DataFrame(data=mean_values)
        sorted_df = df_all.sort_values(by='#alpha_g', key=lambda col: col.astype(float))  # Sort by converting to float
        sorted_df.to_csv(path + f"/all_alpha_a_{alpha_a:.2f}.txt", sep = ' ', index = False, mode="w+")
        
    else:       
        mean_values = {"#alpha_a":[],"#short_mean":[],"#diamater_mean":[],
                "#ass_coeff_mean":[],"#short_err":[],"#diamater_err":[],
                "#ass_coeff_err":[],"#n_samples":[]}

        for i in alpha_a:
            df = pd.read_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{i:.2f}alpha_g{alpha_g:.2f}/prop/properties_set.txt", sep=' ')
            mean_values["#alpha_a"].append(round(i, 2))
            
            mean_values["#short_mean"].append(df["#short_path"].mean())
            mean_values["#diamater_mean"].append(df["#diamater"].mean())
            mean_values["#ass_coeff_mean"].append(df["#ass_coeff"].mean())
            
            mean_values["#short_err"].append(df["#short_path"].sem())
            mean_values["#diamater_err"].append(df["#diamater"].sem())
            mean_values["#ass_coeff_err"].append(df["#ass_coeff"].sem())

            mean_values["#n_samples"].append(len(df["#diamater"]))
        
        df_all = pd.DataFrame(data=mean_values)
        sorted_df = df_all.sort_values(by='#alpha_a', key=lambda col: col.astype(float))  # Sort by converting to float
        sorted_df.to_csv(path + f"/all_alpha_g_{alpha_g:.2f}.txt", sep = ' ', index=False, mode="w+")

def fixing_data(N, dim, alpha_a, alpha_g):
    filen = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/properties_set.txt"

    # Lendo o conteúdo do arquivo
    with open(filen, "r") as file:
        data_lines = file.readlines()
        header_size = len(data_lines[0])
        
    if(header_size == 33):
        print("data ok")
        pass

    else:
        print('fixing data')
        # Remover o cabeçalho duplicado e limpar as linhas
        cleaned_data = []

        # Processar cada linha para substituir espaços por vírgulas, exceto na linha do cabeçalho
        for line in data_lines[1:]:  # Ignorando o cabeçalho duplicado
            line = line.strip()  # Remover espaços em branco no início e no fim
                
            if " " in line:
                # Substituir espaços por vírgulas de forma consistente
                cleaned_data.append(",".join(line.split()))
            else:
                cleaned_data.append(line)

        # Reunir os dados processados em uma string com quebra de linha para cada amostra
        cleaned_data = "\n".join(cleaned_data)

        # Lendo os dados limpos no DataFrame

        df = pd.read_csv(StringIO(cleaned_data), header=None, names=['#short_path', '#diameter', '#ass_coeff'])
        df.to_csv(filen, sep=',', index=False)


def format_file(N, dim):
    for n in N:
        for d in dim:        
            all_combinations_ag =  extract_alpha_values(n, d)
            for i in range(len(all_combinations_ag)):
                filepath = f"../../data/N_{n}/dim_{d}/alpha_a_{all_combinations_ag[i][0]:.2f}_alpha_g_{all_combinations_ag[i][1]:.2f}/properties_set.txt"
                print(n, d, round(all_combinations_ag[i][0], 2), round(all_combinations_ag[i][1], 2))
                    # Verifica se file é realmente uma string (caminho do arquivo) e não um arquivo aberto
                if not isinstance(filepath, str):
                    raise TypeError("O argumento 'file' deve ser uma string representando o caminho do arquivo.")
                
                # Abre o arquivo para leitura
                with open(filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                formatted_lines = []
                for line in lines:
                    # Verifica se a linha já é separada por vírgulas
                    if ',' in line:
                        formatted_lines.append(line.strip())  # Remove espaços extras no final da linha
                    else:
                        # Substitui múltiplos espaços em branco por uma única vírgula
                        formatted_line = ','.join(line.split())
                        formatted_lines.append(formatted_line)
                
                # Escreve o conteúdo formatado de volta no arquivo
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write('\n'.join(formatted_lines))

def remove_cod_file_column(N, dim, alpha_a, alpha_g):
    file_path =  f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/properties_set.txt"
    # Verifica se o arquivo existe
    if not os.path.exists(file_path):
        print(f"O arquivo {file_path} não existe.")
        return
    
    # Carrega o arquivo CSV
    df = pd.read_csv(file_path, delimiter=',')

    # Verifica se a coluna '#cod_file' existe
    if '#cod_file' in df.columns:
        # Remove a coluna '#cod_file'
        df = df.drop(columns=['#cod_file'])
        # Salva o arquivo novamente, sem a coluna '#cod_file'
        df.to_csv(file_path, index=False)
        print(f"Coluna '#cod_file' removida do arquivo {file_path}.")
    else:
        # Caso a coluna não exista, não faz nada
        print(f"A coluna '#cod_file' não existe no arquivo {file_path}. Nenhuma ação realizada.")
        pass

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator

def find_intersection(x, y):
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:  # Troca de sinal entre pontos consecutivos
            x1, x2 = x[i], x[i + 1]
            y1, y2 = y[i], y[i + 1]
            
            x_intercept = x1 - (y1 * (x2 - x1)) / (y2 - y1)
            y_intercept = 0  # Pois estamos buscando a interseção com y=0
            
            return x_intercept, y_intercept
    return None, None