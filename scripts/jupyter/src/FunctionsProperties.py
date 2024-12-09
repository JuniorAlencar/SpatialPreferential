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

# All combinations (alpha_a, alpha_g) folders
def extract_alpha_values(N, dim):
    # Define the directory path
    folder_name = f'../../data/N_{N}/dim_{dim}/'

    # Define the pattern to match filenames
    pattern = re.compile(r'alpha_a_(\d+(\.\d+)?)_alpha_g_(\d+(\.\d+)?)')

    # Lists to store extracted combined values
    combine_values = []

    # Iterate over files in the directory
    for filename in os.listdir(folder_name):
        match = pattern.search(filename)
        if match:
            combine_values.append([float(match.group(1)), float(match.group(3))])

    return combine_values


# IMPORTANT! READ
# the code as it is, deletes the samples from the computer after execution. if you don't want this to happen, 
# comment out the line os.remove(File)
# Create file with all samples 
# def all_properties_file(N, dim, alpha_a, alpha_g):
#     # Diretório onde os arquivos estão localizados
#     path_d = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/prop"
#     path_save = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}"
#     print(f"N = {N}, dim = {dim}, alpha_a = {alpha_a}, alpha_g = {alpha_g}")
#     # Arquivos a serem atualizados
#     properties_file = os.path.join(path_save, "properties_set.txt")
#     filenames_file = os.path.join(path_save, "filenames.txt")
    
#     # Verificar se o diretório 'prop' existe
#     if not os.path.exists(path_d):
#         print(f"O diretório {path_d} não existe. Nada a ser feito.")
#         return
    
#     # Obter todos os arquivos CSV na pasta prop
#     all_files = glob.glob(os.path.join(path_d, "*.csv"))
    
#     # Se não houver arquivos na pasta prop, nada é feito
#     if not all_files:
#         print(f"A pasta {path_d} está vazia. Nada a ser feito.")
#         return
    
#     # Checar se o arquivo filenames.txt existe, caso contrário criar um
#     if os.path.exists(filenames_file):
#         with open(filenames_file, 'r') as f:
#             filenames_set = set(f.read().splitlines())  # Ler todos os arquivos já processados
#     else:
#         filenames_set = set()
    
#     # Se o arquivo properties_set.txt existir, carregar o dataframe, caso contrário criar um novo
#     if os.path.exists(properties_file):
#         df = pd.read_csv(properties_file, sep=' ')
#     else:
#         df = pd.DataFrame(columns=["#short_path", "#diamater", "#ass_coeff", "#cod_file"])
    
#     # Variável para rastrear se houve atualizações
#     updated = False
    
#     # Iterar sobre todos os arquivos CSV e verificar se já foram processados
#     for file in all_files:
#         filename = os.path.basename(file)
        
#         # Se o arquivo já foi processado, ignorar
#         if filename in filenames_set:
#             continue
        
#         # Se o arquivo ainda não foi processado, ler os dados e atualizar o DataFrame
#         new_data = pd.read_csv(file)
#         df = df.append({
#             "#short_path": new_data["#mean shortest path"].values[0],
#             "#diamater": new_data["# diamater"].values[0],
#             "#ass_coeff": new_data["#assortativity coefficient"].values[0],
#             "#cod_file": filename
#         }, ignore_index=True)
        
#         # Adicionar o nome do arquivo ao conjunto de arquivos processados
#         filenames_set.add(filename)
#         updated = True  # Indicar que houve atualizações
#         os.remove(file)
    
#     # Se houver atualizações, salvar os arquivos atualizados
#     if updated:
#         df.to_csv(properties_file, sep=' ', index=False)
#         with open(filenames_file, 'w') as f:
#             f.write("\n".join(sorted(filenames_set)))  # Escrever os nomes dos arquivos processados
#         print(f"Arquivos {properties_file} e {filenames_file} atualizados com sucesso.")
#     else:
#         print("Nenhuma atualização necessária. Todos os arquivos já estavam processados.")
def all_properties_file(N, dim, alpha_a, alpha_g):
    # Diretório onde os arquivos estão localizados
    path_d = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/prop"
    path_save = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}"
    print(f"N = {N}, dim = {dim}, alpha_a = {alpha_a}, alpha_g = {alpha_g}")
    # Arquivos a serem atualizados
    properties_file = os.path.join(path_save, "properties_set.txt")
    filenames_file = os.path.join(path_save, "filenames.txt")
    
    # Verificar se o diretório 'prop' existe
    if not os.path.exists(path_d):
        print(f"O diretório {path_d} não existe. Nada a ser feito.")
        return
    
    # Obter todos os arquivos CSV na pasta prop
    all_files = glob.glob(os.path.join(path_d, "*.csv"))
    
    # Se não houver arquivos na pasta prop, nada é feito
    if not all_files:
        print(f"A pasta {path_d} está vazia. Nada a ser feito.")
        return
    
    # Checar se o arquivo filenames.txt existe, caso contrário criar um
    if os.path.exists(filenames_file):
        with open(filenames_file, 'r') as f:
            filenames_set = set(f.read().splitlines())  # Ler todos os arquivos já processados
    else:
        filenames_set = set()
    
    # Se o arquivo properties_set.txt existir, carregar o dataframe, caso contrário criar um novo
    if os.path.exists(properties_file):
        df = pd.read_csv(properties_file, sep=',')
    else:
        df = pd.DataFrame(columns=["#short_path", "#diamater", "#ass_coeff"])
    
    # Variável para rastrear se houve atualizações
    updated = False
    new_rows = []  # Armazenar novas linhas para adicionar ao dataframe
    
    # Iterar sobre todos os arquivos CSV e verificar se já foram processados
    for file in all_files:
        filename = os.path.basename(file)
        
        # Se o arquivo já foi processado, ignorar
        if filename in filenames_set:
            continue
        
        # Se o arquivo ainda não foi processado, ler os dados e adicionar ao DataFrame
        new_data = pd.read_csv(file)
        new_row = {
            "#short_path": new_data["#mean shortest path"].values[0],
            "#diamater": new_data["# diamater"].values[0],
            "#ass_coeff": new_data["#assortativity coefficient"].values[0]
        }
        new_rows.append(new_row)
        
        # Adicionar o nome do arquivo ao conjunto de arquivos processados
        filenames_set.add(filename)
        updated = True  # Indicar que houve atualizações
        os.remove(file)  # Opcional: remover o arquivo após processamento
    
    # Se houver atualizações, salvar os arquivos atualizados
    if updated:
        # Adicionar as novas linhas ao dataframe
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        
        # Salvar o dataframe atualizado
        df.to_csv(properties_file, sep=',', index=False)
        
        # Atualizar o arquivo filenames.txt
        with open(filenames_file, 'w') as f:
            f.write("\n".join(sorted(filenames_set)))  # Escrever os nomes dos arquivos processados
        
        print(f"Arquivos {properties_file} e {filenames_file} atualizados com sucesso.")
    else:
        print("Nenhuma atualização necessária. Todos os arquivos já estavam processados.")

# Join all files in one dataframe
def all_data(N, dim):
    N_lst = []
    dim_lst = []
    alpha_a_lst = []
    alpha_g_lst = []
    N_samples_lst = []
    short_lst = []
    short_err_lst = []
    short_std_lst = []
    diameter_lst = []
    diameter_err_lst = []
    diameter_std_lst = []
    ass_coeff_lst = []
    ass_coeff_err_lst = []
    ass_coeff_std_lst = []

    #print(f"N={N}, dim = {dim}, alpha_a = {alpha_a}, alpha_g = {alpha_g}")
    for n in N:
        for d in dim:
            
            all_combinations_ag =  extract_alpha_values(n, d)
            for i in range(len(all_combinations_ag)):
                file = f"../../data/N_{n}/dim_{d}/alpha_a_{all_combinations_ag[i][0]}_alpha_g_{all_combinations_ag[i][1]}/properties_set.txt"
                
                #if file no exist, create it
                if not os.path.isfile(file):
                    print("file not exist, running function all_properties")
                    all_properties_file(n, d, all_combinations_ag[i][0], all_combinations_ag[i][1])
                
                try:
                    df = pd.read_csv(file, sep=',')
                    
                    if(len(df)==0):
                        pass
                    else:
                        # Number of nodes
                        N_lst.append(n)
                        dim_lst.append(d)
                        
                        # Alpha_a and Alpha_g values
                        alpha_a_lst.append(all_combinations_ag[i][0])
                        alpha_g_lst.append(all_combinations_ag[i][1])
                        
                        # Number of samples
                        N_samples_lst.append(df['#short_path'].count())
                        
                        # Short_mean and Short_erro
                        short_lst.append(df['#short_path'].mean())
                        short_err_lst.append(df['#short_path'].sem())
                        short_std_lst.append(df['#short_path'].std())
                        
                        # Diameter_mean and diameter erro
                        diameter_lst.append(df[df.columns[1]].mean().tolist())
                        diameter_err_lst.append(df[df.columns[1]].sem().tolist())
                        diameter_std_lst.append(df[df.columns[1]].std().tolist())
                        
                        # Diameter_mean and diameter erro
                        ass_coeff_lst.append(df['#ass_coeff'].mean())
                        ass_coeff_err_lst.append(df['#ass_coeff'].sem())
                        ass_coeff_std_lst.append(df['#ass_coeff'].std())
                except:
                    print("data not found")

    data_all = {"N":N_lst, "dim":dim_lst, "alpha_a":alpha_a_lst, "alpha_g":alpha_g_lst, 
                "N_samples":N_samples_lst, "short_mean":short_lst, "short_err":short_err_lst, 
                "short_std":short_std_lst,"diameter_mean":diameter_lst, "diameter_err":diameter_err_lst, 
                "diameter_std":diameter_std_lst, "ass_coeff_mean":ass_coeff_lst, 
                "ass_coeff_err":ass_coeff_err_lst, "ass_coeff_std":ass_coeff_std_lst}
    
    df_all = pd.DataFrame(data=data_all)
    df_all.to_csv(f"../../data/all_data.txt",sep=' ',index=False)

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
                    df_dim_alpha_a = df_dim[(df_dim["alpha_g"] == 2) & (df_dim["alpha_a"] == alpha)]
                    
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


def filter_N_properties(alpha_filter,properties):
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

def filter_N_linear_regression(alpha_filter,properties):
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
                df = pd.read_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}alpha_g{i}/prop/properties_set.txt", sep=' ')
                mean_values["#alpha_g"].append(i)
                
                mean_values["#short_mean"].append(df["#short_path"].mean())
                mean_values["#diamater_mean"].append(df["#diamater"].mean())
                mean_values["#ass_coeff_mean"].append(df["#ass_coeff"].mean())
                
                mean_values["#short_err"].append(df["#short_path"].sem())
                mean_values["#diamater_err"].append(df["#diamater"].sem())
                mean_values["#ass_coeff_err"].append(df["#ass_coeff"].sem())

                mean_values["#n_samples"].append(len(df["#diamater"]))
        
        df_all = pd.DataFrame(data=mean_values)
        sorted_df = df_all.sort_values(by='#alpha_g', key=lambda col: col.astype(float))  # Sort by converting to float
        sorted_df.to_csv(path + f"/all_alpha_a_{alpha_a}.txt", sep = ' ', index = False, mode="w+")
        
    else:       
        mean_values = {"#alpha_a":[],"#short_mean":[],"#diamater_mean":[],
                "#ass_coeff_mean":[],"#short_err":[],"#diamater_err":[],
                "#ass_coeff_err":[],"#n_samples":[]}

        for i in alpha_a:
            df = pd.read_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{i}alpha_g{alpha_g}/prop/properties_set.txt", sep=' ')
            mean_values["#alpha_a"].append(i)
            
            mean_values["#short_mean"].append(df["#short_path"].mean())
            mean_values["#diamater_mean"].append(df["#diamater"].mean())
            mean_values["#ass_coeff_mean"].append(df["#ass_coeff"].mean())
            
            mean_values["#short_err"].append(df["#short_path"].sem())
            mean_values["#diamater_err"].append(df["#diamater"].sem())
            mean_values["#ass_coeff_err"].append(df["#ass_coeff"].sem())

            mean_values["#n_samples"].append(len(df["#diamater"]))
        
        df_all = pd.DataFrame(data=mean_values)
        sorted_df = df_all.sort_values(by='#alpha_a', key=lambda col: col.astype(float))  # Sort by converting to float
        sorted_df.to_csv(path + f"/all_alpha_g_{alpha_g}.txt", sep = ' ', index=False, mode="w+")

def fixing_data(N, dim, alpha_a, alpha_g):
    filen = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/properties_set.txt"

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
                filepath = f"../../data/N_{n}/dim_{d}/alpha_a_{all_combinations_ag[i][0]}_alpha_g_{all_combinations_ag[i][1]}/properties_set.txt"
                print(n, d, all_combinations_ag[i][0], all_combinations_ag[i][1])
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
    file_path =  f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/properties_set.txt"
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