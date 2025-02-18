import numpy as np
import pandas as pd
import os
import gzip
import glob
import re
import networkx as nx
from decimal import Decimal, getcontext
from pandas.errors import EmptyDataError

# Return: dataframe with all samples
def all_properties_dataframe(N, dim, alpha_a, alpha_g):
    # Directory with all samples
    path_d = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}"
    # dataframe with all samples
    new_file = "/properties_set.txt"
    
    # Check if directory exist
    conditional_ = os.path.exists(path_d)
    if(conditional_ == True):
        pass
    else:
        return "data doesn't exist, run code in c++ to gen data"
    
    # Check if file exist
    check_file = os.path.isfile(path_d+new_file)
    
    # Open all files path in directory .csv
    all_files = glob.glob(os.path.join(path_d,"*.csv"))
    # If file exist, open
    if(check_file == True):
        df = pd.read_csv(path_d+new_file,sep=" ")
        data = df.iloc[:,1:]
        #filter_list to check if files are in dataframe
        df_names = pd.read_csv(f"{path_d}/filenames.txt")

        filter_list = str(df_names["filename"].values) 
        num_samples = len(data)
        for file in all_files:
            # Check if file are in dataframe
            conditional = os.path.basename(file)[5:-4] in filter_list
            # Make nothing if True conditional
            if(conditional==True):
                pass
            # Add new elements in dataframe
            else:
                new_data = pd.read_csv(file)
                print(file)
                df.loc[num_samples,"#short_path"] = new_data["#mean shortest path"].values[0]
                df.loc[num_samples,"#diamater"] = new_data["# diamater"].values[0]
                df.loc[num_samples,"#ass_coeff"] = new_data["#assortativity coefficient"].values[0]
                df_names.loc[num_samples,"filenames"] = os.path.basename(file)[5:-4]
                num_samples += 1
                
        # Save new dataframe update
        df.to_csv(path_d+new_file,sep=' ',index=False)
        df_names.to_csv(path_d+"/filenames.txt",sep=' ',index=False)

    # Else, create it
    else:
        df = pd.DataFrame(columns=["#short_path", "#diamater", "#ass_coeff"])
        df_names = pd.DataFrame(columns="filename")
        i = 0
        # Open all files path in directory .csv
        for file in all_files:
            train = pd.read_csv(file)
            df.loc[i,"#short_path"] = train["#mean shortest path"].values[0]
            df.loc[i,"#diamater"] = train["# diamater"].values[0]
            df.loc[i,"#ass_coeff"] = train["#assortativity coefficient"].values[0]
            df_names.loc[i,"filename"] = os.path.basename(file)[5:-4]
            i += 1
        df.to_csv(path_d+new_file,sep=' ',index=False)
        df_names.to_csv(path_d+"/filenames.txt",sep=' ',index=False)

def extract_alpha_values(folder_data):
    # Caminho inicial
#    base_path = "../../data_2"
    base_path = folder_data

    # Regex para capturar nvalue, dvalue, alpha_a:.2f (aavalue) e alpha_g:.2f (agvalue)
    pattern = r"N_(\d+)/dim_(\d+)/alpha_a_([\d]+\.\d{2})_alpha_g_([\d]+\.\d{2})"


    # Estrutura para armazenar as combinações encontradas
    combinations = set()

    # Percorrer todas as subpastas a partir de base_path
    for root, dirs, files in os.walk(base_path):
        match = re.search(pattern, root)
        if match:
            nvalue = int(match.group(1))  # nvalue como inteiro
            dvalue = int(match.group(2))  # dvalue como inteiro
            aavalue = float(match.group(3))  # alpha_a:.2f como float
            agvalue = float(match.group(4))  # alpha_g:.2f como float
            combinations.add((nvalue, dvalue, round(aavalue, 2), round(agvalue, 2)))
    return combinations

def all_data(folder_data):
    # Caminho inicial
    #    base_path = "../../data_2"
    base_path = folder_data

    # Regex para capturar nvalue, dvalue, alpha_a:.2f (aavalue) e alpha_g:.2f (agvalue)
    pattern = r"N_(\d+)/dim_(\d+)/alpha_a_([\d]+\.\d{2})_alpha_g_([\d]+\.\d{2})"

    # Estrutura para armazenar as combinações encontradas
    combinations = set()

    all_data = {"N":[], "dim":[], "alpha_a:.2f":[], "alpha_g:.2f":[], "N_samples":[],
                "short_mean":[], "short_err":[],"short_err_per":[],"diameter_mean":[],
                "diameter_err":[],"diameter_err_per":[],"ass_coeff_mean":[],"ass_coeff_err":[],
                "ass_coeff_err_per":[]}

    # Percorrer todas as subpastas a partir de base_path
    for root, dirs, files in os.walk(base_path):
        match = re.search(pattern, root)
        if match:
            nvalue = int(match.group(1))  # nvalue como inteiro
            dvalue = int(match.group(2))  # dvalue como inteiro
            aavalue = float(match.group(3))  # alpha_a:.2f como float
            agvalue = float(match.group(4))  # alpha_g:.2f como float
            file = f"../../data/N_{nvalue}/dim_{dvalue}/alpha_a_{aavalue:.2f}_alpha_g_{agvalue:.2f}/properties_set.txt"
            df = pd.read_csv(file, sep=' ')
            
            # add parameters to dictionary
            all_data["N"].append(nvalue)
            all_data["dim"].append(dvalue)
            all_data["alpha_a"].append(round(aavalue, 2))
            all_data["alpha_g"].append(round(agvalue, 2))
            all_data["N_samples"].append(len(df))

            # Add the statistical quantities
            # short_path
            all_data["short_mean"].append(df["#short_path"].mean())
            all_data["short_err"].append(df["#short_path"].sem())
            all_data["short_err_per"].append((df["#short_path"].sem() / df["#short_path"].mean())*100)
            # diameter
            all_data["diameter_mean"].append(df["#diamater"].mean())
            all_data["diameter_err"].append(df["#diamater"].sem())
            all_data["diameter_err_per"].append((df["#diamater"].sem() / df["#diamater"].mean())*100)
            # ass_coeff
            all_data["ass_coeff_mean"].append(df["#ass_coeff"].mean())
            all_data["ass_coeff_err"].append(df["#ass_coeff"].sem())
            all_data["ass_coeff_err_per"].append(abs((df["#ass_coeff"].sem() / df["#ass_coeff"].mean())*100))

    df_all = pd.DataFrame(data=all_data)
    df_all.to_csv("../../data/all_data.txt", sep=' ', index=False, mode="w+")

def all_properties_dataframe_2():
    base_dir = os.path.expanduser("../data")
    all_data = []

    # Regex para capturar N_value, dim_value, alpha_a_value e alpha_g_value
    pattern = re.compile(r'N_(\d+)/dim_(\d+)/alpha_a_([\d\.\-eE]+)/alpha_g_([\d\.\-eE]+)')

    # Percorre recursivamente todas as pastas
    for root, dirs, files in os.walk(base_dir):
        match = pattern.search(root)
        if match:
            # Extrai os valores de N, dim, alpha_a:.2f e alpha_g:.2f a partir do nome da pasta
            N_value = match.group(1)
            dim_value = match.group(2)
            alpha_a_value = match.group(3)
            alpha_g_value = match.group(4)

            # Diretório onde estão os arquivos de amostra
            path_d = os.path.join(root, "prop")
            new_file = "/properties_set.txt"

            # Verificar se o arquivo de propriedades já existe
            check_file = os.path.isfile(path_d + new_file)

            # Pegar todos os arquivos .csv no diretório
            all_files = glob.glob(os.path.join(path_d, "*.csv"))
            total_files = len(all_files)  # Total de arquivos a serem analisados
            analyzed_files = 0  # Contador de arquivos já analisados

            if check_file:
                # Se o arquivo já existe, abrir e carregar o dataframe
                df = pd.read_csv(path_d + new_file, sep=" ")

                # Gerar a lista de filtro a partir da coluna `#cod_file`
                filter_list = df["#cod_file"].astype(str).values

                # Iterar sobre os arquivos e verificar se estão no dataframe
                for i, file in enumerate(all_files):
                    conditional = os.path.basename(file)[5:-4] in filter_list
                    if conditional:
                        continue  # Ignora o arquivo se já está no dataframe
                    else:
                        # Carregar novas propriedades e adicioná-las ao dataframe
                        new_data = pd.read_csv(file)
                        df.loc[len(df), "#short_path"] = new_data["#mean shortest path"].values[0]
                        df.loc[len(df), "#diamater"] = new_data["# diamater"].values[0]
                        df.loc[len(df), "#ass_coeff"] = new_data["#assortativity coefficient"].values[0]
                        df.loc[len(df), "#cod_file"] = os.path.basename(file)[5:-4]

                    # Incrementa o número de arquivos analisados
                    analyzed_files += 1

                    # Exibe o progresso
                    print(f"Processando: N={N_value}, dim={dim_value}, alpha_a:.2f={alpha_a_value}, alpha_g:.2f={alpha_g_value}")
                    print(f"Total de arquivos: {total_files}, Analisados: {analyzed_files}, Restantes: {total_files - analyzed_files}")

                # Salvar o dataframe atualizado
                df.to_csv(path_d + new_file, sep=' ', index=False)
                all_data.append(df)
            else:
                print(f"Arquivo não existe em N={N_value}, dim={dim_value}, alpha_a:.2f={alpha_a_value}, alpha_g:.2f={alpha_g_value}. Criando novo arquivo...")
                # Se o arquivo não existe, criar um novo com os dados
                df = pd.DataFrame(columns=["#short_path", "#diamater", "#ass_coeff", "#cod_file"])
                for i, file in enumerate(all_files):
                    new_data = pd.read_csv(file)
                    df.loc[len(df), "#short_path"] = new_data["#mean shortest path"].values[0]
                    df.loc[len(df), "#diamater"] = new_data["# diamater"].values[0]
                    df.loc[len(df), "#ass_coeff"] = new_data["#assortativity coefficient"].values[0]
                    df.loc[len(df), "#cod_file"] = os.path.basename(file)[5:-4]

                    # Incrementa o número de arquivos analisados
                    analyzed_files += 1

                    # Exibe o progresso
                    print(f"Processando: N={N_value}, dim={dim_value}, alpha_a:.2f={alpha_a_value}, alpha_g:.2f={alpha_g_value}")
                    print(f"Total de arquivos: {total_files}, Analisados: {analyzed_files}, Restantes: {total_files - analyzed_files}")

                # Salvar o novo dataframe
                df.to_csv(path_d + new_file, sep=' ', index=False)
                all_data.append(df)

    # Concatena todos os dados em um único dataframe
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        return df_all
    else:
        print("Nenhum dado foi encontrado.")
        return pd.DataFrame()  # Retorna um DataFrame vazio se nenhum dado foi encontrado


# List all pair of (alpha_a:.2f,alpha_g:.2f) folders in (N,dim) folder
def list_all_folders(N,dim):
    directory = f"../../data/N_{N}/dim_{dim}/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    lst_folders = lst_folders[0]
    set_parms = [(lst_folders[i][8:11],lst_folders[i][20:]) for i in range(len(lst_folders))]
    
    return set_parms

def list_all_folders_for_alpha_fixed(N,dim,alpha_g_variable):
    directory = f"../../data/N_{N}/dim_{dim}/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    lst_folders = lst_folders[0]
    set_parms = [(lst_folders[i][8:11],lst_folders[i][20:]) for i in range(len(lst_folders))]

    if(alpha_g_variable==True):
        alpha_a = 1.0
        alpha_g_V = []
        for i in range(len(set_parms)):
            if(set_parms[i][0]==str(round(alpha_a, 2))):
                alpha_g_V.append(float(set_parms[i][1]))
        return alpha_g_V
    else:
        alpha_g = 2.0
        alpha_a_V = []
        for i in range(len(set_parms)):
            if(set_parms[i][1]==str(round(alpha_g,2))):
                alpha_a_V.append(float(set_parms[i][0]))
        return alpha_a_V

from pandas.errors import EmptyDataError
def all_properties_file(N, dim, alpha_a, alpha_g):
    # Diretório onde os arquivos estão localizados
    path_d = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/prop"
    path_save = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}"
    print(f"N = {N}, dim = {dim}, alpha_a:.2f = {alpha_a:.2f}, alpha_g:.2f = {alpha_g:.2f}")
    
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
        df = pd.read_csv(properties_file, sep=' ')
    else:
        df = pd.DataFrame(columns=["#short_path", "#diamater", "#ass_coeff"])
    
    # Variável para rastrear se houve atualizações
    updated = False
    new_rows = []  # Armazenar novas linhas para adicionar ao dataframe
    
    # Iterar sobre todos os arquivos CSV e verificar se já foram processados
    #block 1
    for file in all_files:
        filename = os.path.basename(file)    
        # Se o arquivo já foi processado, ignorar
        if filename in filenames_set:
            continue
        try:
            print(file)
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
            #os.remove(file)  # Opcional: remover o arquivo após processamento
        except EmptyDataError:
            print("erro")
            os.remove(file)
    # Se houver atualizações, salvar os arquivos atualizados
    if updated:
        # Adicionar as novas linhas ao dataframe
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        
        # Salvar o dataframe atualizado
        df.to_csv(properties_file, sep=' ', index=False)
        
        # Atualizar o arquivo filenames.txt
        with open(filenames_file, 'w') as f:
            f.write("\n".join(sorted(filenames_set)))  # Escrever os nomes dos arquivos processados
        
        print(f"Arquivos {properties_file} e {filenames_file} atualizados com sucesso.")
    else:
        print("Nenhuma atualização necessária. Todos os arquivos já estavam processados.")


def list_N_folders():
    directory = f"../../data/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    dif = lst_folders[0]
    diff_N = [i[2:] for i in dif]
    diff_N.sort(key=int)
    return diff_N

# Create datarframe with alpha_g:.2f fixed and alpha_a:.2f variable or the other way around (N fixed)
def shortest_path_dim_N(N, dim, alpha_g_variable):
    #directory = f"../../data/"
    #lst_folders = []
    #for root, dirs, files in os.walk(directory):
        #lst_folders.append(dirs)
    #dif = lst_folders[0]
    #diff_N = [i[2:] for i in dif]
    #diff_N.sort(key=int)
    diff_head_N = ["N_" + i for i in N]    
    diff_head_err = ["N_"+ i + "_err" for i in N]

    if(alpha_g_variable==True):
        diff_head_N = ["#alpha_g:.2f"] + diff_head_N
        header = diff_head_N + diff_head_err
        df_all = pd.DataFrame(columns=header)
        alpha_a = 1.00
        j = 0
        for i in N:
            if(i==str(0.0)):
                pass
            else:
                file = f"../../data/N_{i}/dim_{dim}/properties_all_alpha_a_1.0.txt"
                df = pd.read_csv(file, sep=' ')
                df_all["#alpha_g:.2f"] = df["#alpha_g:.2f"]
                df_all[f"{i}"] = df["#short_path"]
                df_all["#short_err"] = df["#short_err"]
            j += 1
        df_all.to_csv(f"../../data/short_N_dim_{dim}_{alpha_a:.2f}.txt",sep = ' ',index=False,mode="w")
    else:       
        diff_head_N = ["#alpha_a:.2f"] + diff_head_N
        header = diff_head_N + diff_head_err
        df_all = pd.DataFrame(columns=header)
        j = 0

        for i in N:
            file = f"../../data/N_{i}/dim_{dim}/properties_all_alpha_g_2.00.txt"
            df = pd.read_csv(file, sep=' ')
            df_all["#alpha_a:.2f"] = df["#alpha_a:.2f"]
            df_all[f"{i}"] = df["#short_path"]
            df_all["#short_err"] = df["#short_err"]

            j += 1
        df_all.to_csv(f"../../data/short_N_dim_{dim}_{2.00:.2f}.txt",sep = ' ',index=False,mode="w")

def linear_regression(X,Y,Error):
    # Calculate weighted means
    weighted_x_mean = np.sum(X / Error) / np.sum(1 / Error)
    weighted_y_mean = np.sum(Y / Error) / np.sum(1 / Error)

    # Calculate weighted covariance and variance
    weighted_covar = np.sum(X * Y / Error) / np.sum(1 / Error) - weighted_x_mean * weighted_y_mean
    weighted_x_var = np.sum(X ** 2 / Error) / np.sum(1 / Error) - weighted_x_mean ** 2

    # Calculate the slope (m) and intercept (b) of the linear regression line
    slope = weighted_covar / weighted_x_var
    intercept = weighted_y_mean - slope * weighted_x_mean
    
    y_pred = [slope*i+intercept for i in X]
#    return slope, intercept
    return y_pred

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
    

def r_properties_dataframe(N, dim, alpha_a, alpha_g):
    if(f"{alpha_g:.2f}"==str(0.00)):
        pass
    else:
        # Directory with all samples
        path_d = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/gml/"
        # dataframe with all samples
        new_file = "/properties_set_r.txt"

        # Check if directory exist
        conditional_ = os.path.exists(path_d)
        if(conditional_ == True):
            pass
        else:
            print("data doesn't exist, run code in c++ to gen data")

        # Check if file exist
        check_file = os.path.isfile(path_d+new_file)

        # Open all files path in directory .csv
        all_files = glob.glob(os.path.join(path_d,"*.gz"))
        # If file exist, open
        if(check_file == True):
            df = pd.read_csv(path_d+new_file,sep=" ")
            #filter_list to check if files are in dataframe
            filter_list = str(df["#cod_file"].values) 
            num_samples = len(df)
            for file in all_files:
                # Check if file are in dataframe
                conditional = os.path.basename(file)[4:-7] in filter_list
                # Make nothing if True conditional
                if(conditional==True):
                    pass
                # Add new elements in dataframe
                else:
                    # load node properties
                    node = {"id": [],
                    "position":[],
                    "degree": []}
                    
                    # load edge properties
                    edge = {"connections": [],
                            "distance": []}
                    
                    with gzip.open(file) as file_in:
                        String = file_in.readlines()
                        Lines = [i.decode('utf-8') for i in String]
                        for i in range(len(Lines)):
                            if(Lines[i]=='node\n'):
                                node["id"].append(int(Lines[i+2][4:-2]))
                                node["position"].append([float(Lines[i+6][2:-1]),float(Lines[i+7][2:-1]),float(Lines[i+8][2:-1])])
                                if(Lines[i+9][0]=='q'):
                                    node["degree"].append(int(Lines[i+10][7:-1]))
                                else:
                                    node["degree"].append(int(Lines[i+9][7:-1]))
                            elif(Lines[i]=="edge\n"):
                                edge["connections"].append([int(Lines[i+2][8:-2]),int(Lines[i+3][8:-2])])
                                edge["distance"].append(float(Lines[i+4][9:-1]))
                    
                    D = np.array(node["degree"])
                    getcontext().prec = 50  # Set precision to 50 decimal places
                    Ter_1 = Decimal(int(sum(D)))
                    Ter_3 = Decimal(int(np.dot(D,D)))
                    Ter_4 = Decimal(int(sum(D**3)))
                    
                    G = nx.from_edgelist(edge["connections"])
                    Ter_2 = 0
                    
                    for j in G.edges():
                        d_s = G.degree[j[0]]
                        d_t = G.degree[j[1]]
                        Ter_2 += d_s*d_t 
                    
                    Ter_2 = Decimal(Ter_2)
                    
                    getcontext().prec = 10  # Set precision to 50 decimal places
                    
                    r = Decimal((Ter_1*Ter_2-Ter_3*2)/(Ter_1*Ter_4-Ter_3**2))
                    df.loc[num_samples,"#ass_coeff"] = r
                    df.loc[num_samples,"#cod_file"] = os.path.basename(file)[4:-7]
                    num_samples += 1
            # Save new dataframe update
            df["#cod_file"] = df["#cod_file"].astype(int)
            df.to_csv(path_d+new_file,sep=' ',index=False)

        # Else, create it
        else:
            ass_coeff = []
            cod_file = []
            # Open all files path in directory .csv
            
            for file in all_files:
                node = {"id": [],
                    "position":[],
                    "degree": []}
                edge = {"connections": [],
                        "distance": []}
                with gzip.open(file) as file_in:
                    String = file_in.readlines()
                    Lines = [i.decode('utf-8') for i in String]
                    for i in range(len(Lines)):
                        if(Lines[i]=='node\n'):
                            node["id"].append(int(Lines[i+2][4:-2]))
                            node["position"].append([float(Lines[i+6][2:-1]),float(Lines[i+7][2:-1]),float(Lines[i+8][2:-1])])
                            if(Lines[i+9][0]=='q'):
                                node["degree"].append(int(Lines[i+10][7:-1]))
                            else:
                                node["degree"].append(int(Lines[i+9][7:-1]))
                        elif(Lines[i]=="edge\n"):
                            edge["connections"].append([int(Lines[i+2][8:-2]),int(Lines[i+3][8:-2])])
                            edge["distance"].append(float(Lines[i+4][9:-1]))
                D = np.array(node["degree"])
                getcontext().prec = 50  # Set precision to 50 decimal places
                Ter_1 = Decimal(int(sum(D)))
                Ter_3 = Decimal(int(np.dot(D,D)))
                Ter_4 = Decimal(int(sum(D**3)))
                G = nx.from_edgelist(edge["connections"])
                Ter_2 = 0
                for j in G.edges():
                    d_s = G.degree[j[0]]
                    d_t = G.degree[j[1]]
                    Ter_2 += d_s*d_t 
                Ter_2 = Decimal(Ter_2)
                getcontext().prec = 10  # Set precision to 50 decimal places
                r = Decimal((Ter_1*Ter_2-Ter_3*2)/(Ter_1*Ter_4-Ter_3**2))
                ass_coeff.append(r)
                cod_file.append(os.path.basename(file)[4:-7])
            df = pd.DataFrame(data={"#ass_coeff":ass_coeff,"#cod_file":cod_file})
            df.to_csv(path_d+new_file,sep=' ',index=False)
