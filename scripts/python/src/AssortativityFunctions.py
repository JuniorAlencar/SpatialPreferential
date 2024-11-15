import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  
import glob
import sys
from collections import Counter
from scipy.stats import binned_statistic
from collections import OrderedDict
import networkx as nx
from scipy import stats
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import gzip
#from collections import Counter
from IPython.display import clear_output

# run assortativity calculate with parameters from terminal
def assortativity_calculate():
    # read args
    args = sys.argv[1:]

    # if number of arguments was bigger, return
    if len(args) != 4:
        print("Please, enter with arguments: N, dim, alpha_a, alpha_g")
        return

    # converted
    N = int(args[0])
    dim = int(args[1])
    alpha_a = float(args[2])
    alpha_g = float(args[3])
    
    path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}"

    # If file in all_files, create check_folder to move files
    if not os.path.exists(path + "/all_files"):
        os.makedirs(path + "/all_files")
    else:
        pass

    all_files = glob.glob(os.path.join(path + "/gml","*.gml.gz"))

    if(len(all_files)==0):
        all_files = glob.glob(os.path.join(path + "/gml/check","*.gml.gz"))

    count_files = 0

    degree_list = []
    knn_list = []

    print(f"N = {N}, dim = {dim}, alpha_a = {alpha_a}, alpha_g = {alpha_g}")

    for file in all_files:
        print(f"{len(all_files)} num_files, {len(all_files)-count_files} files remaining")
        
        a = []
        b = []
        with open(file, 'rb') as file_in:
            # decompress gzip
            with gzip.GzipFile(fileobj=file_in, mode='rb') as gzip_file:
                for line in gzip_file:
                    # decode file
                    line = line.decode('utf-8')
                    if(line[:6]=="target"):
                        a.append(line[8:-2])
                    elif(line[:6]=="source"):
                        b.append(line[8:-2])

        connections_list = [(i,j) for i,j in zip(a,b)]
        G = nx.from_edgelist(connections_list)

        k_ij = []

        for i in range(len(G.nodes())):
            degreekij = [G.degree(j) for j in [n for n in G.neighbors(str(i))]]
            Ter = sum(degreekij)/G.degree(str(i))
            k_ij.append(Ter)

        datak = pd.DataFrame(data={"k":[j for i,j in G.degree()]})
        dataknn = pd.DataFrame(data={"knn":k_ij})
        
        degree_list.append(datak)
        knn_list.append(dataknn)
        
        count_files += 1

    k_list = pd.concat(degree_list)
    knn_ = pd.concat(knn_list)
    new_data_frame = pd.DataFrame(data={"k":k_list["k"].values,"knn":knn_["knn"].values})
    new_data_frame
    data_final = new_data_frame.groupby("k").mean()
    file_save = pd.DataFrame(data={"k":data_final.index.values,"knn":data_final["knn"].values})
    file_save.to_csv(path+"/all_files/" + "assortativity.csv",index=False,mode="w")
    clear_output(wait=True)  # Set wait=True if you want to clear the output without scrolling the notebook


# def assortativity(N, dim, alpha_a, alpha_g):
#     pathOriginal = f'../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}'
#     pathSave = f'../../data/N_{N}/Assortativity/dim_{dim}/'
    
#     if not os.path.exists(pathSave):
#         os.makedirs(pathSave)

#     all_files = glob.glob(os.path.join(pathOriginal,"*.csv"))
#     degree_list = []
#     knn_list = []

#     for file in all_files:
#         connections = pd.read_csv(f"{pathOriginal}/connections/connections_{os.path.basename(file)[5:-4]}.csv.gz")
#         degree = pd.read_csv(f"{pathOriginal}/degree/degree_{os.path.basename(file)[5:-4]}.csv.gz")["k"].values
#         con_ = [(connections["#Node1"].values[i],connections["#Node2"].values[i]) for i in range(len(connections["#Node1"].values))]
#         G=nx.from_edgelist(con_)
#         k_ij = []
#         for i in range(len(degree)):
#             degreekij = [degree[j] for j in [n for n in G.neighbors(i)]]
#             Ter = sum(degreekij)/degree[i]
#             k_ij.append(Ter)
#         # dk = {"k":degree}
#         # dknn = {"knn":k_ij}
        
#         datak = pd.DataFrame(data={"k":degree})
#         dataknn = pd.DataFrame(data={"knn":k_ij})
#         degree_list.append(datak)
#         knn_list.append(dataknn)

#     k_list = pd.concat(degree_list)
#     knn_ = pd.concat(knn_list)
#     new_data_frame = pd.DataFrame(data={"k":k_list["k"].values,"knn":knn_["knn"].values})
#     data_final = new_data_frame.groupby("k").mean()
#     file_save = pd.DataFrame(data={"k":data_final.index.values,"knn":data_final["knn"].values})
#     file_save.to_csv(pathSave+f"alpha_a_{alpha_a}_alpha_g_{alpha_g}.csv",index=False)

def drop_zeros(a_list):
    return [i for i in a_list if i>0]

def binning_2d(x, y,num_bins,alpha_a,alpha_g,alphaA):
    if(alphaA==True):
        alpha_g = 2.0
        x_bin, y_bin = [[] for _ in range(len(alpha_a))],[[] for _ in range(len(alpha_a))]
        for i in range(len(alpha_a)):
            counter_dict = Counter(x[i])
            
            max_x = np.log10(max(list(counter_dict.keys())))
            max_y = np.log10(max(list(counter_dict.values())))
            max_base = max([max_x,max_y])
            min_x = np.log10(min(drop_zeros(list(counter_dict.keys()))))

            bins_aux = np.logspace(min_x,max_base,num=num_bins)
            bins = [(bins_aux[q],bins_aux[q+1]) for q in range(len(bins_aux)-1)]

            box_bins_k = [[] for _ in range(len(bins))]
            box_bins_knn = [[] for _ in range(len(bins))]

            for l in range(len(x[i])):
                for m in range(len(bins)):
                    if(bins[m][0]<=x[i][l]<=bins[m][1]):
                        box_bins_k[m].append(x[i][l])
                        box_bins_knn[m].append(y[i][l])
            
            index_list = []
            for idk,element in enumerate(box_bins_k):
                if(len(element)!=0):
                    index_list.append(idk)

            box_bins_k = [box_bins_k[q] for q in index_list]
            box_bins_knn = [box_bins_knn[q] for q in index_list]

            x_bin[i] = [np.mean(box_bins_k[q]) for q in range(len(box_bins_k))]
            y_bin[i] = [np.mean(box_bins_knn[q]) for q in range(len(box_bins_knn))]
        return x_bin,y_bin
    else:
        alpha_a = 2.0
        x_bin, y_bin = [[] for _ in range(len(alpha_g))],[[] for _ in range(len(alpha_g))]
        for i in range(len(alpha_g)):
            counter_dict = Counter(x[i])
            max_x = np.log10(max(list(counter_dict.keys())))
            max_y = np.log10(max(list(counter_dict.values())))
            max_base = max([max_x,max_y])
            min_x = np.log10(min(drop_zeros(list(counter_dict.keys()))))

            bins_aux = np.logspace(min_x,max_base,num=num_bins)
            bins = [(bins_aux[i],bins_aux[i+1]) for i in range(len(bins_aux)-1)]

            box_bins_k = [[] for _ in range(len(bins))]
            box_bins_knn = [[] for _ in range(len(bins))]

            for l in range(len(x[i])):
                for m in range(len(bins)):
                    #c = k[i] in bins[j]
                    if(bins[m][0]<=x[i][l]<=bins[m][1]):
                        box_bins_k[m].append(x[i][l])
                        box_bins_knn[m].append(y[i][l])
            
            index_list = []
            for idk,element in enumerate(box_bins_k):
                if(len(element)!=0):
                    index_list.append(idk)

            box_bins_k = [box_bins_k[q] for q in index_list]
            box_bins_knn = [box_bins_knn[q] for q in index_list]

            x_bin[i] = [np.mean(box_bins_k[q]) for q in range(len(box_bins_k))]
            y_bin[i] = [np.mean(box_bins_knn[q]) for q in range(len(box_bins_knn))]
        return x_bin,y_bin

def q(alpha_a,d):
    ration = alpha_a/d
    if(0 <= ration <= 1):
        return 4/3
    elif(ration > 1):
        return (1/3)*np.exp(1-ration)+1

def Lambda(alpha_a,d):
    ration = alpha_a/d
    if(0 <= ration <= 1):
        return 1/0.3
    else:
        return 1/(-1.15*np.exp(1-ration)+1.45)

def r_assortativity(N, dim, alpha_a, alpha_g, alphaA):
    
    if(alphaA==True):
        alpha_g = 2.0
        k, knn = [[] for _ in range(len(alpha_a))],[[] for _ in range(len(alpha_a))]
        for i in range(len(alpha_a)):
            path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/all_files/assortativity.csv"
            k[i], knn[i] = pd.read_csv(path,delimiter=',')["k"]. values,pd.read_csv(path,delimiter=',')["knn"].values
        return k, knn
    else:
        alpha_a = 1.0
        k, knn = [[] for _ in range(len(alpha_g))],[[] for _ in range(len(alpha_g))]
        for i in range(len(alpha_g)):
            path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/all_files/assortativity.csv"
            k[i], knn[i] = pd.read_csv(path,delimiter=',')["k"]. values,pd.read_csv(path,delimiter=',')["knn"].values
        return k, knn

# run multithread assortativity calcualte
def assortativity_multithread(dim, alpha_a, alpha_g):
    #filename = f"multithread_assortativity.sh"
    if(len(alpha_a)==1):
        filename = f"multithread_assortativity_alpha_a.sh"
    elif(len(alpha_g)==1):
        filename = f"multithread_assortativity_alpha_g.sh"
    
    a = "#!/bin/bash\n\n"
    
    b = "# Define uma função que contêm o código para rodar em paralelo\n"
    
    c = "run_code() {\n\t"
    d = f"time python run_assortativity.py 100000 $1\n"
    e = "}\n"
    f = "# Exportar a função usando o módulo Parallel\n"
    g = "export -f run_code\n\n"
    
    list_of_arguments = []
    
    for i in dim:
        if(len(alpha_a)==1):
            for k in alpha_g:
                list_of_arguments.append(f"{i} {alpha_a[0]} {k}")
        elif(len(alpha_g)==1):
            for j in alpha_a:
                list_of_arguments.append(f"{i} {j} {alpha_g[0]}")
    list_of_arguments = str(list_of_arguments)
    list_of_arguments = list_of_arguments.replace(',', '')

    h = f"arguments=(" 
    i = list_of_arguments[1:-1] + ")\n"
    j = "parallel run_code :::\t" +  """ "${arguments[@]}"  """ "\n\t"

    list_for_loop = [a,b,c,d,e,g,h,i,j]
    
    l = open(filename, "w+") # argument w+: write if don't exist file and overwrite if exist

    for k in list_for_loop:
        l.write(k)
    l.close()

def permission_run(alpha_a, alpha_g):
    if(len(alpha_a) == 1):
        os.system(f"chmod 700 multithread_assortativity_alpha_a.sh")
    
    elif(len(alpha_g) == 1):
        os.system(f"chmod 700 multithread_assortativity_alpha_g.sh")
