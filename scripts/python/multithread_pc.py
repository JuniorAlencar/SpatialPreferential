import src.MultithreadPC_Functions as FunctionsFile
import numpy as np
import glob
import os
import pandas as pd

#multithread_pc(N,num_samples)
#N: number of nodes;
# return .sh file to run parallel in PC in scripts folder
#--------------------------------------------------------
#JsonGenerate(N, alpha_a,alpha_g,dim)
#multithread_pc(N,NumSamples)
#alpha_a: parameter to control preferential attrachment
#alpha_g: parameter to control random power law
#dim: dimension
#N: Number of nodes;
#return: set of .json file with above parameters 

N = 10**5
N_s = 50
dim = [1,2,3,4]
alpha_a = 2.0
alpha_g = np.linspace(1.00, 5.00, 20)

for i in range(len(alpha_g)):
    for d in dim:
        FunctionsFile.JsonGenerate(N, alpha_a, alpha_g[i], d)
    
    #FunctionsFile.JsonGenerate(N, alpha_a[i], float(row["alpha_g"]), float(row["dim"]))        
# df = pd.read_csv("run_multi.txt", delimiter=' ')
# df_n = df[df["N"]==N]

# for _, row in df_n.iterrows():
#     FunctionsFile.JsonGenerate(int(row["N"]), float(row["alpha_a"]), float(row["alpha_g"]), float(row["dim"]))
#     FunctionsFile.JsonGenerate(int(row["N"]), float(row["alpha_a"]), float(row["alpha_g"]), float(row["dim"]))    
# for i in range(len(N)):
#     FunctionsFile.multithread_pc(N[i], N_samples[i])
#     FunctionsFile.permission_run(N[i])

# for i in range(len(alpha_a)):
#     FunctionsFile.JsonGenerate(N, alpha_a[i], alpha_g, dim)

FunctionsFile.multithread_pc(N, N_s)
FunctionsFile.permission_run(N)