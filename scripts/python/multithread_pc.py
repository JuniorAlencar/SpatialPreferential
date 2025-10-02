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


# N = 10**5
# N_s = 600
N = 100000
N_s = 300
dimensions = [1,2,3,4]
alpha_ag_f = 2.0
alpha_a = [1.0, 2.0, 3.0, 4.0, 5.0]
alpha_g = [2.0, 3.0, 4.0, 5.0]
m0 = 2
# centers_30 = [2.5, 5.50, 8.70, 10.7]
# step = 0.1
# n_side = 3  # número de valores para cada lado

# alpha_A_30 = [
#     [float(f"{x:.2f}") for x in np.linspace(c - n_side*step, c + n_side*step, 2*n_side + 1)]
#     for c in centers_30
# ]

# centers_50 = [2.40, 5.3, 8.3, 10.3]
# step = 0.1
# n_side = 3  # número de valores para cada lado

# alpha_A_50 = [
#     [float(f"{x:.2f}") for x in np.linspace(c - n_side*step, c + n_side*step, 2*n_side + 1)]
#     for c in centers_50
# ]

# # #n, dim, alpha_a, alpha_g

parms = {"N":[], "dim":[], "alpha_a":[], "alpha_g":[], "m0":[]}

#for n in N:
for i in range(len(dimensions)):
    for aa in alpha_a:
        FunctionsFile.JsonGenerate(N, aa, alpha_ag_f, dimensions[i], m0)
        parms["N"].append(N)
        parms["dim"].append(dimensions[i])
        parms["alpha_a"].append(aa)
        parms["alpha_g"].append(alpha_ag_f)
        parms["m0"].append(m0)
    
    for ag in alpha_g:
        FunctionsFile.JsonGenerate(N, alpha_ag_f, ag, dimensions[i], m0)
        parms["N"].append(N)
        parms["dim"].append(dimensions[i])
        parms["alpha_a"].append(alpha_ag_f)
        parms["alpha_g"].append(ag)
        parms["m0"].append(m0)

df = pd.DataFrame(data=parms)
df.to_csv("parameters.csv",sep=",")

#for j in range(len(N)):
FunctionsFile.multithread_pc(N, N_s)
FunctionsFile.permission_run(N)