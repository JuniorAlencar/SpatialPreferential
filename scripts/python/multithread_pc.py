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
N_s = 600
#N = [7500, 30000]
#N_s = [7000, 100]
dimensions = [1,2,3,4]

alpha_ag_f = 2.0
#alpha_a_v = [2.0, 9.0]
alpha_g_v = [1.5, 2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
# alpha_A_75 = [[3.1, 3.3, 3.5, 3.7, 3.9],[6.6, 6.8, 7.0, 7.2, 7.4],
#               [10.8, 11.0, 11.2, 11.4, 11.6], [12.1, 12.3, 12.5, 12.7, 12.9]]

# alpha_A_30 = [[2.1, 2.3, 2.5, 2.7, 2.9],[5.2, 5.4, 5.6, 5.8, 6.0],
#               [8.5, 8.7, 8.9, 9.1, 9.3], [10.3, 10.5, 10.7, 10.9, 11.1]]


# #n, dim, alpha_a, alpha_g

parms = {"N":[], "dim":[], "alpha_a":[], "alpha_g":[]}

# for n in N:
#     for i in range(len(dimensions)):
#         alpha_a_v_7 = alpha_A_75[i]
#         for aa in alpha_a_v_7:
#             FunctionsFile.JsonGenerate(n, aa, alpha_ag_f, dimensions[i])
#             parms["N"].append(n)
#             parms["dim"].append(dimensions[i])
#             parms["alpha_a"].append(aa)
#             parms["alpha_g"].append(alpha_ag_f)
        
#         alpha_a_v_3 = alpha_A_30[i]
#         for aa in alpha_a_v_3:
#             FunctionsFile.JsonGenerate(n, aa, alpha_ag_f, dimensions[i])
#             parms["N"].append(n)
#             parms["dim"].append(dimensions[i])
#             parms["alpha_a"].append(aa)
#             parms["alpha_g"].append(alpha_ag_f)

for dim in dimensions:
    for ag in alpha_g_v:
        FunctionsFile.JsonGenerate(N, alpha_ag_f, ag, dim)
        parms["N"].append(N)
        parms["dim"].append(dim)
        parms["alpha_a"].append(alpha_ag_f)
        parms["alpha_g"].append(ag)

df = pd.DataFrame(data=parms)
df.to_csv("parameters.csv",sep=",")

#for j in range(len(N)):
FunctionsFile.multithread_pc(N, N_s)
FunctionsFile.permission_run(N)