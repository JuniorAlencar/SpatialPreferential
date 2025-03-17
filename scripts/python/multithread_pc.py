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
alpha_g_f = 2.0
alpha_a_f = 2.0
# alpha_a, dim
parms_alpha_a = [(1.2,1), (1.4,1), (1.6,1), (1.8, 1),(2.2, 1), (2.4, 1),(2.6, 1),(2.8, 1),(1.2,2), (1.4,2), (1.6,2), (1.8, 2),(2.2, 2), (2.4, 2),(2.6, 2),(2.8, 2),(3.2, 3), (3.4, 3),(3.6, 3),(3.8, 3),(3.2, 4), (3.4, 4),(3.6, 4),(3.8, 4),(4.2, 4), (4.4, 4),(4.6, 4),(4.8, 4)]

for p in parms_alpha_a:
	alpha_a, dim = p[0], p[1]
	FunctionsFile.JsonGenerate(N, alpha_a, alpha_g_f, dim)
#for d in dim:
	#for aa in alpha_a_v:
#		FunctionsFile.JsonGenerate(N, aa, alpha_g_f, d)
	#for ag in alpha_g_v:
#		FunctionsFile.JsonGenerate(N, alpha_a_f, ag, d)
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
