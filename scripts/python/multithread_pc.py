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

N = [5000, 10000, 20000, 40000, 80000, 160000, 320000]
N_samples = [20000, 2000, 200, 60, 20, 8, 4]

df = pd.read_csv("run_multi.txt", delimiter=' ')

for _, row in df.iterrows():
    FunctionsFile.JsonGenerate(int(row["N"]), float(row["alpha_a"]), float(row["alpha_g"]), float(row["dim"]))
    FunctionsFile.JsonGenerate(int(row["N"]), float(row["alpha_a"]), float(row["alpha_g"]), float(row["dim"]))    
for i in range(len(N)):
    FunctionsFile.multithread_pc(N[i], N_samples[i])
    FunctionsFile.permission_run(N[i])


