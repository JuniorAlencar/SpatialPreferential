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


N = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]
alpha_a = 2.0
alpha_g = 2.0
m0 = 2
dim = 4
#alpha_ag_f = 2.0
NumSamples = [50, 30, 20, 10, 10, 5, 5, 5]

run_mode = 2

#for n in N:

for idx, n in enumerate(N):
    FunctionsFile.JsonGenerate(n, alpha_a, alpha_g, dim, m0, run_mode)

    FunctionsFile.multithread_pc(n, NumSamples[idx])
    FunctionsFile.permission_run(n)

#for j in range(len(N)):
#FunctionsFile.multithread_pc(N, N_s)
#FunctionsFile.permission_run(N)
