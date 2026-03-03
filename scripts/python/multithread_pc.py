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


N = [10**5]
alpha_a = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
alpha_g = [1.0, 2.0, 3.0, 4.0]
#alpha_ag_f = 2.0
NumSamples = [10]
m0 = 2
dim = [2]
run_mode = 2

#for n in N:

for d in dim:
    for ag in alpha_g:
        for aa in alpha_a:
            FunctionsFile.JsonGenerate(N[0], aa, ag, d, m0, run_mode)

FunctionsFile.multithread_pc(N[0], NumSamples[0])
FunctionsFile.permission_run(N[0])

#for j in range(len(N)):
#FunctionsFile.multithread_pc(N, N_s)
#FunctionsFile.permission_run(N)
