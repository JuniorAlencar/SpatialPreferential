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
dimensions = [1,2,3,4]

alpha_ag_f = 2.0
#alpha_a_v = [2.0, 9.0]
alpha_g_v = [1.5, 2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]

# for dim in dimensions:
#     for aa in alpha_a_v:
#         FunctionsFile.JsonGenerate(N, aa, alpha_ag_f, dim)

for dim in dimensions:
    for ag in alpha_g_v:
        FunctionsFile.JsonGenerate(N, alpha_ag_f, ag, dim)

FunctionsFile.multithread_pc(N, N_s)
FunctionsFile.permission_run(N)