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
# N = [2**12, 2**13, 2**14, 2**15, 2**16, 2**17]
# NumSamples = [10000, 5000, 2500, 1000, 500, 100]
N = [10**5]
NumSamples = [29]
# alpha_a = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
# alpha_g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

alpha_a = [2.0, 3.0, 4.0, 5.0]
alpha_g = [1.0, 2.0, 3.0, 4.0, 5.0]

alpha_ag_f = 2.0

m0 = 2
dim = [1,2,3,4]

run_mode=2 # Just Network
parms = {"N":[], "dim":[], "alpha_a":[], "alpha_g":[], "m0":[]}

#for n in N:

for i in range(len(N)):
    for d in dim:
        for aa in alpha_a:
            FunctionsFile.JsonGenerate(N[i], aa, alpha_ag_f, d, m0, run_mode)
            parms["N"].append(N[i])
            parms["dim"].append(d)
            parms["alpha_a"].append(aa)
            parms["alpha_g"].append(alpha_ag_f)
            parms["m0"].append(m0)
        for ag in alpha_g:
            FunctionsFile.JsonGenerate(N[i], alpha_ag_f, ag, d, m0, run_mode)
            parms["N"].append(N[i])
            parms["dim"].append(d)
            parms["alpha_a"].append(alpha_ag_f)
            parms["alpha_g"].append(ag)
            parms["m0"].append(m0)
    FunctionsFile.multithread_pc(N[i], NumSamples[i])
    FunctionsFile.permission_run(N[i])

df = pd.DataFrame(data=parms)
df.to_csv("parameters.csv",sep=",")

#for j in range(len(N)):
#FunctionsFile.multithread_pc(N, N_s)
#FunctionsFile.permission_run(N)
