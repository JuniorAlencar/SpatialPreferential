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

df = pd.read_csv("parameters.csv", sep=',')
N_s = 350
N = 20000

for _, row in df.iterrows():
    dim, alpha_a = int(row["dim"]),float(row["alpha_a"])
    if (N==10000):
    	FunctionsFile.JsonGenerate(N, alpha_a, 2.0, dim)

FunctionsFile.multithread_pc(N, N_s)
FunctionsFile.permission_run(N)