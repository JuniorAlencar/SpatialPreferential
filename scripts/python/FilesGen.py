import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd
import os

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

N = [10**5]
dim = 2
alpha_a = [i*dim for i in range(9)]
alpha_g = [1.0, 2.0, 3.0, 4.0, 5.0]
NumSamples = [30]
m0 = 2
dim = [2]
run_mode = 1
compute_dijkstra = False
compute_spearman = False

for d in dim:
    for ag in alpha_g:
        for aa in alpha_a:
            FunctionsFile.JsonGenerate(N[0], aa, ag, d, m0, run_mode,
                                       compute_dijkstra, compute_spearman)
            FunctionsFile.ScriptGenerate(N[0], aa, ag, d, NumSamples[0], m0)

FunctionsFile.text_terminal()
