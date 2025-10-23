import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd
import os

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

N = [2**12, 2**13, 2**14, 2**15, 2**16, 2**17]
alpha_a = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
alpha_g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
alpha_ag_f = 2.0
NumSamples = [900, 300, 200, 100, 50, 10]
m0 = 2
dim = [1,2,3,4]
run_mode = 1

for i in range(len(N)):
    for d in dim:
        for aa in alpha_a:
            FunctionsFile.JsonGenerate(N[i], aa, alpha_ag_f, d, m0, run_mode)
            FunctionsFile.ScriptGenerate(N[i], aa, alpha_ag_f, d, NumSamples[i], m0)
        for ag in alpha_g:
            FunctionsFile.JsonGenerate(N[i], alpha_ag_f, ag, d, m0, run_mode)
            FunctionsFile.ScriptGenerate(N[i], alpha_ag_f, ag, d, NumSamples[i], m0)

FunctionsFile.text_terminal()