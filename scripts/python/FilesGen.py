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
alpha_a = [0.0, 1.0, 2.0, 3.0, 4.0]
alpha_g = [1.0, 2.0, 3.0, 4.0]
#alpha_ag_f = 2.0
NumSamples = [40]
m0 = 2
dim = [2,3,4]
run_mode = 2


for d in dim:
    for ag in alpha_g:
        for aa in alpha_a:
            FunctionsFile.JsonGenerate(N[0], aa, ag, d, m0, run_mode)
            FunctionsFile.ScriptGenerate(N[0], aa, ag, d, NumSamples[0], m0)

FunctionsFile.text_terminal()
