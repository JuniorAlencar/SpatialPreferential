import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd
import os

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

N = 10**5
N_s = 600
dimensions = [1,2,3,4]

alpha_ag_f = 2.0
#alpha_a_v = [2.0, 9.0]
alpha_a_v = [1.5, 2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]

for alpha_a in alpha_a_v:
    for d in dimensions:
        FunctionsFile.JsonGenerate(N, alpha_a, alpha_ag_f, d)
        FunctionsFile.ScriptGenerate(N, alpha_a,alpha_ag_f , d, N_s)

FunctionsFile.text_terminal()