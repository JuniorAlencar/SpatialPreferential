import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd
import os

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

N = [120000, 250000]
N_s = [10, 5]
alpha_ag_f = 2.0
dimensions = [1,2,3,4]
centers_120 = [2.34, 4.98, 7.84, 9.96]
step = 0.1
n_side = 3  # número de valores para cada lado

alpha_A_120 = [
    [float(f"{x:.2f}") for x in np.linspace(c - n_side*step, c + n_side*step, 2*n_side + 1)]
    for c in centers_120
]

centers_250 = [2.30, 4.86, 7.60, 9.77]
step = 0.1
n_side = 3  # número de valores para cada lado

alpha_A_250 = [
    [float(f"{x:.2f}") for x in np.linspace(c - n_side*step, c + n_side*step, 2*n_side + 1)]
    for c in centers_250
]


for d in range(len(dimensions)):
    
    alpha_a_v_120 = alpha_A_120[d]
    for alpha_a in alpha_a_v_120:
        FunctionsFile.JsonGenerate(N[0], alpha_a, alpha_ag_f, dimensions[d])
        FunctionsFile.ScriptGenerate(N[0], alpha_a, alpha_ag_f , dimensions[d], N_s[0])
    
    alpha_a_v_250 = alpha_A_250[d]
    for alpha_a in alpha_a_v_250:
        FunctionsFile.JsonGenerate(N[1], alpha_a, alpha_ag_f, dimensions[d])
        FunctionsFile.ScriptGenerate(N[1], alpha_a, alpha_ag_f , dimensions[d], N_s[1])

FunctionsFile.text_terminal()