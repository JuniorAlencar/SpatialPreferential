import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd
import os

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

N = 400000
N_s = 3
alpha_ag_f = 2.0
dimensions = [1,2,3,4]
centers_120 = [2.27, 4.80, 7.50, 9.87]
step = 0.1
n_side = 3  # número de valores para cada lado

alpha_A_120 = [
    [float(f"{x:.2f}") for x in np.linspace(c - n_side*step, c + n_side*step, 2*n_side + 1)]
    for c in centers_120
]

for d in range(len(dimensions)):
    
    alpha_a_v_120 = alpha_A_120[d]
    for alpha_a in alpha_a_v_120:
        FunctionsFile.JsonGenerate(N, alpha_a, alpha_ag_f, dimensions[d])
        FunctionsFile.ScriptGenerate(N, alpha_a, alpha_ag_f , dimensions[d], N_s)

FunctionsFile.text_terminal()