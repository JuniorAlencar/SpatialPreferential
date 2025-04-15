import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd
import os

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

alpha_a_N32 = [0.50, 1.25, 1.50, 1.75, 2.50, 3.25, 4.25, 4.75, 5.50, 5.75]
alpha_a_ALL = [11.25, 11.50, 11.75, 12.25, 12.50, 12.75, 13.25, 13.50, 13.75, 14.25, 14.50,14.75, 15.25, 15.50, 15.75]
N = [5000, 10000, 20000, 40000, 80000, 160000, 320000]	
N_s = [10000, 1600, 350, 250, 100, 15, 7]
dim = [1,2,3,4]
# for d in dim:
# 	for aa32 in alpha_a_N32:
# 		FunctionsFile.JsonGenerate(320000, aa32, 2.0, d)
# 		FunctionsFile.ScriptGenerate(320000, aa32, 2.0, d, 7)   
for n in range(len(N)):
	for d in dim:
		for aa in alpha_a_ALL:
			FunctionsFile.JsonGenerate(N[n], aa, 2.0, d)
			FunctionsFile.ScriptGenerate(N[n], aa, 2.0, d, N_s[n])

FunctionsFile.text_terminal()