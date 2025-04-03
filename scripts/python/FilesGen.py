import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd
import os

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

df = pd.read_csv("parameters.csv", sep=',')
x,y = 143, 293
for index, row in df.iloc[x:y].iterrows():
	N, N_s, dim, alpha_a = int(row["N"]), int(row["N_s"]), int(row["dim"]), row["alpha_a"]
	FunctionsFile.JsonGenerate(N, alpha_a, 2.0, dim)
	FunctionsFile.ScriptGenerate(N, alpha_a, 2.0, dim, N_s)    
    		                
FunctionsFile.text_terminal()