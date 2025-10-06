import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd
import os

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

N = [5000, 10000, 20000]
