import src.MultithreadPC_Functions as FunctionsFile

import numpy as np

N = 10**5
dim = 2
alpha_a_MID = dim
alpha_a_MAX = 2*dim
alpha_a = np.concatenate([
    np.linspace(0, alpha_a_MID, 10),
    np.linspace(alpha_a_MID, alpha_a_MAX, 10)[1:]
])

alpha_g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
m0 = 2
NumSamples = 50
run_mode = 3
compute_dijkstra = False
compute_spearman = False
# False keeps the default BFS from 1,000 random source sites. Set to True
# only when the exhaustive, all-source BFS is also required.
compute_bfs_all = False

for aA in alpha_a:
    for aG in alpha_g:
        FunctionsFile.JsonGenerate(N, aA, aG, dim, m0, run_mode,
                                   compute_dijkstra, compute_spearman,
                                   compute_bfs_all)

launcher = FunctionsFile.multithread_pc(N, NumSamples)
print(f"JSONs gerados em parms_pc_{N}; executor: {launcher}")
