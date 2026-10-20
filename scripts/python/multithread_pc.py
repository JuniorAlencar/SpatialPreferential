import src.MultithreadPC_Functions as FunctionsFile
N = 10**5
dim = 2
alpha_a = [i * dim for i in range(9)]
alpha_g = [1.0, 3.0, 4.0, 5.0, 6.0]
m0 = 4
NumSamples = 10
run_mode = 1
compute_dijkstra = True
compute_spearman = True

for aA in alpha_a:
    for aG in alpha_g:
        FunctionsFile.JsonGenerate(N, aA, aG, dim, m0, run_mode,
                                   compute_dijkstra, compute_spearman)

launcher = FunctionsFile.multithread_pc(N, NumSamples)
print(f"JSONs gerados em parms_pc_{N}; executor: {launcher}")
