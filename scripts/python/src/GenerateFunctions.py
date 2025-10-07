import numpy as np
import glob
import os
import json
from pathlib import Path

def ScriptGenerate(N, alpha_a, alpha_g, dim, NumSamples, m0):
    """
        N: number of nodes,\n
        alpha_a: expoential of distances between nodes\n
        alpha_g: expoential of power law distances\n
        dim: dimension of imerse network\n
        NumSamples: number of samples\n
        m0: number of connections of each new nodes\n
    """
    filename = f"N{N}_a{alpha_a:.2f}_g{alpha_g:.2f}_d{dim}.sh"

    a = "#!/bin/bash\n"
    b = f"#SBATCH -J a{alpha_a:.1f}_g{alpha_g:.1f}_d{dim}\n"
    d = "#SBATCH --ntasks=1\n"

    g = "x=0\n"
    h = f"n_samples={NumSamples}\n"
    i = "while [ $x -le $n_samples ]\n"
    j = "do\n\t"
    m = f"srun ../build/exe1 ../parms/N{N}_a{alpha_a:.2f}_g{alpha_g:.2f}_d{dim}_m0_{m0}.json\n\t"
    q = "x=$(( $x + 1 ))\n"
    v = "done"
    
    newpath = "../multithread/"
    
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    list_for_loop = [a,b,d,g,h,i,j,m,q,v]
    l = open(newpath + filename, "w") # argument w: write if don't exist file

    for k in list_for_loop:
        l.write(k)
    l.close()

def JsonGenerate(N : int, alpha_a : float, alpha_g : float, dim : int, m0 : int, run_mode : int):
    """
        N: number of nodes,\n
        alpha_a: expoential of distances between nodes\n
        alpha_g: expoential of power law distances\n
        dim: dimension of imerse network\n
        m0: number of connections of each new nodes\n
        run_mode: Decides what will be calculated\n
        props = 1 - just network\n
        props = 2 - just properties\n
        props = 3 - network and properties\n
    """
    
    filename = f"N{N}_a{alpha_a:.2f}_g{alpha_g:.2f}_d{dim}_m0_{m0}.json"
    outdir = Path(f"../../parms")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    
    if run_mode not in {1, 2, 3}: raise ValueError("run_mode must be 1 (props), 2 (network) or 3 (both).")
    
    data = {
        "comment": "use seed= -1 for random seed",
        "num_vertices": N,
        "alpha_a": float(f"{alpha_a:.6f}"),
        "alpha_g": float(f"{alpha_g:.6f}"),
        "r_min": 1,
        "r_max": 10000000,
        "dim": dim,
        "seed": -1,
        "m0": m0,
        "run_mode": int(run_mode)
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def text_terminal():
    newpath = "../../data/"
    
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    os.chdir(newpath)
    
    filename = "../scripts/run_all_script.txt"
    
    path = "../scripts/multithread"
    all_files = glob.glob(os.path.join(path,"*.sh"))

    l = open(filename, "w") # argument w: write if don't exist file
    a = 0
    for file in all_files:
        a += 1
        if(a==len(all_files)):
            file_ = f"sbatch\t {file}"
        else:
            file_ = f"sbatch\t {file} \t && \t"
        l.write(file_)

    l.close()
