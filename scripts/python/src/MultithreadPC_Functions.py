import numpy as np
import glob
import os
import pandas as pd
import re
import json
from pathlib import Path

def JsonGenerate(N: int, alpha_a : float, alpha_g : float, dim : int, m0 : int, run_mode : int):

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
    
    if run_mode not in {1, 2, 3}: raise ValueError("run_mode must be 1 (props), 2 (network) or 3 (both).")
    
    filename = f"N{N}_a{alpha_a:.2f}_g{alpha_g:.2f}_d{dim}_m0_{m0}.json"
    outdir = Path(f"../../parms_pc_{N}")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    
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
        "run_mode" : int(run_mode)
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def multithread_pc(N, NumSamples):
    filename = f"N_{N}_multithread_pc.sh"

    a = "#!/bin/bash\n\n"
    b = "# Define uma função que contêm o código para rodar em paralelo\n"
    c = "run_code() {\n\t"
    d = f"time ../build/exe1 ../parms_pc_{N}/$1\n"
    e = "}\n"
    e1 = """progress_bar() {\n \tlocal current=$1\n \tlocal total=$2\n \tlocal percent=$(( current * 100 / total))\n \tlocal filled=$(( percent * 50 / 100))\n \tlocal empty=$(( 50 - filled))\n"""
    e2 = """\tprintf "\\r[%-${filled}s%${empty}s] %d%% (%d/%d)" "#" "" "$percent" "$current" "$total" \n}\n\n"""
    g = "export -f run_code\n\n"

    # check number of files in specific folders
    df = pd.read_csv("parameters.csv", sep=',')
    df_n = df[df["N"]==N]
    counts = []
    for _, row in df_n.iterrows():
        dim, alpha_a, alpha_g = int(row["dim"]),float(row["alpha_a"]), float(row["alpha_g"])
        
        data_path = f"../../data_3/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/gml"
        counts.append(len(glob.glob(os.path.join(data_path, "*.gml.gz"))))
    
    # Se nenhuma pasta foi encontrada
    if not counts:
        print("[WARN] Nenhuma pasta 'gml/' encontrada. Continuando com o valor total.")
        n_to_generate = NumSamples
    else:
        nss_min = min(counts)
        print(f"[INFO] Mínimo de arquivos encontrados em uma pasta: {nss_min}")
        n_to_generate = max(0, NumSamples - nss_min)
        print(f"[INFO] Gerando script para os {n_to_generate} restantes de {NumSamples}.")

    if n_to_generate == 0:
        print("[INFO] Todos os arquivos já foram gerados. Nenhum script necessário.")
        return

    # Caminho para arquivos json de parâmetros
    path_d = f"../../parms_pc_{N}"
    all_jsons = glob.glob(os.path.join(path_d, "*.json"))
    arguments = [os.path.basename(f) for f in all_jsons]

    h = "arguments=("
    i = " ".join(arguments) + ")\n\n"

    j = "x=0\n"
    k = f"n_samples={n_to_generate}\n"
    l = "while [ $x -lt $n_samples ]\n"
    m = "do\n\t"
    n = 'parallel run_code ::: "${arguments[@]}"\n\t'
    o = "x=$(( $x + 1))\n"
    o1 = '\tprogress_bar "$x" "$n_samples"\n '
    p = "done"

    list_for_loop = [a, b, c, d, e, e1, e2, g, h, i, j, k, l, m, n, o, o1, p]

    with open("../" + filename, "w") as l:
        for k in list_for_loop:
            l.write(k)

def permission_run(N):
	os.system(f"chmod 700 ../N_{N}_multithread_pc.sh")
