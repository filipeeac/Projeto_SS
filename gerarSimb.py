#!/usr/bin/env python3
"""
Gera um CSV com n linhas contendo símbolos S4QAM.
Saída: um ficheiro CSV com uma única coluna ("complex") contendo strings
no formato: 0.707107+0.707107j (real e imag com 6 casas decimais).

Exemplos:
    python generate_s4qam_numpy.py -n 100 -o s4qam.csv
    python generate_s4qam_numpy.py -n 16 --mode cycle --seed 42 -o out.csv
"""

import argparse
import numpy as np

def build_symbols():
    h = np.sqrt(2.0) / 2.0
    # vetor de quatro símbolos S1..S4
    return np.array([h + 1j*h, -h + 1j*h, -h - 1j*h, h - 1j*h], dtype=np.complex128)

def generate_symbols(n, mode="random", seed=None):
    syms = build_symbols()
    rng = np.random.default_rng(seed)

    if mode == "cycle":
        # repetir os símbolos em ciclo
        idx = np.arange(n) % syms.size
        chosen = syms[idx]
    else:
        # seleção aleatória uniforme
        chosen = rng.choice(syms, size=n)

    # formatar: "real+imagj" com 6 casas decimais
    formatted = [f"{z.real:.6f}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.6f}j" for z in chosen]
    return np.array(formatted, dtype=str)

def write_single_column_csv(filename, data, header="complex"):
    # np.savetxt escreve rapidamente, comments='' evita o '#' antes do header
    np.savetxt(filename, data, fmt="%s", header=header, comments='')

def parse_args():
    p = argparse.ArgumentParser(description="Gerador S4QAM (uma coluna: real+imagj)")
    p.add_argument("-n", "--num", required=True, type=int, help="Número de linhas (símbolos)")
    p.add_argument("-o", "--output", default="s4qam.csv", help="Ficheiro CSV de saída")
    p.add_argument("--mode", choices=["random", "cycle"], default="random",
                   help="Modo: 'random' (aleatório) ou 'cycle' (S1,S2,S3,S4 repetido)")
    p.add_argument("--seed", type=int, default=None, help="Semente RNG (opcional)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    arr = generate_symbols(args.num, mode=args.mode, seed=args.seed)
    write_single_column_csv(args.output, arr)
    print(f"Arquivo '{args.output}' criado com {args.num} linhas (uma coluna).")

