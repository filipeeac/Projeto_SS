import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import sys

# O arquivo que deve ser lido deve possuir um símbolo por linha.
# Mais especificamente, os símbolos devem estar no conjunto S4qam
# Os símbolos são complexos e a parte real do símbolo corresponde
# a aceleração no eixo x e a parte imaginária corresponde a aceleração
# do eixo y

def validar_dados(data: np.ndarray, Ns: int, tol: float = 1e-6) -> Tuple[bool, np.ndarray]:
    """
    Função que verifica se os simbolos de um array pertencem ao conjunto S4qam
    com precisão arbitrária.
    Caso sim, gera um Tuple com Verdadeiro, caso contrário, gera Falso junto a 
    um array contendo os valores inválidos.
    """
    h = np.sqrt(2)/2
    S4qam = np.array([h+1j*h, -h+1j*h, -h-1j*h, h-1j*h], dtype = np.complex128)
    diffs = np.abs(data.reshape(-1,1) - S4qam.reshape(1,-1))
    diff_min = diffs.min(axis=1)
    val_invalidos = diff_min > tol
    ret = not val_invalidos.any()
    if len(data) != Ns:
        print("Número esperado de símbolos inválido...")
        print(f'Quantidade esperada: {Ns}... Quantidade recebida: {len(data)}')
        ret = False
    return (ret, val_invalidos)

def simular(data: np.ndarray, alpha: float, V0: Tuple[float,float] = (0.0,0.0), T: float = 1.0, pos0: Tuple[float,float] = (0.0,0.0)):
    """
    Gera dois arrays contendo os valores de posição e velocidade do drone a partir de um array de simbolos.
    """
    n = data.size
    acel = alpha*np.column_stack((data.real, data.imag))

    V = np.zeros((n+1,2), dtype = float)
    V[0, :] = np.array(V0, dtype=float)
    V[1:, :] = V[0, :] + np.cumsum(acel, axis=0)

    pos = np.zeros((n+1,2), dtype = float)
    pos[0, :] = np.array(pos0, dtype = float)
    pos[1:, :] = pos[0, :] + T*np.cumsum(V[:-1,:], axis=0)

    return pos, V

def plotar(pos: np.ndarray, V: np.ndarray, show_quiver: bool = True,
                    quiver_every: int = 5, save_as: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(pos[:, 0], pos[:, 1], '-o', markersize=4, linewidth=1)
    ax.set_xlabel('x (posição)')
    ax.set_ylabel('y (posição)')
    ax.set_title('Trajetória (S4QAM → acelerações)')
    ax.grid(True)
    ax.axis('equal')

    if show_quiver:
        # desenha vetores de velocidade em alguns pontos
        idx = np.arange(0, pos.shape[0], quiver_every)
        ax.quiver(pos[idx, 0], pos[idx, 1], V[idx, 0], V[idx, 1], angles='xy', scale_units='xy', scale=1)

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
        print(f"Figura salva em {save_as}")
    plt.show()

if __name__ == '__main__':
    path = './data/simbolos.csv'
    try:
        data = np.loadtxt(path, skiprows = 0, dtype = np.complex128)
    except Exception as e:
        print(f'Deu ruim nobre. Erro: {e}')
        sys.exit(1)


    Ns = 100 # Número de símbolos esperado
    ok, val_errados = validar_dados(data, Ns=Ns)
    if not ok:
        print('Simbolos invalidos... valores errados: ')
        print(np.where(val_errados)[0].tolist())
        sys.exit(1)

    V0 = (2,2)
    T = 1.0
    alpha = 0.5

    pos, V = simular(data,alpha=alpha,T=T,V0=V0)
    print(pos[-1])
    plotar(pos, V)
