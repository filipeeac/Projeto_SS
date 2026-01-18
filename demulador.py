import matplotlib.pyplot as plt
import numpy as np

def obter_sinc(M: int,  w_c: float):
    """
    Gera um sinal no domínio do tempo na forma da função sinc
    """
    m = np.arange(-(M//2), (M//2)+1)
    h = np.sinc((w_c*m)/np.pi)*(w_c/np.pi)
    return h

def obter_filtro_freq(h: np.ndarray, nfft: int):
    """
    Obtém o espectro em frequência de um filtro h[n]
    """
    H = np.fft.fft(h, n=nfft)
    return H


def extrair_sim(H: np.ndarray, yRb: np.ndarray, yIb: np.ndarray, N: int, M: int) -> np.complex128:
    """
    Extrai o símbolo modulado de um sinal ruidoso
    """
    nfft = len(H)
    mid = M//2
    
    # Espectro de yR e yI do sinal
    YRb = np.fft.fft(yRb, n=nfft)
    YIb = np.fft.fft(yIb, n=nfft)

    # Multiplicação do espectro do sinal modulado pelo espectro do filtro
    # Isso equivale a fazer a convolução no domínio do tempo
    SR = H*YRb
    SI = H*YIb

    # O sinal filtrado no domínio do tempo
    # Esse sinal é aproximadamente uma constante, logo fazemos a média do sinal para conseguir
    # o valor da componente do nosso símbolo
    sR = np.fft.ifft(SR)
    sI = np.fft.ifft(SI)

    simR = np.sum(sR[mid: mid + N])/N
    simI = np.sum(sI[mid: mid + N])/N

    X = simR + 1j*simI
    return X


def sim_mais_proximo(z: np.complex128) -> np.complex128:
    h = np.sqrt(2)/2
    S4qam = np.array([(h+1j*h),(-h+1j*h),(-h-1j*h),(h-1j*h)])
    diff = np.abs(z-S4qam)
    idx = np.argmin(diff)
    return S4qam[idx]

N = 21 # Número de amostras por simbolo
fc = 1000 # frequência do sinal
fs = fc*21 # frequência de amostragem
Ts = 1/fs # step do dominio do tempo
Ns = 100 # numero de símbolos transmitidos
Nt = N*Ns # numero total de amostras de y[n]
n = np.arange(N)
t = n*Ts

S = np.zeros(Ns, dtype=np.complex128)

y = np.loadtxt('./data/rx_sgn_y.csv', dtype=np.complex128)


# Como y se dá por: x + z com x real, é possivel dizer que y.imag é puramente ruído
# Logo, y.real será x + z_r
y = y.real

yb = np.zeros((Ns, N))
for m in range(Ns):
    yb[m] = y[(21*m):(21*m+21)]

f_cutoff = fc/2 # Parâmetro ajustável
w_c = 2*np.pi*f_cutoff/fs
M = 201 # Parâmetro ajustável
h = obter_sinc(M, w_c)

nfft = int(2**np.ceil(np.log2(N+len(h)-1)))
H = obter_filtro_freq(h,nfft)

ybR = np.sqrt(2)*np.cos(2*np.pi*fc*t)*yb
ybI = -np.sqrt(2)*np.sin(2*np.pi*fc*t)*yb

freqs = np.fft.fftfreq(nfft, d = 1/fs)

S_raw = np.zeros(Ns, dtype=np.complex128)
for m in range(Ns):
    S_raw[m] = extrair_sim(H, ybR[m], ybI[m], N, M)
    S[m] = sim_mais_proximo(S_raw[m])


plt.figure(figsize=(6,6))
plt.scatter(np.real(S_raw), np.imag(S_raw), s=30, alpha=0.8)
plt.title("Constelação S4qam")
plt.xlabel('Parte Real (I)')
plt.ylabel('Parte Imaginária (Q)')
plt.axhline(0, color='k', linewidth=0.7)
plt.axvline(0, color='k', linewidth=0.7)
plt.grid()
plt.show()


np.savetxt('./data/simbolos.csv', S, fmt="%s")
