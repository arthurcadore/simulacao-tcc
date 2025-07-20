import numpy as np
import matplotlib.pyplot as plt
import json
import os
import scienceplots
import komm 
from collections import defaultdict
import matplotlib.gridspec as gridspec
from formatter import Formatter

# Estilo science
plt.style.use('science')
plt.rcParams["figure.figsize"] = (16, 9)
plt.rc('font', size=16)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=22)
plt.rc('legend', frameon=True, edgecolor='black', facecolor='white', fancybox=True, fontsize=12)


class Modulator:
    def __init__(self, fc, fs):
        self.fc = fc
        self.fs = fs

    def modulate(self, i_signal, q_signal):
        """
        Aplica modulação IQ no domínio do tempo:
        s(t) = I(t) * cos(2πf_ct) - Q(t) * sin(2πf_ct)
        """
        n = len(i_signal)
        if len(q_signal) != n:
            raise ValueError("i_signal e q_signal devem ter o mesmo tamanho.")
        
        t = np.arange(n) / self.fs
        carrier_cos = np.cos(2 * np.pi * self.fc * t)
        carrier_sin = np.sin(2 * np.pi * self.fc * t)
        
        modulated_signal = i_signal * carrier_cos - q_signal * carrier_sin
        return modulated_signal

import os

def plot_modulation_signals(dI, dQ, s, fs, t_xlim=0.05, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    t = np.arange(len(dI)) / fs
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    # Linha 1: dI e dQ
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, dI, label=r"$d_I(t)$", color='navy', linewidth=2)
    ax1.plot(t, dQ, label=r"$d_Q(t)$", color='darkgreen', linewidth=2)
    ax1.set_xlim(0, t_xlim)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Sinais I/Q - Formatados RRC')
    ax1.grid(True)
    leg1 = ax1.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='white', fontsize=14, fancybox=True)
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_edgecolor('black')
    leg1.get_frame().set_alpha(1.0)

    # Linha 2: sinal modulado
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, s, label=r"$s(t)$", color='darkred', linewidth=1.5)
    ax2.set_xlim(0, t_xlim)
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(r'Sinal Modulado $IQ$')
    ax2.grid(True)
    leg2 = ax2.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='white', fontsize=14, fancybox=True)
    leg2.get_frame().set_facecolor('white')
    leg2.get_frame().set_edgecolor('black')
    leg2.get_frame().set_alpha(1.0)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":

    fs = 128_000
    fc = 4000
    Rb = 400
    alpha = 0.8
    span = 8

    Xnrz = np.random.randint(0, 2, 30)
    Yman = np.random.randint(0, 2, 30)
    print("Xnrz:", ''.join(str(b) for b in Xnrz))
    print("Yman:", ''.join(str(b) for b in Yman))

    formatter = Formatter(alpha=alpha, fs=fs, Rb=Rb, span=span)
    dI = formatter.format(Xnrz)
    dQ = formatter.format(Yman)
    
    print("dI:", ''.join(str(b) for b in dI[:5]))
    print("dQ:", ''.join(str(b) for b in dQ[:5]))
    
    modulator = Modulator(fc=fc, fs=fs)
    s = modulator.modulate(dI, dQ)
    
    print("s:", ''.join(str(b) for b in s[:5]))
    
    output_path = os.path.join("out", "example_modulator.pdf")
    plot_modulation_signals(dI, dQ, s, fs=fs, save_path=output_path)
    