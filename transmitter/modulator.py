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
        return t, modulated_signal

    @staticmethod
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

    @staticmethod
    def plot_eye_diagrams(dI, dQ, fs, Rb, save_path=None):
        sps = int(fs / Rb)
        eye_len = 2 * sps
        n_traces = (len(dI) - eye_len) // sps
        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # Diagrama de olho dI
        ax_eyeI = fig.add_subplot(gs[0, 0])
        for i in range(n_traces):
            start = i * sps
            ax_eyeI.plot(np.arange(eye_len) / fs * 1e3, dI[start:start+eye_len], color='navy', alpha=0.18, linewidth=2)
        ax_eyeI.set_xlabel('Tempo (ms)')
        ax_eyeI.set_ylabel('Amplitude')
        ax_eyeI.set_title(r'Diagrama de Olho de $d_I(t)$')
        ax_eyeI.grid(True)
        ax_eyeI.set_xlim(0, eye_len / fs * 1e3)

        # Diagrama de olho dQ
        ax_eyeQ = fig.add_subplot(gs[0, 1])
        for i in range(n_traces):
            start = i * sps
            ax_eyeQ.plot(np.arange(eye_len) / fs * 1e3, dQ[start:start+eye_len], color='darkgreen', alpha=0.18, linewidth=2)
        ax_eyeQ.set_xlabel('Tempo (ms)')
        ax_eyeQ.set_ylabel('Amplitude')
        ax_eyeQ.set_title(r'Diagrama de Olho de $d_Q(t)$')
        ax_eyeQ.grid(True)
        ax_eyeQ.set_xlim(0, eye_len / fs * 1e3)

        plt.tight_layout()
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_iq_and_constellation(dI, dQ, save_path=None, amplitude=None):
        # Centralizar sinais para o plot
        dI_c = dI - np.mean(dI)
        dQ_c = dQ - np.mean(dQ)
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # Apenas pontos ideais da constelação QPSK
        ax_const = fig.add_subplot(gs[0, 0])
        if amplitude is None:
            amp = np.mean(np.abs(np.concatenate([dI_c*1.1, dQ_c*1.1])))
        else:
            amp = amplitude

        # Scatter IQ
        ax_iq = fig.add_subplot(gs[0, 1])
        ax_iq.scatter(dI_c, dQ_c, color='darkgreen', alpha=0.5, s=2, label='Amostras IQ')
        ax_iq.set_xlabel(r'$d_I(t)$')
        ax_iq.set_ylabel(r'$d_Q(t)$')
        ax_iq.set_title('Plano IQ (Scatter)')
        # Linhas apenas sobre os pontos ideais
        for v in [-amp, amp]:
            ax_iq.axhline(v, color='darkred', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
            ax_iq.axvline(v, color='darkred', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
        from matplotlib.lines import Line2D
        custom_legend = [Line2D([0], [0], marker='o', color='w', label='Amostras IQ',
                                markerfacecolor='darkgreen', markersize=16, alpha=0.7)]
        leg_iq = ax_iq.legend(handles=custom_legend,
                loc='upper right', frameon=True, edgecolor='black',
                facecolor='white', fontsize=12, fancybox=True
            )
        leg_iq.get_frame().set_facecolor('white')
        leg_iq.get_frame().set_edgecolor('black')
        leg_iq.get_frame().set_alpha(1.0)
        ax_iq.set_xlim(-0.06, 0.06)
        ax_iq.set_ylim(-0.06, 0.06)
        ax_iq.set_aspect('equal')

        qpsk_points = np.array([[amp, amp], [amp, -amp], [-amp, amp], [-amp, -amp]])
        ax_const.scatter(qpsk_points[:, 0], qpsk_points[:, 1], color='black', s=160, marker='o', label='Simbolos QPSK', linewidth=5)
        ax_const.set_xlabel(r'$I$')
        ax_const.set_ylabel(r'$Q$')
        ax_const.set_title('Plano IQ (Constelação)')
        # Linhas apenas sobre os pontos ideais
        for v in [-amp, amp]:
            ax_const.axhline(v, color='darkred', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
            ax_const.axvline(v, color='darkred', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
        leg_const = ax_const.legend(
                loc='upper right', frameon=True, edgecolor='black',
                facecolor='white', fontsize=12, fancybox=True
            )
        leg_const.get_frame().set_facecolor('white')
        leg_const.get_frame().set_edgecolor('black')
        leg_const.get_frame().set_alpha(1.0)
        ax_const.set_xlim(-0.06, 0.06)
        ax_const.set_ylim(-0.06, 0.06)
        ax_const.set_aspect('equal')

        plt.tight_layout()
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

    Xnrz = np.random.randint(0, 2, 240)
    Yman = np.random.randint(0, 2, 240)
    print("Xnrz:", ''.join(str(b) for b in Xnrz))
    print("Yman:", ''.join(str(b) for b in Yman))

    formatter = Formatter(alpha=alpha, fs=fs, Rb=Rb, span=span)
    dI = formatter.format(Xnrz)
    dQ = formatter.format(Yman)
    
    print("dI:", ''.join(str(b) for b in dI[:5]))
    print("dQ:", ''.join(str(b) for b in dQ[:5]))
    
    modulator = Modulator(fc=fc, fs=fs)
    t, s = modulator.modulate(dI, dQ)
    
    print("s:", ''.join(str(b) for b in s[:5]))
    
    output_path = os.path.join("out", "example_modulator.pdf")
    Modulator.plot_modulation_signals(dI, dQ, s, fs=fs, save_path=output_path)
    
    output_eye = os.path.join("out", "example_eye.pdf")
    Modulator.plot_eye_diagrams(dI, dQ, fs=fs, Rb=Rb, save_path=output_eye)

    output_constellation = os.path.join("out", "example_constellation.pdf")
    Modulator.plot_iq_and_constellation(dI, dQ, save_path=output_constellation)
    