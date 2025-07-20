import numpy as np
import matplotlib.pyplot as plt
import json
import os
import scienceplots
import komm 
from collections import defaultdict
import matplotlib.gridspec as gridspec

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


class Formatter:
    def __init__(self, alpha=0.8, fs=128_000, Rb=400, span=6):
        self.alpha = alpha
        self.fs = fs
        self.Rb = Rb
        self.Tb = 1 / Rb
        self.sps = int(fs / Rb)
        self.span = span
        self.t_rc = np.linspace(-span * self.Tb, span * self.Tb, span * self.sps * 2)
        self.g = self.rrc_pulse(self.t_rc, self.Tb, self.alpha)

    @staticmethod
    def rrc_pulse(t, Tb, alpha):
        t = np.array(t, dtype=float)
        rc = np.zeros_like(t)
        for i, ti in enumerate(t):
            if np.isclose(ti, 0.0):
                rc[i] = 1.0 + alpha * (4/np.pi - 1)
            elif alpha != 0 and np.isclose(np.abs(ti), Tb/(4*alpha)):
                rc[i] = (alpha/np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                    (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
                )
            else:
                num = np.sin(np.pi * ti * (1 - alpha) / Tb) + \
                      4 * alpha * (ti / Tb) * np.cos(np.pi * ti * (1 + alpha) / Tb)
                den = np.pi * ti * (1 - (4 * alpha * ti / Tb) ** 2) / Tb
                rc[i] = num / den
        # Normaliza energia para 1
        rc = rc / np.sqrt(np.sum(rc**2))
        return rc

    def format(self, symbols, pulse=None, sps=None):
        if pulse is None:
            pulse = self.g
        if sps is None:
            sps = self.sps
        upsampled = np.zeros(len(symbols) * sps)
        upsampled[::sps] = symbols
        return np.convolve(upsampled, pulse, mode='same')

    def plot_rrc_and_signals(self, d_I, d_Q, save_path=None, t_xlim=0.05):

        t_interp = np.arange(len(d_I)) / self.fs

        fig_interp = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

        # Pulso RRC
        ax_rcc = fig_interp.add_subplot(gs[0, :])
        ax_rcc.plot(self.t_rc, self.g, label=fr'Pulso RRC ($\alpha={self.alpha}$)', color='red', linewidth=2)
        ax_rcc.set_title('Pulso Root Raised Cosine (RRC)')
        ax_rcc.set_ylabel('Amplitude')
        ax_rcc.grid(True)
        leg_rcc = ax_rcc.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_rcc.get_frame().set_facecolor('white')
        leg_rcc.get_frame().set_edgecolor('black')
        leg_rcc.get_frame().set_alpha(1.0)
        ax_rcc.set_xlim(-self.Tb*4, self.Tb*4)

        # Sinal I
        ax_I = fig_interp.add_subplot(gs[1, 0])
        ax_I.plot(t_interp, d_I, label=r"$d_I(t)$", color='navy', linewidth=2)
        ax_I.set_title(r"Sinal $d_I(t)$")
        ax_I.set_xlabel('Tempo (s)')
        ax_I.set_ylabel('Amplitude')
        ax_I.set_xlim(0, t_xlim)
        ax_I.grid(True)
        leg_I = ax_I.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_I.get_frame().set_facecolor('white')
        leg_I.get_frame().set_edgecolor('black')
        leg_I.get_frame().set_alpha(1.0)

        # Sinal Q
        ax_Q = fig_interp.add_subplot(gs[1, 1])
        ax_Q.plot(t_interp, d_Q, label=r"$d_Q(t)$", color='darkgreen', linewidth=2)
        ax_Q.set_title(r"Sinal $d_Q(t)$")
        ax_Q.set_xlabel('Tempo (s)')
        ax_Q.set_ylabel('Amplitude')
        ax_Q.set_xlim(0, t_xlim)
        ax_Q.grid(True)
        leg_Q = ax_Q.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_Q.get_frame().set_facecolor('white')
        leg_Q.get_frame().set_edgecolor('black')
        leg_Q.get_frame().set_alpha(1.0)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4)
        if save_path:
            plt.savefig(save_path)
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
    
    output_path = os.path.join("out", "example_formatter.pdf")
    formatter.plot_rrc_and_signals(dI, dQ, output_path)
    