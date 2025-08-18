"""
Implementação de um canal para aplicação de ruido AWGN.

Autor: Arthur Cadore
Data: 16-08-2025
"""

import numpy as np
from datagram import Datagram
from transmitter import Transmitter
from plotter import save_figure, create_figure, TimePlot, FrequencyPlot

class Noise:
    def __init__(self, snr=10):
        r"""
        Implementação de canal para aplicação de ruido AWGN no sinal transmitido.

        Args:
            snr (float): Relação sinal-ruído em decibéis (dB). Padrão é 10 dB.
        """
        self.snr = snr
    
    def add_noise(self, signal):
        r"""
        Adiciona ruído AWGN ao sinal fornecido.

        Args:
            signal (np.ndarray): Sinal ao qual o ruído será adicionado.
        
        Returns:
            np.ndarray: Sinal com ruído adicionado.
        """
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (self.snr / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise

if __name__ == "__main__":
    datagram = Datagram(pcdnum=1234, numblocks=1)
    transmitter = Transmitter(datagram, output_print=False)
    t, s = transmitter.run()

    snr_db = 15
    add_noise = Noise(snr=snr_db)
    s_noisy = add_noise.add_noise(s)

    fig_time, grid_time = create_figure(2, 1, figsize=(16, 9))

    TimePlot(
        fig_time, grid_time, (0,0),
        t=t,
        signals=[s],
        labels=["$s(t)$"],
        title="Domínio do Tempo - Sem Ruído",
        xlim=(0, 0.1),
        ylim=(-0.1, 0.1),
        colors="darkblue",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    TimePlot(
        fig_time, grid_time, (1,0),
        t=t,
        signals=[s_noisy],
        labels=["$s(t) + AWGN$"],
        title="Domínio do Tempo - Com Ruído",
        xlim=(0, 0.1),
        ylim=(-0.1, 0.1),
        colors="darkred",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    fig_time.tight_layout()
    save_figure(fig_time, "example_noise_time.pdf")

    fig_freq, grid_freq = create_figure(2, 1, figsize=(16, 9))

    FrequencyPlot(
        fig_freq, grid_freq, (0,0),
        fs=transmitter.fs,
        signal=s,
        fc=transmitter.fc,
        labels=["$S(f)$"],
        title="Domínio da Frequência - Sem Ruído",
        xlim=(-8, 8),
        colors="darkblue",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    FrequencyPlot(
        fig_freq, grid_freq, (1,0),
        fs=transmitter.fs,
        signal=s_noisy,
        fc=transmitter.fc,
        labels=["$S(f) + AWGN$"],
        title="Domínio da Frequência - Com Ruído",
        xlim=(-8, 8),
        colors="darkred",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    fig_freq.tight_layout()
    save_figure(fig_freq, "example_noise_freq.pdf")