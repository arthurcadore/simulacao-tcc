# """
# Implementação de um canal para agregação de multiplos sinais.

# Autor: Arthur Cadore
# Data: 16-08-2025
# """

import numpy as np

from .transmitter import Transmitter
from .datagram import Datagram
from .plotter import save_figure, create_figure, TimePlot, FrequencyPlot
from .noise import Noise, NoiseEBN0


class Channel:
    def __init__(self, fs=128_000, duration=1, noise_mode="snr", noise_db=20, seed=10):
        r"""
        Implementação de um canal para agregação de multiplos sinais.
        
        Args:
            fs (int): taxa de amostragem do sinal.
            duration (int): duração do canal em segundos.
            noise_mode (str): modo de ruído ('snr' ou 'ebn0').
            noise_db (int): nível de ruído em dB.
            seed (int): semente para geração de números aleatórios. 
    
        Returns:
            Channel: objeto do canal.
    
        Raises:
            ValueError: se o modo de ruído for inválido.
        """
        self.fs = fs
        self.channel = np.zeros(int(fs * duration))
        self.t = np.arange(0, duration, 1/fs)

        noise_map = {
            "ebn0": 0,
            "snr": 1
        }

        noise_mode = noise_mode.lower()
        if noise_mode not in noise_map:
            raise ValueError("Modo de ruído inválido. Use 'EBN0', 'SNR'.")

        self.noise_mode = noise_map[noise_mode]
        self.noise_db = noise_db
        self.seed = seed

    def add_signal(self, signal, position_factor=0.5):
        r"""
        Adiciona um sinal ao canal em uma posição relativa.

        Args:
            signal (np.ndarray): vetor de amostras do sinal a inserir.
            position_factor (float): fator de posição entre [0, 1] (0 = início do canal, 1 = final).

        Raises:
            ValueError: se position_factor não estiver entre [0, 1].
        
        Example: 
            ![pageplot](assets/example_channel_time_subchannels.svg)
        """
        if not 0 <= position_factor <= 1:
            raise ValueError("position_factor deve estar entre 0 e 1.")

        chan_len = len(self.channel)
        sig_len = len(signal)

        # Calcula posição inicial no vetor do canal
        start_idx = int(round(position_factor * (chan_len - sig_len)))
        if start_idx < 0:
            start_idx = 0
        if start_idx + sig_len > chan_len:
            sig_len = chan_len - start_idx
            signal = signal[:sig_len]  # corta se não couber

        # Insere (soma) o sinal no canal
        self.channel[start_idx:start_idx + sig_len] += signal

    def add_noise(self):
        r"""
        Adiciona ruído ao canal.

        Example: 
            ![pageplot](assets/example_channel_time_channel.svg)
        """
        if self.noise_mode == 0:
            noise = NoiseEBN0(ebn0_db=self.noise_db, seed=self.seed)
        elif self.noise_mode == 1:
            noise = Noise(snr=self.noise_db, seed=self.seed)
        
        self.channel = noise.add_noise(self.channel)
    

if __name__ == "__main__":

    # Cria transmissor e gera o vetor de sinal.
    tx = Transmitter()
    datagram = Datagram(pcdnum=1234, numblocks=1, seed=10)
    t, s = tx.transmit(datagram)
    
    # Cria o canal
    canal1 = Channel(fs=tx.fs, duration=1, noise_mode="snr", noise_db=20, seed=10)
    canal2 = Channel(fs=tx.fs, duration=1, noise_mode="snr", noise_db=20, seed=10)
    canal3 = Channel(fs=tx.fs, duration=1, noise_mode="snr", noise_db=20, seed=10)
    
    # coloca o sinal no meio do canal
    canal1.add_signal(s, position_factor=0.1)
    canal2.add_signal(s, position_factor=0.5)
    canal3.add_signal(s, position_factor=0.9)

    fig_time, grid = create_figure(4, 1, figsize=(16, 9))

    TimePlot(
        fig_time, grid, (0, 0),
        t=np.arange(0, len(s)/tx.fs, 1/tx.fs),
        signals=[s],
        labels=["$s(t)$"],
        title="Sinal Modulado $s(t)$",
        colors=["darkred"],
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()    
    
    TimePlot(
        fig_time, grid, (1, 0),
        t=canal1.t,
        signals=[canal1.channel],
        labels=["$s(t)$"], 
        title="Sinal Modulado - Canal 1",
        colors="darkred",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()

    TimePlot(
        fig_time, grid, (2, 0),
        t=canal2.t,
        signals=[canal2.channel],
        labels=["$s(t)$"], 
        title="Sinal Modulado - Canal 2",
        colors="darkred",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()

    TimePlot(
        fig_time, grid, (3, 0),
        t=canal3.t,
        signals=[canal3.channel],
        labels=["$s(t)$"], 
        title="Sinal Modulado - Canal 3",
        colors="darkred",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()

    fig_time.tight_layout()
    save_figure(fig_time, "example_channel_time_subchannels.pdf")
    
    canalT = canal1.channel + canal2.channel + canal3.channel

    canalT_NoiseEBN0 = NoiseEBN0(ebn0_db=20, seed=10).add_noise(canalT)

    fig_time, grid = create_figure(2, 1, figsize=(16, 9))
    
    TimePlot(
        fig_time, grid, (0, 0),
        t=np.arange(0, len(canalT)/tx.fs, 1/tx.fs),
        signals=[canalT],
        labels=["$s(t)$"], 
        title="Sinal Modulado - Canal Total",
        colors="darkred",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()

    TimePlot(
        fig_time, grid, (1, 0),
        t=np.arange(0, len(canalT_NoiseEBN0)/tx.fs, 1/tx.fs),
        signals=[canalT_NoiseEBN0],
        labels=["$s(t)$"], 
        title="Sinal Modulado - Canal Total + Ruido",
        colors="darkred",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    fig_time.tight_layout()
    save_figure(fig_time, "example_channel_time_channel.pdf")
    
        
    