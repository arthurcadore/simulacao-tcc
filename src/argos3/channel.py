# """
# Implementation of a channel for aggregation of multiple signals.
#
# Author: Arthur Cadore
# Date: 16-08-2025
# """

import numpy as np

from .transmitter import Transmitter
from .datagram import Datagram
from .plotter import save_figure, create_figure, TimePlot, GaussianNoisePlot
from .noise import Noise, NoiseEBN0
from .env_vars import *


class Channel:
    def __init__(self, fs=128_000, duration=1, noise_mode="snr", noise_db=20, seed=10):
        r"""
        Implementation of a channel for aggregation of multiple signals, as displayed on block diagram below. 

        ![pageplot](../assets/channel.svg)
        
        Args:
            fs (int): sampling rate of the signal.
            duration (int): duration of the channel in seconds.
            noise_mode (str): noise mode ('snr' or 'ebn0').
            noise_db (int): noise level in dB.
            seed (int): seed for random number generation. 
    
        Returns:
            Channel: channel object.
    
        Raises:
            ValueError: if the noise mode is invalid.

        Examples: 
            >>> import argos3
            >>> import numpy as np 
            >>>
            >>> transmitter = argos3.Transmitter(fc=2400, output_print=False, output_plot=False)
            >>> t, s = transmitter.transmit(argos3.Datagram(pcdnum=1234, numblocks=1))
            >>> 
            >>> channel = argos3.Channel(duration=1, noise_mode="ebn0", noise_db=20)
            >>> channel.add_signal(s, position_factor=0.5)
            >>> channel.add_noise()
            >>> st = channel.channel
            >>>                   
            >>> receiver = argos3.Receiver(fc=2400, output_print=False, output_plot=False)
            >>> datagramRX, success = receiver.receive(st)
            >>> 
            >>> print(success)
            True
            >>> print(datagramRX.parse_datagram())
            {
              "msglength": 1,
              "pcdid": 1234,
              "data": {
                "bloco_1": {
                  "sensor_1": 37,
                  "sensor_2": 198,
                  "sensor_3": 9
                }
              },
              "tail": 7
            }

            - Time Domain Plot Example: ![pageplot](assets/example_channel_time_channel.svg)
            

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
            raise ValueError("Invalid noise mode. Use 'EBN0', 'SNR'.")

        self.noise_mode = noise_map[noise_mode]
        self.noise_db = noise_db
        self.seed = seed

    def add_signal(self, signal, position_factor=0.5):
        r"""
        Adds a signal to the channel at a relative position.

        Args:
            signal (np.ndarray): signal samples to insert.
            position_factor (float): position factor between [0, 1] (0 = start of the channel, 1 = end).

        Raises:
            ValueError: if position_factor is not between [0, 1].
        
        Examples: 
            - Time Domain Plot Example: ![pageplot](assets/example_channel_time_subchannels.svg)
        """
        if not 0 <= position_factor <= 1:
            raise ValueError("Position factor must be between 0 and 1.")

        chan_len = len(self.channel)
        sig_len = len(signal)

        # Calculate initial position in the channel vector
        start_idx = int(round(position_factor * (chan_len - sig_len)))
        if start_idx < 0:
            start_idx = 0
        if start_idx + sig_len > chan_len:
            sig_len = chan_len - start_idx
            signal = signal[:sig_len]  # cut if it doesn't fit

        # Insert (sum) the signal into the channel
        self.channel[start_idx:start_idx + sig_len] += signal

    def add_noise(self):
        r"""
        Adds noise to the channel.

        Examples: 
            - Time Domain Plot Example: ![pageplot](assets/example_channel_time_channel.svg)
        """
        if self.noise_mode == 0:
            noise = NoiseEBN0(ebn0_db=self.noise_db, seed=self.seed)
        elif self.noise_mode == 1:
            noise = Noise(snr=self.noise_db, seed=self.seed)
        
        self.channel = noise.add_noise(self.channel)
        self.noise = noise.noise
    

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

    fig_time, grid = create_figure(4, 1, figsize=(16, 16))

    TimePlot(
        fig_time, grid, (0, 0),
        t=np.arange(0, len(s)/tx.fs, 1/tx.fs),
        signals=[s],
        labels=["$s(t)$"],
        title=MODULATED_STREAM_TITLE,
        colors=[COLOR_COMBINED],
        amp_norm=True,
    ).plot()    
    
    TimePlot(
        fig_time, grid, (1, 0),
        t=canal1.t,
        signals=[canal1.channel],
        labels=["$s_1(t)$"], 
        title=(MODULATED_STREAM_TITLE + " - Channel 1"),
        colors=[COLOR_COMBINED],
        amp_norm=True,
    ).plot()

    TimePlot(
        fig_time, grid, (2, 0),
        t=canal2.t,
        signals=[canal2.channel],
        labels=["$s_2(t)$"], 
        title=(MODULATED_STREAM_TITLE + " - Channel 2"),
        colors=[COLOR_COMBINED],
        amp_norm=True,
    ).plot()

    TimePlot(
        fig_time, grid, (3, 0),
        t=canal3.t,
        signals=[canal3.channel],
        labels=["$s_3(t)$"], 
        title=(MODULATED_STREAM_TITLE + " - Channel 3"),
        colors=[COLOR_COMBINED],
        amp_norm=True,
    ).plot()

    fig_time.tight_layout()
    save_figure(fig_time, "example_channel_time_subchannels.pdf")
    
    canalT = canal1.channel + canal2.channel + canal3.channel

    Noise = NoiseEBN0(ebn0_db=20, seed=10)
    canalT_NoiseEBN0 = Noise.add_noise(canalT)

    fig_time, grid = create_figure(3, 2, figsize=(16, 12))
    
    TimePlot(
        fig_time, grid, (0, slice(0,2)),
        t=np.arange(0, len(canalT)/tx.fs, 1/tx.fs),
        signals=[canalT],
        labels=["$s(t)$"], 
        title=MODULATED_STREAM_TITLE,
        colors=[COLOR_COMBINED],
        amp_norm=True,
    ).plot()

    TimePlot(
        fig_time, grid, (1,0),
        t=np.arange(0, len(canalT_NoiseEBN0)/tx.fs, 1/tx.fs),
        signals=[Noise.noise],
        labels=[r"$r(t)$"],
        title=NOISE_TITLE,
        xlim=TIME_XLIM,
        ylim=NOISE_DENSITY_YLIM,
        colors=COLOR_COMBINED,
    ).plot()

    GaussianNoisePlot(
        fig_time, grid, (1,1),
        variance=Noise.variance,
        colors=NOISE_DENSITY_COLOR,
        title=(NOISE_DENSITY_TITLE + f" - $E_B/N_0$ {Noise.ebn0_db} $dB$")
    ).plot()

    TimePlot(
        fig_time, grid, (2, slice(0,2)),
        t=np.arange(0, len(canalT_NoiseEBN0)/tx.fs, 1/tx.fs),
        signals=[canalT_NoiseEBN0],
        labels=["$s(t) + r(t)$"], 
        title=(MODULATED_STREAM_TITLE + " + $r(t)$"),
        colors=[COLOR_COMBINED],
        amp_norm=True,
    ).plot()
    
    fig_time.tight_layout()
    save_figure(fig_time, "example_channel_time_channel.pdf")
    
        
    