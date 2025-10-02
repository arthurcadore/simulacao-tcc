# """
# Implements a Matched Filter to maximize the SNR of the received signal.
#
# Author: Arthur Cadore
# Date: 15-08-2025
# """

import numpy as np
from .plotter import create_figure, save_figure, ImpulseResponsePlot, TimePlot, BitsPlot
from .formatter import Formatter
from .encoder import Encoder
from .env_vars import * 

class MatchedFilter:
    def __init__(self, alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC-Inverted", channel=None, bits_per_symbol=1):
        r"""
        Initializes a matched filter to maximize the SNR of the received signal.

        Args:
            alpha (float): Roll-off factor of the matched filter.
            fs (int): Sampling frequency.
            Rb (int): Bit rate.
            span (int): Duration of the pulse in terms of bit periods.
            type (str): Type of filter, currently only "RRC-Inverted" and "Manchester-Inverted" are supported.
            channel (str): Channel type, select between "Q" and "I". 
            bits_per_symbol (int): Number of bits per symbol.

        Raises:     
            ValueError: If the filter type is not supported.

        Examples: 
            >>> import argos3
            >>> import numpy as np 
            >>> 
            >>> Y = np.random.randint(0, 2, 20)
            >>> X = np.random.randint(0, 2, 20)
            >>> 
            >>> Xn = argos3.Encoder().encode(X)
            >>> Yn = argos3.Encoder().encode(Y)
            >>> 
            >>> formatterI = argos3.Formatter(type="RRC", channel="I", bits_per_symbol=1)
            >>> formatterQ = argos3.Formatter(type="Manchester", channel="Q", bits_per_symbol=2)
            >>> 
            >>> dI = formatterI.apply_format(Xn)
            >>> dQ = formatterQ.apply_format(Yn)
            >>> 
            >>> mfI = argos3.MatchedFilter(type="RRC-Inverted", channel="I", bits_per_symbol=1)
            >>> mfQ = argos3.MatchedFilter(type="Manchester-Inverted", channel="Q", bits_per_symbol=2)
            >>> 
            >>> dI_prime = mfI.apply_filter(dI)
            >>> dQ_prime = mfQ.apply_filter(dQ)

            - Time Domain: ![pageplot](assets/receiver_mf_time.svg) 
            - Frequency Domain: ![pageplot](assets/receiver_mf_freq.svg)
        """

        # Attributes
        self.alpha = alpha
        self.fs = fs
        self.Rb = Rb
        self.Tb = 1 / Rb
        self.sps = int(fs / Rb)
        self.span = span
        self.channel = channel
        self.bits_per_symbol = bits_per_symbol
        
        # Type mapping
        type_map = {
            "rrc-inverted": 0,
            "manchester-inverted": 1
        }

        type = type.lower()
        if type not in type_map:
            raise ValueError("Tipo de filtro inválido. Use 'RRC-inverted' ou 'Manchester-inverted'.")
        
        self.type = type_map[type]


        # Create a formatter pulse
        if self.type == 0:  # RRC
            self.formatter = Formatter(alpha=self.alpha, fs=self.fs, Rb=self.Rb, span=self.span, type="RRC", channel=self.channel, bits_per_symbol=self.bits_per_symbol)
        elif self.type == 1:  # Manchester
            self.formatter = Formatter(alpha=self.alpha, fs=self.fs, Rb=self.Rb, span=self.span, type="Manchester", channel=self.channel, bits_per_symbol=self.bits_per_symbol)
        
        self.g = self.formatter.g
        self.t_rc = self.formatter.t_rc

        # Invert the pulse (matched filter)
        self.g_inverted = self.inverted_pulse(self.g)

    def inverted_pulse(self, pulse):
        r"""
        Inverts the pulse g(t) to be used as the matched filter g(-t).

        Args:
            pulse (np.ndarray): Pulse to be inverted.

        Returns:
            pulse_inverted (np.ndarray): Inverted pulse.

        Examples:
            - RRC Matched Filter Impulse Response: ![pageplot](assets/example_mf_impulse.svg)
            - Manchester Matched Filter Impulse Response: ![pageplot](assets/example_mf_impulse_man.svg)

        """
        return pulse[::-1]


    def apply_filter(self, signal):
        r"""
        Applies the matched filter to the input signal $s(t)$.

        $$
            x(t) = s(t) \ast g(-t)
        $$

        Where: 
            - $x(t)$: Filtered signal.
            - $s(t)$: Input signal.
            - $g(-t)$: Inverted pulse.

        Args:
            signal (np.ndarray): Input signal $s(t)$.

        Returns:
            signal_filtered (np.ndarray): Filtered signal $x(t)$.
        """
        # Full convolution
        y_full = np.convolve(signal, self.g_inverted, mode='full')

        # Filter delay
        delay = (len(self.g_inverted) - 1) // 2

        # Extract the part aligned with the original signal
        start = delay
        end = start + len(signal)
        if end > len(y_full):  # padding de segurança
            y_full = np.pad(y_full, (0, end - len(y_full)), mode='constant')

        signal_filtered = y_full[start:end]

        # Safe normalization
        pulse_energy = np.sum(self.g_inverted**2)
        if pulse_energy > 0:
            signal_filtered = signal_filtered / pulse_energy

        return signal_filtered

if __name__ == "__main__":

    bit1 = np.random.randint(0, 2, 10)
    bit2 = bit1

    encoder_1 = Encoder(method="Manchester")
    encoder_2 = Encoder(method="NRZ")

    Yman1 = encoder_1.encode(bit1)
    Yman2 = encoder_2.encode(bit2)

    formatterQ1 = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC", channel="Q", bits_per_symbol=1)
    formatterQ2 = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6, type="Manchester", channel="Q", bits_per_symbol=2)

    dQ1 = formatterQ1.apply_format(Yman1, add_prefix=False)
    dQ2 = formatterQ2.apply_format(Yman2, add_prefix=False)

    filtroQ1 = MatchedFilter(alpha=0.8, fs=128_000, Rb=400, span=12, type="RRC-Inverted", channel="Q", bits_per_symbol=1)
    filtroQ2 = MatchedFilter(alpha=0.8, fs=128_000, Rb=400, span=12, type="Manchester-Inverted", channel="Q", bits_per_symbol=2)

    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))
    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0,0),
        filtroQ1.t_rc, [filtroQ1.g, filtroQ1.g_inverted],
        t_unit="ms",
        colors=[COLOR_AUX1, COLOR_AUX2],
        label=[r"$g(t)$", r"$g(-t)$"],
        xlabel=IMPULSE_X,
        ylabel=IMPULSE_Y,
        xlim=IMPULSE_XLIM,
        amp_norm=True
    ).plot()
    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_mf_impulse.pdf")
    

    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))
    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0,0),
        filtroQ2.t_rc, [filtroQ2.g, filtroQ2.g_inverted],
        t_unit="ms",
        colors=[COLOR_AUX1, COLOR_AUX2],
        label=[r"$g(t)$", r"$g(-t)$"],
        xlabel=IMPULSE_X,
        ylabel=IMPULSE_Y,
        xlim=IMPULSE_XLIM,
        amp_norm=True
    ).plot()
    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_mf_impulse_man.pdf")


    dQ1_filtered = filtroQ1.apply_filter(dQ1)
    dQ2_filtered = filtroQ2.apply_filter(dQ2)

    fig_time, grid_time = create_figure(3, 2, figsize=(16, 9))

    BitsPlot(
        fig_time, grid_time, (0,0),
        bits_list=[bit1],
        sections=[("Bits", len(bit1))],
        title=BITSTREAM_X,
        colors=[COLOR_AUX1],
    ).plot()

    BitsPlot(
        fig_time, grid_time, (0,1),
        bits_list=[bit2],
        sections=[("Bits", len(bit2))],
        title=BITSTREAM_X,
        colors=[COLOR_AUX2],
    ).plot()

    TimePlot(
        fig_time, grid_time, (1,0),
        t= np.arange(len(dQ1)) / formatterQ1.fs,
        signals=[dQ1],
        labels=[r"$d_Q1(t)$"],
        title=r"Canal $Q1$",
        amp_norm=True,
        colors=[COLOR_AUX1],
    ).plot()

    TimePlot(
        fig_time, grid_time, (1,1),
        t= np.arange(len(dQ2)) / formatterQ2.fs,
        signals=[dQ2],
        labels=[r"$d_Q2(t)$"],
        title=r"Canal $Q2$",
        amp_norm=True,
        colors=[COLOR_AUX2],
    ).plot()
    
    TimePlot(
        fig_time, grid_time, (2,0),
        t= np.arange(len(dQ1_filtered)) / formatterQ1.fs,
        signals=[dQ1_filtered],
        labels=[r"$d_Q1(t)$"],
        title=r"Canal $Q1$",
        amp_norm=True,
        colors=[COLOR_AUX1],
    ).plot()

    TimePlot(
        fig_time, grid_time, (2,1),
        t= np.arange(len(dQ2_filtered)) / formatterQ2.fs,
        signals=[dQ2_filtered],
        labels=[r"$d_Q2(t)$"],
        title=r"Canal $Q2$",
        amp_norm=True,
        colors=[COLOR_AUX2],
    ).plot()

    fig_time.tight_layout()
    save_figure(fig_time, "example_mf_time.pdf")
    