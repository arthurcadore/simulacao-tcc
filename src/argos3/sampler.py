# """
# Implements a sampler to sample and quantize the received signal.
#
# Author: Arthur Cadore
# Date: 15-08-2025
# """

import numpy as np
from .plotter import save_figure, create_figure, SampledSignalPlot, SymbolsPlot
from .env_vars import *
from .lowpassfilter import LPF
from .matchedfilter import MatchedFilter
from .formatter import Formatter
from .encoder import Encoder

class Sampler:
    def __init__(self, fs=128_000, Rb=400, t=None, delay=0.08):
        r"""
        Initializes the sampler, used for sampling and quantizing the received signal.

        Args: 
            fs (int): Sampling frequency.
            delay (float): Sampling delay, in seconds.
            Rb (int): Bit rate.
            t (numpy.ndarray): Time vector.

        Examples: 
            >>> import argos3
            >>> import numpy as np 
            >>> 
            >>> fs = 128000
            >>> t = np.arange(10000) / fs
            >>> 
            >>> signal1 = np.cos(2 * np.pi * 1000 * t)
            >>> signal2 = np.cos(2 * np.pi * 4000 * t)
            >>> 
            >>> signal = signal1 + signal2
            >>> 
            >>> sampler = argos3.Sampler(t=t)
            >>> 
            >>> signal_sampled = sampler.sample(signal)
            >>> t_indexes = sampler.sample(t)

            - Time Domain: ![pageplot](assets/example_sampler_time.svg) 
            - Symbols Stream Plot Example: ![pageplot](assets/example_sampler_symbols.svg)
        """

        # Attributes
        self.fs = fs
        self.Rb = Rb
        self.sps = int(self.fs / self.Rb)
        self.delay = int(round(delay * self.fs))

        # check if t is not None
        if t is not None:
            self.indexes = self.calc_indexes(t)

    def update_sampler(self, delay, t):
        # function to update the sampler delay and indexes
        self.delay = int(round(delay * self.fs))
        self.indexes = self.calc_indexes(t)
    
    def calc_indexes(self, t):
        r"""
        Calculates the sampling indexes $I[n]$ based on the time vector $t$. The sampling indexes vector $I[n]$ is given by the expression below. 

        $$
        \begin{align}
        I[n] = \tau + n \cdot \left( \frac{f_s}{R_b}\right) \text{ , where: } \quad I[n] < \text{len}(t)
        \end{align}
        $$

        Where:
            - $\tau$: Sampling delay.
            - $f_s$: Sampling frequency.
            - $R_b$: Bit rate.
            - $n$: Sample index.
            - $\text{len}(t)$: Length of the time vector.

        Args:
            t (numpy.ndarray): Time vector.

        Returns:
            indexes (numpy.ndarray): Sampling indexes $I[n]$.
        """
        indexes = np.arange(self.delay, len(t), self.sps)
        indexes = indexes[indexes < len(t)]
        return indexes
    
    def sample(self, signal):
        r"""
        Samples the signal $s(t)$ based on the sampling indexes $I[n]$.

        $$
            s(t) \rightarrow  s([I[n]) \rightarrow s[n]
        $$

        Where:
            - $s(t)$: Input signal $s(t)$.
            - $s[n]$ Sampled signal $s[n]$.
            - $I[n]$ Sampling indexes $I[n]$.

        Args:
            signal (numpy.ndarray): Input signal $s(t)$ to be sampled.

        Returns:
            sampled_signal (numpy.ndarray): Sampled signal $s[n]$.
        """
        sampled_signal = signal[self.indexes]
        return sampled_signal

    def quantize(self, signal):
        r"""
        Quantizes the signal $s[n]$ into discrete values. The quantization process is given by the expression below.

        $$
        \begin{align}
        s'[n] = \begin{cases}
            +1 & \text{if } s[n] \geq 0 \\
            -1 & \text{if } s[n] < 0
        \end{cases}
        \end{align}
        $$

        Where:
            - $s[n]$ Sampled signal $s[n]$.
            - $s'[n]$ Quantized signal $s'[n]$.

        Args:
            signal (numpy.ndarray): Sampled signal $s[n]$.

        Returns:
            symbols (numpy.ndarray): Quantized signal $s'[n]$.
        """
        symbols = []
        for i in range(len(signal)):
            if signal[i] >= 0:
                symbols.append(+1)
            else:
                symbols.append(-1)
        return symbols

if __name__ == "__main__":

    Rb=400

    bit1 = np.random.randint(0, 2, 500)
    bit2 = np.random.randint(0, 2, 500)

    encoder = Encoder(method="NRZ")
    In = encoder.encode(bit1)
    Qn = encoder.encode(bit2)

    fI = Formatter(alpha=0.8, fs=128_000, Rb=Rb, span=6, type="RRC", channel="I", bits_per_symbol=1, prefix_duration=0.005)
    fQ = Formatter(alpha=0.8, fs=128_000, Rb=Rb, span=10, type="Manchester", channel="Q", bits_per_symbol=2, prefix_duration=0.005)
    
    dI = fI.apply_format(In, add_prefix=True)
    dQ = fQ.apply_format(Qn, add_prefix=True)

    r = np.random.normal(0, 1, len(dI))*0.05
    dI += r
    dQ += r

    mfI = MatchedFilter(fs=128_000, Rb=Rb, type="RRC-Inverted", channel="Q", bits_per_symbol=1)
    mfQ = MatchedFilter(fs=128_000, Rb=Rb, type="Manchester-Inverted", channel="Q", bits_per_symbol=2)
    
    signal = mfI.apply_filter(dI)
    signal2 = mfQ.apply_filter(dQ)
    
    t = np.arange(len(signal)) / 128_000
    sampler_I = Sampler(fs=128_000, Rb=Rb, t=t, delay=0.005)
    s_I = sampler_I.sample(signal)
    t_I = sampler_I.sample(t)

    sampler_Q = Sampler(fs=128_000, Rb=Rb, t=t, delay=0.005)
    s_Q = sampler_Q.sample(signal2)
    t_Q = sampler_Q.sample(t)

    In_prime = sampler_I.quantize(s_I)
    Qn_prime = sampler_Q.quantize(s_Q)
    print(In_prime[:20], "...")
    print(Qn_prime[:20], "...")

    fig_sampler, grid_sampler = create_figure(2, 2, figsize=(16, 9))

    SampledSignalPlot(
        fig_sampler, grid_sampler, (0, 0),
        t,
        signal,
        t_I,
        s_I,
        colors=COLOR_I,
        label_signal=r"$I'(t)$", 
        label_samples=r"Samples $I'[n]$", 
        xlim=(0,100),
        title=I_CHANNEL_TITLE,
    ).plot()

    SampledSignalPlot(
        fig_sampler, grid_sampler, (0, 1),
        t,
        signal2,
        t_Q,
        s_Q,
        colors=COLOR_Q,
        label_signal=r"$Q'(t)$", 
        label_samples=r"Samples $Q'[n]$", 
        xlim=(0,100), 
        title=Q_CHANNEL_TITLE,
    ).plot()

    SymbolsPlot(
        fig_sampler, grid_sampler, (1, 0),
        symbols_list=[s_I],
        samples_per_symbol=1,
        colors=[COLOR_I],
        xlabel=SYMBOLS_X,
        ylabel=SYMBOLS_Y,
        label=r"$I'[n]$",
        ylim=[min(s_I)*1.1, max(s_I)*1.1],
        x_axis_label=(min(s_I), max(s_I)),
        show_symbol_values=False,
        xlim=(0,40), 
    ).plot()

    SymbolsPlot(
        fig_sampler, grid_sampler, (1, 1),
        symbols_list=[s_Q],
        samples_per_symbol=1,
        colors=[COLOR_Q],
        xlabel=SYMBOLS_X,
        ylabel=SYMBOLS_Y,
        label=r"$Q'[n]$",
        ylim=[min(s_Q)*1.1, max(s_Q)*1.1],
        x_axis_label=(min(s_Q), max(s_Q)),
        show_symbol_values=False,
        xlim=(0,40), 
    ).plot()

    fig_sampler.tight_layout()
    save_figure(fig_sampler, "example_sampler_time.pdf")

    fig_symbols, grid_symbols = create_figure(2, 1, figsize=(16, 9))

    SymbolsPlot(
        fig_symbols, grid_symbols, (0, 0),
        symbols_list=[In, In_prime],
        samples_per_symbol=1,
        colors=[COLOR_AUX1, COLOR_I],
        xlabel=SYMBOLS_X,
        ylabel=SYMBOLS_Y,
        label=[r"$I[n]$", r"$I \prime[n]$"], 
        linestyles=["-", ":"],
        xlim=(0,100),
        title=I_CHANNEL_TITLE,
    ).plot()

    SymbolsPlot(
        fig_symbols, grid_symbols, (1, 0),
        symbols_list=[Qn, Qn_prime],
        samples_per_symbol=1,
        colors=[COLOR_AUX1, COLOR_Q],
        xlabel=SYMBOLS_X,
        ylabel=SYMBOLS_Y,
        label=[r"$Q[n]$", r"$Q \prime[n]$"], 
        linestyles=["-", ":"],
        xlim=(0,100),
        title=Q_CHANNEL_TITLE,
    ).plot()

    fig_symbols.tight_layout()
    save_figure(fig_symbols, "example_sampler_symbols.pdf")