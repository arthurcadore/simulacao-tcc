# """
# Implements a low-pass filter to remove the high frequency component of the received signal.
#
# Author: Arthur Cadore
# Date: 28-07-2025
# """

import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from .plotter import create_figure, save_figure, ImpulseResponsePlot, TimePlot, PoleZeroPlot, FrequencyResponsePlot, FrequencyPlot
from .env_vars import *

class LPF:
    def __init__(self, cut_off=600, order=6, fs=128_000, type="butter"):
        r"""
        Initializes a low-pass filter with a cutoff frequency $f_{cut}$ and an order $N$, used to remove high frequency components from the received signal.

        Args:
            cut_off (float): Cutoff frequency $f_{cut}$ of the filter.
            order (int): Order $N$ of the filter.
            fs (int, optional): Sampling frequency $f_s$. 
            type (str, optional): Filter type. Default is "butter".
        
        Raises:
            ValueError: If the filter type is invalid.

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
            >>> lpf = argos3.LPF(cut_off=1500, order=6, fs=fs, type="butter")
            >>> 
            >>> signal = signal1 + signal2
            >>> 
            >>> signal_filtered = lpf.apply_filter(signal)

            - Time Domain Example: ![pageplot](assets/example_lpf_signals.svg) 
            - Frequency Domain Example: ![pageplot](assets/example_lpf_freq.svg)
        """

        # Attributes
        self.cut_off = cut_off
        self.order = order
        self.fs = fs

        type = type.lower()
        if type != "butter":
            raise ValueError("Tipo de filtro inválido. Use 'butter'.")

        # Filter Coefficients
        self.b, self.a = self.butterworth_filter()
        self.impulse_response, self.t_impulse = self.calc_impulse_response()

    def butterworth_filter(self, fNyquist=0.5):
        r"""
        Calculates the Butterworth filter coefficients using the `scipy.signal` library. The continuous-time transfer function $H(s)$ of a Butterworth filter is given by the expression below.

        $$
        H(s) = \frac{1}{1 + \left(\frac{s}{2 \pi f_{cut}}\right)^{2n}}
        $$

        Where:
            - $s$: Complex variable in the Laplace domain.
            - $2 \pi f_{cut}$: Cutoff frequency of the filter.
            - $n$: Order of the filter.

        Args: 
            fNyquist (float): Nyquist factor. Default is 0.5 * fs.

        Returns:
            b (np.ndarray): Coefficients $b$ corresponding to the transfer function of the Butterworth filter.
            a (np.ndarray): Coefficients $a$ corresponding to the transfer function of the Butterworth filter.

        Examples:
            - Pole-Zero Plot: ![pageplot](assets/example_lpf_pz.svg)
        """

        # Calculate filter coefficients
        b, a = butter(self.order, self.cut_off / (fNyquist * self.fs), btype='low')
        return b, a

    def calc_impulse_response(self, impulse_len=1024):
        r"""
        To obtain the impulse response in the time domain, a unit impulse is applied as input. For a Butterworth filter, the calculation is given by the expression below. 

        $$
        h(t) = \mathcal{L}^{-1}\left\{H(f)\right\}
        $$

        Where:
            - $h(t)$: Impulse response of the filter.
            - $H(f)$: Transfer function of the filter.
            - $\mathcal{L}^{-1}$: Inverse Laplace transform.

        Args: 
            impulse_len (int): Length of the impulse vector.

        Returns:
            impulse_response (tuple[np.ndarray, np.ndarray]): Impulse response and time vector.
        
        Examples: 
            - Impulse Response: ![pageplot](assets/example_lpf_impulse.svg)
        """
        # Unit impulse
        impulse_input = np.zeros(impulse_len)
        impulse_input[0] = 1

        # Impulse response
        impulse_response = lfilter(self.b, self.a, impulse_input)
        t_impulse = np.arange(impulse_len) / self.fs
        return impulse_response, t_impulse

    def apply_filter(self, signal):
        r"""
        Applies the low-pass filter with impulse response $h(t)$ to the input signal $s(t)$, using the `scipy.signal.filtfilt` function. The filtering process is given by the expression below. 

        $$
            x(t) = s(t) \ast h(t)
        $$

        Where: 
            - $x(t)$: Filtered signal.
            - $s(t)$: Input signal.
            - $h(t)$: Impulse response of the filter.

        Args:
            signal (np.ndarray): Input signal $s(t)$.

        Returns:
            signal_filtered (np.ndarray): Filtered signal $x(t)$.
        """
        # Filter signal
        signal_filtered = filtfilt(self.b, self.a, signal)
        return signal_filtered


if __name__ == "__main__":
    
    fs = 128_000
    t = np.arange(10000) / fs

    # create two cossine signals with different frequencies
    f1 = 1000
    f2 = 4000
    signal = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)

    filtro = LPF(cut_off=1500, order=6, fs=fs, type="butter")
    signal_filtered = filtro.apply_filter(signal)

    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0, 0),
        filtro.t_impulse, filtro.impulse_response,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$h(t)$", 
        xlim=(0, 5), 
        amp_norm=True, 
        title=LPF_IMPULSE_TITLE
    ).plot()

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_lpf_impulse.pdf")

    fig_signal, grid_signal = create_figure(2, 2, figsize=(16, 9))

    ImpulseResponsePlot(
        fig_signal, grid_signal, (0, slice(0, 2)),
        filtro.t_impulse, filtro.impulse_response,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$h(t)$", 
        xlim=(0, 5), 
        amp_norm=True,
        title=LPF_IMPULSE_TITLE
    ).plot()
    
    TimePlot(
        fig_signal, grid_signal, (1, 0),
        t, 
        signal,
        labels=[r"$x(t)$"],
        xlim=(0, 8),
        amp_norm=True,
        colors=COLOR_I,
        title=INPUT_STREAM_TITLE
    ).plot()

    TimePlot(
        fig_signal, grid_signal, (1, 1),
        t, 
        signal_filtered,
        labels=[r"$x'(t)$"],
        xlim=(0, 8),
        amp_norm=True,
        colors=COLOR_I,
        title=OUTPUT_STREAM_TITLE
    ).plot()

    fig_signal.tight_layout()
    save_figure(fig_signal, "example_lpf_signals.pdf")

    fig_pz, grid_pz = create_figure(1, 1, figsize=(10,10))
    PoleZeroPlot(
            fig_pz, grid_pz, (0,0), 
            filtro.b, filtro.a,
            colors=COLOR_AUX1,
            title=LPF_PZ_TITLE,
        ).plot()
    save_figure(fig_pz, "example_lpf_pz.pdf")

    freq_response, grid_freq_response = create_figure(1, 1, figsize=(16,6))
    FrequencyResponsePlot(
            freq_response, grid_freq_response, (0,0), 
            filtro.b, filtro.a, 
            fs=filtro.fs, 
            f_cut=filtro.cut_off, 
            xlim=(0, 3*filtro.cut_off),
            title=LPF_FREQ_TITLE    
        ).plot()
    save_figure(freq_response, "example_lpf_freq_response.pdf")

    fig_freq, grid_freq = create_figure(2, 2, figsize=(16,6))
    FrequencyResponsePlot(
            fig_freq, grid_freq, (0,slice(0,2)), 
            filtro.b, filtro.a, 
            fs=filtro.fs, 
            f_cut=filtro.cut_off, 
            xlim=(0, 3*filtro.cut_off),
            title=LPF_FREQ_TITLE
        ).plot()

    FrequencyPlot(
        fig_freq, grid_freq, (1,0),
        fs=filtro.fs,
        signal=signal,
        labels=[r"$X(f)$"],
        title=INPUT_STREAM_TITLE,
        xlim=(-5000, 5000),
        colors=COLOR_I
    ).plot()

    FrequencyPlot(
        fig_freq, grid_freq, (1,1),
        fs=filtro.fs,
        signal=signal_filtered,
        labels=[r"$X'(f)$"],
        title=OUTPUT_STREAM_TITLE,
        xlim=(-5000, 5000),
        colors=COLOR_I
    ).plot()
    
    save_figure(fig_freq, "example_lpf_freq.pdf")
    