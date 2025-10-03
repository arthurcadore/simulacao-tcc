# """
# Implementation of a QPSK modulator for digital signal transmission.
# 
# Author: Arthur Cadore
# Date: 28-07-2025
# """

from audioop import rms
import numpy as np
from .formatter import Formatter
from .encoder import Encoder
from .plotter import PhasePlot, create_figure, save_figure, TimePlot, FrequencyPlot, ConstellationPlot 
from scipy.signal import hilbert
from .env_vars import *

class Modulator:
    def __init__(self, fc=None, fs=128_000):
        r"""
        Initializes a QPSK modulator in the ARGOS-3 standard. The modulator can be represented by the block diagram shown below.

        ![pageplot](../assets/modulator.svg)

        Args:
            fc (float): Carrier frequency.
            fs (int): Sampling frequency.

        Raises:
            ValueError: If the sampling frequency is not greater than twice the carrier frequency. (Nyquist Theorem)
       
        Examples: 
            >>> import argos3
            >>> import numpy as np 
            >>> 
            >>> X = np.random.randint(0, 2, 20)
            >>> Y = np.random.randint(0, 2, 20)
            >>>
            >>> print(X)
            [0 1 1 1 0 1 1 1 0 0 0 0 0 0 1 0 1 1 1 1]
            >>> print(Y)
            [0 0 0 1 0 1 1 0 0 1 0 0 0 1 0 0 1 1 0 1]
            >>> 
            >>> symbols_I = argos3.Encoder(method="NRZ").encode(X)
            >>> symbols_Q = argos3.Encoder(method="NRZ").encode(Y)
            >>> 
            >>> formatter_I = argos3.Formatter(Rb=400, type="RRC", channel="I", bits_per_symbol=1, prefix_duration=0.082)
            >>> formatter_Q = argos3.Formatter(Rb=400, type="RRC", channel="Q", bits_per_symbol=2, prefix_duration=0.082)
            >>> 
            >>> dI = formatter_I.apply_format(symbols_I, add_prefix=True)
            >>> dQ = formatter_Q.apply_format(symbols_Q, add_prefix=True)
            >>> 
            >>> modulator = argos3.Modulator(fc=4000)
            >>> t, s = modulator.modulate(dI, dQ)
            >>> 
            >>> print(s[:10])
            [-0.07484442 -0.05456886 -0.03230473 -0.00879478  0.01503904  0.03826391
              0.05997333  0.07932302  0.09556429  0.10807341]
            >>> print(t[:10])
            [0.00000e+00 7.81250e-06 1.56250e-05 2.34375e-05 3.12500e-05 3.90625e-05
              4.68750e-05 5.46875e-05 6.25000e-05 7.03125e-05]

            - Time Domain Example: ![pageplot](assets/example_modulator_time.svg)
            - Constellation/Phase Example: ![pageplot](assets/example_modulator_constellation.svg)
            - Frequency Domain Example: ![pageplot](assets/example_modulator_freq.svg)
            - Pure Carrier Example: ![pageplot](assets/example_modulator_carrier.svg)

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (section 3.2.5.3)
        </div>
        """
        # Checks if the carrier frequency is valid
        if fc is None or fc <= 0:
            raise ValueError("Carrier frequency must be greater than zero.")
        
        # Checks if the sampling frequency is valid
        if fs <= fc*2:
            raise ValueError("Sampling frequency must be greater than twice the carrier frequency.")
        
        self.fc = fc
        self.fs = fs

    def modulate(self, i_signal, q_signal):
        r"""
        Modulates the signals $d_I$ and $d_Q$ with a carrier $f_c$, resulting in the modulated signal $s(t)$. The modulation process is given by the expression below.

        $$
            s(t) = d_I(t) \cdot \cos(2\pi f_c t) - d_Q(t) \cdot \sin(2\pi f_c t)
        $$

        Where: 
            - $s(t)$: Modulated signal.
            - $d_I(t)$ and $d_Q(t)$: Formatted signals corresponding to the $I$ and $Q$ channels.
            - $f_c$: Carrier frequency.
            - $t$: Time vector.

        Args:
            i_signal (np.ndarray): Signal $d_I$ corresponding to the $I$ channel to be modulated.
            q_signal (np.ndarray): Signal $d_Q$ corresponding to the $Q$ channel to be modulated.

        Returns:
            modulated_signal (np.ndarray): Modulated signal $s(t)$ result.
            t (np.ndarray): Time vector $t$ corresponding to the modulated signal.

        Raises:
            ValueError: If the signals I and Q do not have the same size.
        """
        # Checks if the signals I and Q have the same size
        n = len(i_signal)
        if len(q_signal) != n:
            raise ValueError("i_signal and q_signal must have the same size.")
        
        # Time vector
        t = np.arange(n) / self.fs
        
        # Carrier signals
        carrier_cos = np.cos(2 * np.pi * self.fc * t)
        carrier_sin = np.sin(2 * np.pi * self.fc * t)
        
        # Modulated signal
        modulated_signal = (i_signal * carrier_cos - q_signal * carrier_sin)

        return t, modulated_signal
    
    def demodulate(self, modulated_signal, carrier_length=0.07, carrier_delay=0):
        r"""
        Demodulates the modulated signal $s(t)$, with phase synchronization using the first `carrier_length` seconds of carrier.

        Args:
            modulated_signal (np.ndarray): Modulated signal $s(t)$ to be demodulated.
            carrier_length (float): The length of the carrier for phase synchronization (in seconds).
            carrier_delay (float): The delay of the carrier for phase synchronization (in seconds).

        Returns:
            i_signal (np.ndarray): Signal $x_I'(t)$ recovered.
            q_signal (np.ndarray): Signal $y_Q'(t)$ recovered.
        """
        # Checks if the modulated signal is empty
        n = len(modulated_signal)
        if n == 0:
            raise ValueError("The modulated signal cannot be empty.")

        # Time vector
        t = np.arange(n) / self.fs

        # Extracts a subvector of the modulated signal to estimate the phase
        carrier_signal = modulated_signal[int(carrier_delay * self.fs):(int(carrier_delay * self.fs) + int(carrier_length * self.fs))]

        # Applies the Hilbert Transform to obtain the instantaneous phase
        analytic_signal = hilbert(carrier_signal)
        original_phase = np.angle(analytic_signal)

        # Estimates the phase and corrects the signal
        phase_estimate = np.mean(original_phase)
        modulated_signal_corrected = modulated_signal * np.exp(-1j * phase_estimate)

        # Calculates the demodulation components
        carrier_cos = 2 * np.cos(2 * np.pi * self.fc * t)
        carrier_sin = 2 * np.sin(2 * np.pi * self.fc * t)

        # Demodulates the signal
        i_signal = modulated_signal_corrected * carrier_cos
        q_signal = -modulated_signal_corrected * carrier_sin

        # Polarity correction
        if np.mean(i_signal) < 0:
            i_signal = -i_signal
            q_signal = -q_signal

        return np.real(i_signal), np.real(q_signal)


if __name__ == "__main__":

    fs = 128_000
    fc = 4000
    Rb = 400
    alpha = 0.8
    span = 8
    carrier_duration = 0.082

    Xnrz = np.random.randint(0, 2, 200)
    Yman = np.random.randint(0, 2, 200)

    encoder_nrz = Encoder(method="NRZ")
    encoder_man = Encoder(method="NRZ")

    Xnrz = encoder_nrz.encode(Xnrz)
    Yman = encoder_man.encode(Yman)

    print("Xnrz:", ''.join(str(b) for b in Xnrz[:20]))
    print("Yman:", ''.join(str(b) for b in Yman[:20]))

    formatterI = Formatter(alpha=alpha, fs=fs, Rb=Rb, type="RRC", span=span, channel="I", bits_per_symbol=1, prefix_duration=carrier_duration)
    formatterQ = Formatter(alpha=alpha, fs=fs, Rb=Rb, type="Manchester", span=span, channel="Q", bits_per_symbol=2, prefix_duration=carrier_duration)
    
    dI = formatterI.apply_format(Xnrz)
    dQ = formatterQ.apply_format(Yman)

    print("dI:", ''.join(str(b) for b in dI[:5]))
    print("dQ:", ''.join(str(b) for b in dQ[:5]))
    
    modulator = Modulator(fc=fc, fs=fs)
    t, s = modulator.modulate(dI, dQ)
    print("s:", ''.join(str(b) for b in s[:5]))

    fig_time, grid = create_figure(2, 1, figsize=(16, 9))
    TimePlot(
        fig_time, grid, (0, 0),
        t=t,
        signals=[dI, dQ],
        labels=["$dI(t)$", "$dQ(t)$"],
        xlim=TIME_XLIM,
        amp_norm=True,
        colors=[COLOR_I, COLOR_Q],
        title=IQ_COMPONENTS_TITLE,
    ).plot()
    
    TimePlot(
        fig_time, grid, (1, 0),
        t=t,
        signals=[s],
        labels=["$s(t)$"],
        xlim=TIME_XLIM,
        amp_norm=True,
        colors=COLOR_COMBINED,
        title=MODULATED_STREAM_TITLE,
    ).plot()
    
    fig_time.tight_layout()
    save_figure(fig_time, "example_modulator_time.pdf")

    fig_freq, grid = create_figure(2, 2, figsize=(16, 9))
    FrequencyPlot(
        fig_freq, grid, (0, 0),
        fs=fs,
        signal=dI,
        fc=fc,
        labels=["$D_I(f)$"],
        xlim=FREQ_COMPONENTS_XLIM,
        colors=COLOR_I,
        title=I_CHANNEL_TITLE,
    ).plot()

    FrequencyPlot(
        fig_freq, grid, (0, 1),
        fs=fs,
        signal=dQ,
        fc=fc,
        labels=["$D_Q(f)$"],
        xlim=FREQ_COMPONENTS_XLIM,
        colors=COLOR_Q,
        title=Q_CHANNEL_TITLE,
    ).plot()

    FrequencyPlot(
        fig_freq, grid, (1, slice(0, 2)),
        fs=fs,
        signal=s,
        fc=fc,
        labels=["$S(f)$"],
        xlim=FREQ_COMBINED_XLIM,
        colors=COLOR_COMBINED,
        title=MODULATED_STREAM_TITLE,
    ).plot()

    fig_freq.tight_layout()
    save_figure(fig_freq, "example_modulator_freq.pdf")

    fig_const, grid = create_figure(1, 2, figsize=(16, 8))
    PhasePlot(
        fig_const, grid, (0, 0),
        t=t,
        signals=[dI, dQ],
        labels=["Phase $I + jQ$"],
        xlim=PHASE_XLIM,
        colors=[COLOR_COMBINED],
        title=PHASE_TITLE,
    ).plot()

    ConstellationPlot(
        fig_const, grid, (0, 1),
        dI=dI[:20000:5],
        dQ=dQ[:20000:5],
        xlim=CONSTELLATION_XLIM,
        ylim=CONSTELLATION_YLIM,
        colors=[COLOR_COMBINED],
        rms_norm=True,
        title=IQ_CONSTELLATION_TITLE,
    ).plot()

    fig_const.tight_layout()
    save_figure(fig_const, "example_modulator_constellation.pdf")

    fig_portadora, grid = create_figure(1, 2, figsize=(16, 8))
    FrequencyPlot(
        fig_portadora, grid, (0, 0),
        fs=fs,
        signal=s[0:(int(round(carrier_duration * fs)))],
        fc=fc,
        labels=["$S(f)$"],
        xlim=FREQ_MODULATED_XLIM,
        colors=COLOR_COMBINED,
        title=(MODULATED_STREAM_TITLE + " - (0 to " + str(carrier_duration*1000) + " ms)"),
    ).plot()

    FrequencyPlot(
        fig_portadora, grid, (0, 1),
        fs=fs,
        signal=s[(int(round(carrier_duration * fs))):],
        fc=fc,
        labels=["$S(f)$"],
        xlim=FREQ_MODULATED_XLIM,
        colors=COLOR_COMBINED,
        title=MODULATED_STREAM_TITLE,
    ).plot()

    fig_portadora.tight_layout()
    save_figure(fig_portadora, "example_modulator_carrier.pdf")