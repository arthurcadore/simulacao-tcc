# """
# Implements a symbol synchronizer to identify the moment of maximum correlation between the received signal and the synchronization signal.
#
# Autor: Arthur Cadore
# Data: 28-07-2025
# """

import numpy as np
import matplotlib.pyplot as plt

from .preamble import Preamble
from .formatter import Formatter
from .encoder import Encoder
from .plotter import create_figure, save_figure, TimePlot, SincronizationPlot, CorrelationPlot
from .multiplexer import Multiplexer
from .matchedfilter import MatchedFilter
from .env_vars import *

class Synchronizer:
    def __init__(self, fs=128_000, Rb=400, sync_word="2BEEEEBF", channel_encode=("nrz", "man"), sync_window=None):
        r"""
        Initializes the symbol synchronizer to identify the moment of maximum correlation between the received signal and the synchronization signal.

        ![pageplot](../assets/sync.svg)

        Args:
            fs (int): Sampling frequency of the received signal.
            Rb (int): Transmission rate of the received signal.
            sync_word (str): Synchronization word.
            channel_encode (tuple): Tuple with the type of encoding of the I and Q channels respectively.

        Examples:
            >>> import argos3
            >>> import numpy as np 
            >>> 
            >>> Si, Sq = argos3.Preamble(preamble_hex="2BEEEEBF").generate_preamble()
            >>> 
            >>> X = np.random.randint(0,2,20)
            >>> Y = np.random.randint(0,2,20)
            >>> X, Y = argos3.Multiplexer().concatenate(Si, Sq, X, Y)
            >>> 
            >>> Xn = argos3.Encoder().encode(X)
            >>> Yn = argos3.Encoder().encode(Y)
            >>> 
            >>> formatterI = argos3.Formatter(type="RRC", channel="I", bits_per_symbol=1)
            >>> formatterQ = argos3.Formatter(type="Manchester", channel="Q", bits_per_symbol=2)
            >>> 
            >>> mfI = argos3.MatchedFilter(type="RRC-Inverted", channel="I", bits_per_symbol=1)
            >>> mfQ = argos3.MatchedFilter(type="Manchester-Inverted", channel="Q", bits_per_symbol=2)
            >>> 
            >>> dI = mfI.apply_filter(formatterI.apply_format(Xn))
            >>> dQ = mfQ.apply_filter(formatterQ.apply_format(Yn))
            >>> 
            >>> sync = argos3.Synchronizer()
            >>> delayQ_min, delayQ_max, delayQ, corr_vec = sync.correlation(dQ, "Q")
            >>> 
            >>> print(delayQ_min)
            0.079984375
            >>> print(delayQ_max)
            0.117484375
            >>> print(delayQ)
            0.098734375
            
            - Time Domain: ![pageplot](assets/example_synchronizer_sync.svg)
        """

        # Validar os valores de channel_encode
        valid_encodings = ["nrz", "man"]
        if channel_encode[0] not in valid_encodings or channel_encode[1] not in valid_encodings:
            raise ValueError("The encoding types must be 'nrz' or 'manchester'.")
        
        # Parameters
        self.fs = fs
        self.Rb = Rb
        self.Tb = 1 / Rb
        self.sps = int(fs / Rb)

        # Fixed parameters
        self.alpha = 0.8
        self.span = 6
        self.cI_encoder = "nrz"
        self.cQ_encoder = "nrz"
        self.sync_window = sync_window

        # Channel encoding
        self.cI_type = channel_encode[0]
        self.cQ_type = channel_encode[1]

        # Encoding configuration mapping
        encoding_params = {
            "nrz": {"format": "RRC", "bits_per_symbol": 1, "Rb_multiplier": 1, "matched": "RRC-Inverted"},
            "man": {"format": "Manchester", "bits_per_symbol": 2, "Rb_multiplier": 2, "matched": "Manchester-Inverted"}
        }

        # Channel parameters
        cI_params = encoding_params[self.cI_type]
        self.cI_format = cI_params["format"]
        self.cI_bits_per_symbol = cI_params["bits_per_symbol"]
        self.cI_Rb = self.Rb
        self.cI_matched = cI_params["matched"]
        cQ_params = encoding_params[self.cQ_type]
        self.cQ_format = cQ_params["format"]
        self.cQ_bits_per_symbol = cQ_params["bits_per_symbol"]
        self.cQ_Rb = self.Rb
        self.cQ_matched = cQ_params["matched"]

        # Build sync world
        self.encoder_I = Encoder(method=self.cI_encoder)
        self.encoder_Q = Encoder(method=self.cQ_encoder)
        self.formatterI = Formatter(alpha=self.alpha, fs=self.fs, Rb=self.cI_Rb, span=self.span, type=self.cI_format, channel="I", bits_per_symbol=self.cI_bits_per_symbol)
        self.formatterQ = Formatter(alpha=self.alpha, fs=self.fs, Rb=self.cQ_Rb, span=self.span, type=self.cQ_format, channel="Q", bits_per_symbol=self.cQ_bits_per_symbol)
        self.matched_filter_I = MatchedFilter(alpha=self.alpha, fs=self.fs, Rb=self.cI_Rb, span=self.span, type=self.cI_matched, channel="I", bits_per_symbol=self.cI_bits_per_symbol)
        self.matched_filter_Q = MatchedFilter(alpha=self.alpha, fs=self.fs, Rb=self.cQ_Rb, span=self.span, type=self.cQ_matched, channel="Q", bits_per_symbol=self.cQ_bits_per_symbol)
        self.create_sincronized_word(sync_word)

    def create_sincronized_word(self, sync_word):
        r"""
        Creates the vectors of symbol $S_I(t)$ and $S_Q(t)$, corresponding to the synchronization word of channel $I$ and $Q$, respectively. The length of the synchronization word is given by $\Delta \tau$, according to the expression below.

        $$
        \Delta \tau = L_{sync} \cdot \frac{f_s}{R_b}
        $$

        Where: 
            - $\Delta \tau$ is the length of the synchronization word.
            - $L_{sync}$ is the length of the synchronization word of $S_I(t)$ and $S_Q(t)$.
            - $R_b$ is the bit rate.
            - $f_s$ is the sampling frequency.

        Args:
            sync_word (str): Synchronization word.

        Examples: 
            - Time Domain: ![pageplot](assets/example_synchronizer_word.svg)
        """

        self.preamble = Preamble(sync_word)
        self.preamble_sI = self.preamble.preamble_sI
        self.preamble_sQ = self.preamble.preamble_sQ

        self.sincronized_word_I = self.formatterI.apply_format(self.encoder_I.encode(self.preamble_sI), add_prefix=False)
        self.sincronized_word_Q = self.formatterQ.apply_format(self.encoder_Q.encode(self.preamble_sQ), add_prefix=False)

        self.sincronized_word_I = self.matched_filter_I.apply_filter(self.sincronized_word_I)
        self.sincronized_word_Q = self.matched_filter_Q.apply_filter(self.sincronized_word_Q)

    def correlation(self, signal, channel):
        r"""

        Performs the cross-correlation between the received signal $s(t)$ and the synchronization word $d(t)$, for each time index $t$.

        $$
        c[k] = \sum_{t=0} s[t] d[t - k]
        $$ 

        Where: 
            - $s(t)$ and $d(t)$ are the symbol vectors of the received signal and the synchronization word, respectively.
            - $k$ is the time index in the cross-correlation vector.
            - $c[k]$ is the cross-correlation value for the index $k$.

        Subsequently, the index of $c[k]$ with the highest value is located, resulting in $k_{max}$, this is the sample index with the highest correlation between the received signal and the synchronization word, finally, the delay $\tau$ is calculated. 

        $$
            \tau = \frac{k_{max}}{f_s}
        $$

        Where: 
            - $\tau$: Delay between the received signal and the synchronization word.
            - $f_s$: Sampling frequency of the received signal.
            - $k_{max}$: Sample index with the highest correlation between the received signal and the synchronization word.

        Args:
            signal (np.ndarray): Received signal.
            channel (str): Channel of reception, $I$ or $Q$.

        Returns:
           delay (tuple): Tuple containing the delay $\tau$, the delay $\tau_{min}$ and the delay $\tau_{max}$.
        
        Examples: 
            - Correlation Factor: ![pageplot](assets/example_synchronizer_corr.svg)
        """
        if channel == "I":
            correlation_vec = np.correlate(signal, self.sincronized_word_I, mode="same")
        elif channel == "Q":
            correlation_vec = np.correlate(signal, self.sincronized_word_Q, mode="same")
        else:
            raise ValueError("Invalid channel. Use 'I' or 'Q'.")


        # converts seconds window to indices
        if self.sync_window is not None:
            t_start, t_end = self.sync_window
            start_idx = int(t_start * self.fs)
            end_idx = int(t_end * self.fs)
            start_idx = max(0, start_idx)
            end_idx = min(len(correlation_vec), end_idx)
        else:
            start_idx, end_idx = 0, len(correlation_vec)        

        local_argmax = correlation_vec[start_idx:end_idx].argmax()
        max_correlation_index = start_idx + local_argmax

        # normalizes the vector
        correlation_vec = (correlation_vec - correlation_vec.min()) / (correlation_vec.max() - correlation_vec.min())
        
        # calculates the index of the start and end of the synchronization word
        low_index = max_correlation_index - len(self.sincronized_word_I) // 2
        high_index = max_correlation_index + len(self.sincronized_word_I) // 2

        # calculates the delay based on the index
        low_delay = low_index / self.fs
        high_delay = high_index / self.fs
        delay = max_correlation_index / self.fs
    
        return low_delay, high_delay, delay, correlation_vec

if __name__ == "__main__":
    
    synchronizer = Synchronizer()

    fig_format, grid_format = create_figure(2,1, figsize=(16, 9))

    TimePlot(
        fig_format, grid_format, (0,0),
        t= np.arange(len(synchronizer.sincronized_word_I)) / synchronizer.formatterI.fs,
        signals=[synchronizer.sincronized_word_I],
        labels=[r"$S_I(t)$"],
        amp_norm=True,
        colors=[COLOR_I],
        title=I_CHANNEL_TITLE,
    ).plot()
    
    TimePlot(
        fig_format, grid_format, (1,0),
        t= np.arange(len(synchronizer.sincronized_word_Q)) / synchronizer.formatterQ.fs,
        signals=[synchronizer.sincronized_word_Q],
        labels=[r"$S_Q(t)$"],
        amp_norm=True,
        colors=[COLOR_Q],
        title=Q_CHANNEL_TITLE,
    ).plot()
    
    fig_format.tight_layout()
    save_figure(fig_format, "example_synchronizer_word.pdf")

    preamble = Preamble()  
    SI = preamble.preamble_sI
    SQ = preamble.preamble_sQ
    X = np.random.randint(0, 2, 60)
    Y = np.random.randint(0, 2, 60)
    
    mux = Multiplexer()
    Xn, Yn = mux.concatenate(SI, SQ, X, Y)

    encoder_nrz = Encoder(method="NRZ")
    encoder_man = Encoder(method="NRZ")
    
    Xnrz = encoder_nrz.encode(Xn)
    Yman = encoder_man.encode(Yn)
    
    formatterI = Formatter(alpha=0.8, fs=128_000, Rb=400, span=12, type="RRC", channel="I", bits_per_symbol=1)
    formatterQ = Formatter(alpha=0.8, fs=128_000, Rb=400, span=12, type="Manchester", channel="Q", bits_per_symbol=2)
    matched_filter_I = MatchedFilter(alpha=0.8, fs=128_000, Rb=400, span=12, type="RRC-Inverted", channel="I", bits_per_symbol=1)
    matched_filter_Q = MatchedFilter(alpha=0.8, fs=128_000, Rb=400, span=12, type="Manchester-Inverted", channel="Q", bits_per_symbol=2)
    
    dI = formatterI.apply_format(Xnrz)
    dQ = formatterQ.apply_format(Yman)
    
    dI = matched_filter_I.apply_filter(dI)
    dQ = matched_filter_Q.apply_filter(dQ)
    
    delayQ_min, delayQ_max, delayQ, corr_vec = synchronizer.correlation(dQ, "Q")
    delayI_min, delayI_max, delayI = delayQ_min, delayQ_max, delayQ

    print("Delay I (ms):", delayI_min)
    print("Delay Q (ms):", delayQ_min)

    fig_sync, grid_sync = create_figure(2,1, figsize=(16, 9))
    
    SincronizationPlot(
        fig_sync, grid_sync, (0,0),
        t= np.arange(len(dI)) / formatterI.fs,
        signal=dI,
        sync_start=delayI_min,
        sync_end=delayI_max,
        max_corr=delayI,
        labels=[r"$d_I(t)$"],
        colors=[COLOR_I],
        xlim=SYNC_XLIM,
        title=I_CHANNEL_TITLE,  
    ).plot()

    SincronizationPlot(
        fig_sync, grid_sync, (1,0),
        t=np.arange(len(dQ)) / formatterQ.fs,
        signal=dQ,
        sync_start=delayQ_min,
        sync_end=delayQ_max,
        max_corr=delayQ,
        labels=[r"$d_Q(t)$"],
        colors=[COLOR_Q],
        xlim=SYNC_XLIM,
        title=Q_CHANNEL_TITLE,
    ).plot()

    fig_sync.tight_layout()
    save_figure(fig_sync, "example_synchronizer_sync.pdf")

    fig_corr, grid_corr = create_figure(1, 1, figsize=(16, 9))
    CorrelationPlot(
        fig_corr, grid_corr, (0, 0),
        corr_vec=corr_vec,  
        fs=formatterQ.fs,
        xlim=SYNC_XLIM,
        colors=[CORR_PLOT_COLOR],
        title=SYNC_CORR_TITLE,
    ).plot()

    fig_corr.tight_layout()
    save_figure(fig_corr, "example_synchronizer_corr.pdf")
    
    
    
        
