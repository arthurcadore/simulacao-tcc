# """
# Implementation of a PTT-A3 receiver with its components.
#
# Author: Arthur Cadore
# Date: 16-08-2025
# """

import numpy as np
from .datagram import Datagram
from .modulator import Modulator
from .scrambler import Scrambler
from .encoder import Encoder
from .transmitter import Transmitter
from .noise import NoiseEBN0
from .lowpassfilter import LPF
from .matchedfilter import MatchedFilter
from .sampler import Sampler
from .convolutional import DecoderViterbi
from .synchronizer import Synchronizer
from .channel import Channel
from .plotter import save_figure, create_figure, TimePlot, FrequencyPlot, ImpulseResponsePlot, SampledSignalPlot, BitsPlot, PhasePlot, ConstellationPlot, FrequencyResponsePlot, SincronizationPlot, CorrelationPlot, SymbolsPlot
from .env_vars import *

class Receiver:
    def __init__(self, fs=128_000, Rb=400, fc=None, lpf_cutoff=600, preamble="2BEEEEBF", channel_encode=("nrz", "man"), G=np.array([[0b1111001, 0b1011011]]), output_print=True, output_plot=True):
        r"""
        Class that encapsulates the entire reception process in the ARGOS-3 standard.

        Args:
            fs (int): Sampling frequency in Hz.
            Rb (int): Bit rate in bps.
            fc (int): Carrier frequency in Hz.
            lpf_cutoff (int): Cutoff frequency of the low-pass filter in Hz.
            preamble (str): String of preamble in hex.
            channel_encode (tuple): Tuple with the type of encoding of channels I and Q respectively.
            G (np.ndarray): Generation matrix for convolutional encoding.
            output_print (bool): If `True`, prints the intermediate vectors to the console. 
            output_plot (bool): If `True`, generates and saves the graphs of the intermediate processes.

        Raises:
            ValueError: If the encoding types are not 'nrz' or 'manchester'.

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-2097-CNES (section 3.1 and 3.2)
        </div>
        """

        # Validate the values of channel_encode
        valid_encodings = ["nrz", "man"]
        if channel_encode[0] not in valid_encodings or channel_encode[1] not in valid_encodings:
            raise ValueError("The encoding types must be 'nrz' or 'manchester'.")

        # Attributes
        self.fs = fs
        self.Rb = Rb
        self.fc = fc
        self.lpf_cutoff = lpf_cutoff
        self.output_print = output_print
        self.output_plot = output_plot
        self.preamble = preamble
        self.G = G

        # Fixed Attributes
        self.alpha = 0.8
        self.span = 24
        self.lpf_order = 6
        self.delayI = 0
        self.delayQ = 0
        self.cI_encoder = "nrz"
        self.cQ_encoder = "nrz"

        # Channel encoding types
        self.cI_type = channel_encode[0]
        self.cQ_type = channel_encode[1]
        encoding_params = {
            "nrz": {"format": "RRC", "bits_per_symbol": 1, "Rb_multiplier": 1, "matched": "RRC-Inverted"},
            "man": {"format": "Manchester", "bits_per_symbol": 2, "Rb_multiplier": 2, "matched": "Manchester-Inverted"}
        }

        # Attributes for the I and Q channels
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


        # Submodules
        self.demodulator = Modulator(fc=self.fc, fs=self.fs)
        self.lpf = LPF(cut_off=self.lpf_cutoff, order=self.lpf_order, fs=self.fs, type="butter")
        self.matched_filterI = MatchedFilter(alpha=self.alpha, fs=self.fs, Rb=self.Rb, span=self.span, type="RRC-Inverted", channel="I", bits_per_symbol=1)
        self.matched_filterQ = MatchedFilter(alpha=self.alpha, fs=self.fs, Rb=self.Rb, span=self.span, type="Manchester-Inverted", channel="Q", bits_per_symbol=2)
        self.synchronizerI = Synchronizer(fs=self.fs, Rb=self.Rb, sync_word=self.preamble, sync_window=(0, 0.2))
        self.synchronizerQ = Synchronizer(fs=self.fs, Rb=self.Rb, sync_word=self.preamble, sync_window=(0, 0.2))
        self.samplerI = Sampler(fs=self.fs, Rb=self.Rb, delay=self.delayI)
        self.samplerQ = Sampler(fs=self.fs, Rb=self.Rb, delay=self.delayQ)
        self.decoderI = Encoder("nrz")
        self.decoderQ = Encoder("nrz")
        self.unscrambler = Scrambler()
        self.conv_viterbi = DecoderViterbi(G=self.G)

    def bandpass_demodulate(self, s_prime, t):
        r"""
        Demodulates the received signal $s'(t)$ with noise $r(t)$, returning the signals $dX'_{I}(t)$ and $dY'_{Q}(t)$.

        Args:
            s_prime (np.ndarray): Received signal $s'(t)$ to be demodulated.
            t (np.ndarray): Time vector.

        Returns:
            dX_prime (np.ndarray): Demodulated signal $dX'_{I}(t)$.
            dY_prime (np.ndarray): Demodulated signal $dY'_{Q}(t)$.
        
        Examples:
            - Time Domain Plot Example: ![pageplot](assets/receiver_demodulator_time.svg)
            - Frequency Domain Plot Example: ![pageplot](assets/receiver_demodulator_freq.svg)
        """

        dX_prime, dY_prime = self.demodulator.demodulate(s_prime)

        if self.output_print:
            print("\n ==== DEMODULATOR ==== \n")
            print("dX'(t):", ''.join(map(str, dX_prime[:5])),"...")
            print("dY'(t):", ''.join(map(str, dY_prime[:5])),"...")
            print("\n")

        if self.output_plot:
            fig_time, grid = create_figure(2, 2, figsize=(16, 9))

            TimePlot(
                fig_time, grid, (0, slice(0,2)),
                t=t,
                signals=[s_prime],
                labels=[r"$s'(t) + r(t)$"],
                title=MODULATED_STREAM_TITLE,
                xlim=TIME_XLIM_RECEIVER,
                amp_norm=True,
                colors=COLOR_COMBINED
            ).plot()

            TimePlot(
                fig_time, grid, (1, 0),
                t=t,
                signals=[dX_prime],
                labels=[r"$dX'(t)$"],
                title=I_CHANNEL_TITLE,
                xlim=TIME_XLIM_RECEIVER,
                amp_norm=True,
                colors=[COLOR_I]
            ).plot()

            TimePlot(
                fig_time, grid, (1, 1),
                t=t,
                signals=[dY_prime],
                labels=[r"$dY'(t)$"],
                title=Q_CHANNEL_TITLE,
                xlim=TIME_XLIM_RECEIVER,
                amp_norm=True,
                colors=[COLOR_Q]
            ).plot()


            fig_time.tight_layout()
            save_figure(fig_time, "receiver_demodulator_time.pdf")

            fig_freq, grid = create_figure(3, 1, figsize=(16, 12))

            FrequencyPlot(
                fig_freq, grid, (0, 0),
                fs=self.fs,
                signal=s_prime,
                fc=self.fc,
                labels=[r"$S'(f)$"],
                title=MODULATED_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_COMBINED
            ).plot()

            FrequencyPlot(
                fig_freq, grid, (1, 0),
                fs=self.fs,
                signal=dX_prime,
                fc=self.fc,
                labels=[r"$X_I'(f)$"],
                title=I_CHANNEL_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_I,
            ).plot()

            FrequencyPlot(
                fig_freq, grid, (2, 0),
                fs=self.fs,
                signal=dY_prime,
                fc=self.fc,
                labels=[r"$Y_Q'(f)$"],
                title=Q_CHANNEL_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_Q,
            ).plot()

            fig_freq.tight_layout()
            save_figure(fig_freq, "receiver_demodulator_freq.pdf")

        return dX_prime, dY_prime
    
    def low_pass_filter(self, dX_prime, dY_prime, t):
        r"""
        Applies the low-pass filter (LPF) with impulse response $h(t)$ to the signals $dX'(t)$ and $dY'(t)$, returning the filtered signals $d_{I}'(t)$ and $d_{Q}'(t)$.

        Args:
            dX_prime (np.ndarray): Signal $dX'(t)$ to be filtered.
            dY_prime (np.ndarray): Signal $dY'(t)$ to be filtered.
            t (np.ndarray): Time vector.

        Returns:
            dI_prime (np.ndarray): Signal $d_{I}'(t)$ filtered.
            dQ_prime (np.ndarray): Signal $d_{Q}'(t)$ filtered.

        Examples:
            - Time Domain Plot Example: ![pageplot](assets/receiver_lpf_time.svg)
            - Frequency Domain Plot Example: ![pageplot](assets/receiver_lpf_freq.svg)
        """

        impulse_response, t_impulse = self.lpf.calc_impulse_response()

        dI_prime = self.lpf.apply_filter(dX_prime)
        dQ_prime = self.lpf.apply_filter(dY_prime)

        if self.output_print:
            print("\n ==== LOW-PASS FILTER ==== \n")
            print("dI'(t):", ''.join(map(str, dI_prime[:5])),"...")
            print("dQ'(t):", ''.join(map(str, dQ_prime[:5])),"...")
        
        if self.output_plot:
            fig_signal, grid_signal = create_figure(3, 2, figsize=(16, 12))

            ImpulseResponsePlot(
                fig_signal, grid_signal, (0, slice(0, 2)),
                t_impulse, impulse_response,
                t_unit="ms",
                colors=COLOR_IMPULSE,
                label=r"$h(t)$", 
                xlim=(0, 8),
                amp_norm=True,
                title=LPF_IMPULSE_TITLE
            ).plot()

            TimePlot(
                fig_signal, grid_signal, (1, 0),
                t=t, 
                signals=[dX_prime],
                labels=[r"$dX'(t)$"],  
                title=INPUT_STREAM_TITLE,
                xlim=TIME_XLIM_RECEIVER,
                amp_norm=True,
                colors=COLOR_I
            ).plot()

            TimePlot(
                fig_signal, grid_signal, (1, 1),
                t=t, 
                signals=[dY_prime],
                labels=[r"$dY'(t)$"],
                title=INPUT_STREAM_TITLE,
                xlim=TIME_XLIM_RECEIVER,
                amp_norm=True,
                colors=COLOR_Q
            ).plot()

            TimePlot(
                fig_signal, grid_signal, (2, 0),
                t=t, 
                signals=[dI_prime],
                labels=[r"$d_I'(t)$"],  
                title=OUTPUT_STREAM_TITLE,
                xlim=TIME_XLIM_RECEIVER,
                amp_norm=True,
                colors=COLOR_I
            ).plot()

            TimePlot(
                fig_signal, grid_signal, (2, 1),
                t=t, 
                signals=[dQ_prime],
                labels=[r"$d_Q'(t)$"],
                title=OUTPUT_STREAM_TITLE,
                xlim=TIME_XLIM_RECEIVER,
                amp_norm=True,
                colors=COLOR_Q
            ).plot()

            fig_signal.tight_layout()
            save_figure(fig_signal, "receiver_lpf_time.pdf")

            fig_freq, grid_freq = create_figure(3, 2, figsize=(16, 12))

            FrequencyResponsePlot(
                fig_freq, grid_freq, (0, slice(0, 2)),
                self.lpf.b,
                self.lpf.a,
                self.fs,
                f_cut=self.lpf_cutoff,
                xlim=(0, 3*self.lpf_cutoff),
                title=LPF_FREQ_TITLE
            ).plot()

            FrequencyPlot(
                fig_freq, grid_freq, (1, 0), 
                fs=self.fs,
                signal=dX_prime,
                fc=self.fc,
                labels=[r"$DX'(f)$"],
                title=INPUT_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_I,
            ).plot()

            FrequencyPlot(
                fig_freq, grid_freq, (1, 1), 
                fs=self.fs,
                signal=dY_prime,
                fc=self.fc,
                labels=[r"$DY'(f)$"],
                title=INPUT_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_Q,
            ).plot()

            FrequencyPlot(
                fig_freq, grid_freq, (2, 0), 
                fs=self.fs,
                signal=dI_prime,
                fc=self.fc,
                labels=[r"$D_I'(f)$"],
                title=OUTPUT_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_I,
            ).plot()

            FrequencyPlot(
                fig_freq, grid_freq, (2, 1), 
                fs=self.fs,
                signal=dQ_prime,
                fc=self.fc,
                labels=[r"$D_Q'(f)$"],
                title=OUTPUT_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_Q,
            ).plot()

            fig_freq.tight_layout()
            save_figure(fig_freq, "receiver_lpf_freq.pdf")


        return dI_prime, dQ_prime

    def matched_filter(self, dI_prime, dQ_prime, t):
        r"""
        Applies the matched filter with impulse response $g(-t)$ to the signals $d_{I}'(t)$ and $d_{Q}'(t)$, returning the filtered signals $I'(t)$ and $Q'(t)$.

        Args:
            dI_prime (np.ndarray): Signal $d'_{I}(t)$ to be filtered.
            dQ_prime (np.ndarray): Signal $d'_{Q}(t)$ to be filtered.
            t (np.ndarray): Time vector.

        Returns:
            It_prime (np.ndarray): Signal $I'(t)$ filtered.
            Qt_prime (np.ndarray): Signal $Q'(t)$ filtered.

        Examples:
            - Time Domain Plot Example: ![pageplot](assets/receiver_mf_time.svg)
            - Frequency Domain Plot Example: ![pageplot](assets/receiver_mf_freq.svg)
        """

        It_prime = self.matched_filterI.apply_filter(dI_prime)
        Qt_prime = self.matched_filterQ.apply_filter(dQ_prime)

        if self.output_print:
            print("\n ==== MATCHED FILTER ==== \n")
            print("I'(t):", ''.join(map(str, It_prime[:5])),"...")
            print("Q'(t):", ''.join(map(str, Qt_prime[:5])),"...")

        if self.output_plot:
            fig_matched, grid_matched = create_figure(3, 2, figsize=(16, 12))

            ImpulseResponsePlot(
                fig_matched, grid_matched, (0, 0),
                self.matched_filterI.t_rc, self.matched_filterI.g_inverted,
                t_unit="ms",
                colors=COLOR_IMPULSE,
                label=r"$g(-t)$", 
                xlabel=IMPULSE_X,
                ylabel=IMPULSE_Y,
                xlim=IMPULSE_XLIM_400,
                amp_norm=True,
                title=I_CHANNEL_TITLE
            ).plot()

            ImpulseResponsePlot(
                fig_matched, grid_matched, (0, 1),
                self.matched_filterQ.t_rc, self.matched_filterQ.g_inverted,
                t_unit="ms",
                colors=COLOR_IMPULSE,
                label=r"$g(-t)$", 
                xlabel=IMPULSE_X,
                ylabel=IMPULSE_Y,
                xlim=IMPULSE_XLIM_400,
                amp_norm=True,
                title=Q_CHANNEL_TITLE,
            ).plot()

            TimePlot(
                fig_matched, grid_matched, (1, 0),
                t,
                dI_prime,
                labels=[r"$d_I'(t)$"],
                amp_norm=True,
                xlim=TIME_XLIM_RECEIVER,
                colors=COLOR_I,
                title=INPUT_STREAM_TITLE,
            ).plot()

            TimePlot(
                fig_matched, grid_matched, (1, 1),
                t,
                dQ_prime,
                labels=[r"$d_Q'(t)$"],
                amp_norm=True,
                xlim=TIME_XLIM_RECEIVER,
                colors=COLOR_Q,
                title=INPUT_STREAM_TITLE,
            ).plot()

            TimePlot(
                fig_matched, grid_matched, (2, 0),
                t,
                It_prime,
                labels=[r"$I'(t)$"],
                amp_norm=True,
                xlim=TIME_XLIM_RECEIVER,
                colors=COLOR_I,
                title=OUTPUT_STREAM_TITLE,
            ).plot()

            TimePlot(
                fig_matched, grid_matched, (2, 1),
                t,
                Qt_prime,
                labels=[r"$Q'(t)$"],
                amp_norm=True,
                xlim=TIME_XLIM_RECEIVER,
                colors=COLOR_Q,
                title=OUTPUT_STREAM_TITLE,
            ).plot()

            fig_matched.tight_layout()
            save_figure(fig_matched, "receiver_mf_time.pdf")

            fig_matched_freq, grid_matched_freq = create_figure(3, 2, figsize=(16, 12))

            ImpulseResponsePlot(
                fig_matched_freq, grid_matched_freq, (0, 0),
                self.matched_filterI.t_rc, self.matched_filterI.g_inverted,
                t_unit="ms",
                colors=COLOR_IMPULSE,
                label=r"$g(-t)$", 
                xlabel=IMPULSE_X,
                ylabel=IMPULSE_Y,
                xlim=IMPULSE_XLIM_400,
                amp_norm=True,
                title=I_CHANNEL_TITLE
            ).plot()

            ImpulseResponsePlot(
                fig_matched_freq, grid_matched_freq, (0, 1),
                self.matched_filterQ.t_rc, self.matched_filterQ.g_inverted,
                t_unit="ms",
                colors=COLOR_IMPULSE,
                label=r"$g(-t)$", 
                xlabel=IMPULSE_X,
                ylabel=IMPULSE_Y,
                xlim=IMPULSE_XLIM_400,
                amp_norm=True,
                title=Q_CHANNEL_TITLE
            ).plot()

            FrequencyPlot(
                fig_matched_freq, grid_matched_freq, (1, 0),
                fs=self.fs,
                signal=dI_prime,
                fc=self.fc,
                labels=[r"$d_I'(f)$"],
                title=INPUT_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_I,
            ).plot()

            FrequencyPlot(
                fig_matched_freq, grid_matched_freq, (1, 1),
                fs=self.fs,
                signal=dQ_prime,
                fc=self.fc,
                labels=[r"$d_Q'(f)$"],
                title=INPUT_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_Q,
            ).plot()

            FrequencyPlot(
                fig_matched_freq, grid_matched_freq, (2, 0),
                fs=self.fs,
                signal=It_prime,
                fc=self.fc,
                labels=[r"$I'(f)$"],
                title=OUTPUT_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_I,
            ).plot()

            FrequencyPlot(
                fig_matched_freq, grid_matched_freq, (2, 1),
                fs=self.fs,
                signal=Qt_prime,
                fc=self.fc,
                labels=[r"$Q'(f)$"],
                title=OUTPUT_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_Q,
            ).plot()

            fig_matched_freq.tight_layout()
            save_figure(fig_matched_freq, "receiver_mf_freq.pdf")

        return It_prime, Qt_prime

    def synchronizer(self, It_prime, Qt_prime):
        r"""
        Performs the signal synchronization, returning the synchronized signal.

        Args:
            It_prime (np.ndarray): Signal $I'(t)$ to be synchronized.
            Qt_prime (np.ndarray): Signal $Q'(t)$ to be synchronized.

        Returns:
            delayI (float): Delay of the signal $I'(t)$.
            delayQ (float): Delay of the signal $Q'(t)$.

        Examples:
            - Time Domain Plot Example: ![pageplot](assets/receiver_sync_time.svg)
            - Correlation Module Plot Example: ![pageplot](assets/receiver_sync_corr.svg)
        """

        delayI_min, delayI_max, delayI, corr_vec_I = self.synchronizerI.correlation(It_prime, "I")
        delayQ_min, delayQ_max, delayQ, corr_vec_Q = self.synchronizerQ.correlation(Qt_prime, "Q")

        if self.output_print:
            print("\n ==== SYNCHRONIZER ==== \n")
            print("Delay Q Min  :", delayQ_min)
            print("Delay Q Max  :", delayQ_max)
            print("Delay Q Corr :", delayQ)
            print("Delay I Min  :", delayI_min)
            print("Delay I Max  :", delayI_max)
            print("Delay I Corr :", delayI)

        # NOTE: delayI and delayQ should be equal, so delayI is set to delayQ
        delayI_min, delayI_max, delayI = delayQ_min, delayQ_max, delayQ
        
        if self.output_plot:
            fig_corr, grid_corr = create_figure(1, 1, figsize=(16, 9))
            CorrelationPlot(
                fig_corr, grid_corr, (0, 0),
                corr_vec=corr_vec_Q,  
                fs=self.fs,
                xlim=CORR_XLIM_RECEIVER,
                colors=CORR_PLOT_COLOR,
            ).plot()
            fig_corr.tight_layout()
            
            save_figure(fig_corr, "receiver_sync_corr.pdf")
    
            fig_sync, grid_sync = create_figure(2,1, figsize=(16, 9))

            SincronizationPlot(
                fig_sync, grid_sync, (0,0),
                t= np.arange(len(It_prime)) / self.fs,
                signal=It_prime,
                sync_start=delayI_min,
                sync_end=delayI_max,
                max_corr=delayI,
                title=I_CHANNEL_TITLE,   
                labels=[r"$d_I(t)$"],
                colors=COLOR_I,
                xlim=SYNC_XLIM_RECEIVER,
            ).plot()

            SincronizationPlot(
                fig_sync, grid_sync, (1,0),
                t=np.arange(len(Qt_prime)) / self.fs,
                signal=Qt_prime,
                sync_start=delayQ_min,
                sync_end=delayQ_max,
                max_corr=delayQ,
                title=Q_CHANNEL_TITLE,
                labels=[r"$d_Q(t)$"],
                colors=COLOR_Q,
                xlim=SYNC_XLIM_RECEIVER,
            ).plot()

            fig_sync.tight_layout()
            save_figure(fig_sync, "receiver_sync_time.pdf")

        return delayI_max, delayQ_max

    def sampler(self, It_prime, Qt_prime, t):
        r"""
        Performs the decision (sampling and quantization) of the signals $I'(t)$ and $Q'(t)$, returning the symbol vectors $I'[n]$ and $Q'[n]$.

        Args:
            It_prime (np.ndarray): Signal $I'(t)$, matched filtered.
            Qt_prime (np.ndarray): Signal $Q'(t)$, matched filtered.
            t (np.ndarray): Time vector.

        Returns:
            In_prime (np.ndarray): Symbol vector $I'[n]$ sampled and quantized.
            Qn_prime (np.ndarray): Symbol vector $Q'[n]$ sampled and quantized.
        
        Examples:
            - Time Domain Plot Example: ![pageplot](assets/receiver_sampler_time.svg)
            - Constellation Plot Example: ![pageplot](assets/receiver_sampler_const.svg)  
            - Phase Plot Example: ![pageplot](assets/receiver_sampler_phase.svg)  
        """ 

        s_sampledI = self.samplerI.sample(It_prime)
        t_sampledI = self.samplerI.sample(t)
        In_prime = self.samplerI.quantize(s_sampledI)

        s_sampledQ = self.samplerQ.sample(Qt_prime)
        t_sampledQ = self.samplerQ.sample(t)
        Qn_prime = self.samplerQ.quantize(s_sampledQ)

        if self.output_print:
            print("\n ==== SAMPLER ==== \n")
            print("I'n:", ' '.join(f"{x:+d}" for x in In_prime[:20]),"...")
            print("Q'n:", ' '.join(f"{y:+d}" for y in Qn_prime[:20]),"...")

        if self.output_plot:
            fig_sampler, grid_sampler = create_figure(2, 1, figsize=(16, 9))

            SampledSignalPlot(
                fig_sampler, grid_sampler, (0, 0),
                t,
                It_prime,
                t_sampledI,
                s_sampledI,
                colors=COLOR_I,
                label_signal=r"$I'(t)$", 
                label_samples=r"Samples $I'[n]$", 
                xlim=SYNC_XLIM_RECEIVER,
                title=I_CHANNEL_TITLE
            ).plot()

            SampledSignalPlot(
                fig_sampler, grid_sampler, (1, 0),
                t,
                Qt_prime,
                t_sampledQ,
                s_sampledQ,
                colors=COLOR_Q,
                label_signal=r"$Q'(t)$", 
                label_samples=r"Samples $Q'[n]$", 
                xlim=SYNC_XLIM_RECEIVER,
                title=Q_CHANNEL_TITLE
            ).plot()

            fig_sampler.tight_layout()
            save_figure(fig_sampler, "receiver_sampler_time.pdf")            

            fig_const, grid_const = create_figure(1, 2, figsize=(16, 9))

            ConstellationPlot(
                fig_const, grid_const, (0, 0),
                dI=It_prime[:40000:5],
                dQ=Qt_prime[:40000:5],
                xlim=CONSTELLATION_XLIM,
                ylim=CONSTELLATION_YLIM,
                title=IQ_CONSTELLATION_TITLE,
                colors=COLOR_COMBINED,
            ).plot() 

            ConstellationPlot(
                fig_const, grid_const, (0, 1),
                dI=s_sampledI,
                dQ=s_sampledQ,
                xlim=CONSTELLATION_XLIM,
                ylim=CONSTELLATION_YLIM,
                title=IQ_CONSTELLATION_TITLE,
                colors=COLOR_COMBINED,
            ).plot() 

            fig_const.tight_layout()
            save_figure(fig_const, "receiver_sampler_const.pdf")

            fig_phase, grid_phase = create_figure(1, 2, figsize=(16, 9))

            PhasePlot(
                fig_phase, grid_phase, (0, 0),
                t=t,
                signals=[It_prime, Qt_prime],
                labels=[r"Phase $I + jQ$"],
                title=PHASE_TITLE,
                xlim=PHASE_XLIM,
                colors=COLOR_COMBINED,
            ).plot()

            PhasePlot(
                fig_phase, grid_phase, (0, 1),
                t=t_sampledI,
                signals=[np.array(In_prime), np.array(Qn_prime)],
                labels=[r"Phase $I + jQ$"],
                title=PHASE_TITLE,
                xlim=PHASE_XLIM,
                colors=COLOR_COMBINED,
            ).plot()

            fig_phase.tight_layout()
            save_figure(fig_phase, "receiver_sampler_phase.pdf")

        return In_prime, Qn_prime

    def line_decoder(self, In_prime, Qn_prime):
        r"""
        Decodes the symbol vectors $I'[n]$ and $Q'[n]$, returning the bit vectors $X'[n]$ and $Y'[n]$.

        Args:
            In_prime (np.ndarray): Symbol vector $I'[n]$ quantized.
            Qn_prime (np.ndarray): Symbol vector $Q'[n]$ quantized.

        Returns:
            Xn_prime (np.ndarray): Bit vector $X'[n]$ line decoded.
            Yn_prime (np.ndarray): Bit vector $Y'[n]$ line decoded.
        
        Examples:
            - Time Domain Plot Example: ![pageplot](assets/receiver_decoder_time.svg)
        """

        i_quantized = np.array(In_prime)
        q_quantized = np.array(Qn_prime)
        
        Xn_prime = self.decoderI.decode(i_quantized)
        Yn_prime = self.decoderQ.decode(q_quantized)

        if self.output_print:
            print("\n ==== CHANNEL DECODER ==== \n")
            print("X'[n]:", ''.join(map(str, Xn_prime)))
            print("Y'[n]:", ''.join(map(str, Yn_prime)))
        
        if self.output_plot:
            fig_decoder, grid_decoder = create_figure(4, 1, figsize=(16, 12))

            SymbolsPlot(
                fig_decoder, grid_decoder, (0, 0),
                symbols_list=[In_prime],
                samples_per_symbol=1,
                colors=[COLOR_I],
                xlabel=SYMBOLS_X,
                ylabel=SYMBOLS_Y,
                xlim=SYMBOLS_XLIM,
                title=I_CHANNEL_TITLE,
                label=r"$I'[n]$"
            ).plot()

            BitsPlot(
                fig_decoder, grid_decoder, (1, 0),
                bits_list=[Xn_prime],
                sections=[("$X_n$", len(Xn_prime))],
                colors=[COLOR_I],
                xlabel=BITSTREAM_X,
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
                label=r"$X'[n]$"
            ).plot()

            SymbolsPlot(
                fig_decoder, grid_decoder, (2, 0),
                symbols_list=[Qn_prime],
                samples_per_symbol=1,
                colors=[COLOR_Q],
                xlabel=SYMBOLS_X,
                ylabel=SYMBOLS_Y,
                xlim=SYMBOLS_XLIM,
                title=Q_CHANNEL_TITLE,
                label=r"$Q'[n]$"
            ).plot()

            BitsPlot(
                fig_decoder, grid_decoder, (3, 0),
                bits_list=[Yn_prime],
                sections=[("$Y_n$", len(Yn_prime))],
                colors=[COLOR_Q],
                xlabel=BITSTREAM_X,
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
                label=r"$Y'[n]$",
            ).plot()

            fig_decoder.tight_layout()
            save_figure(fig_decoder, "receiver_decoder_time.pdf")
                 
        return Xn_prime, Yn_prime

    def unscramble(self, Xn_prime, Yn_prime):
        r"""
        Unscrambles the bit vectors $X'[n]$ and $Y'[n]$, returning the bit vectors $v_{t}^{(0)} \prime$ and $v_{t}^{(1)} \prime$.

        Args:
            Xn_prime (np.ndarray): Bit vector $X'[n]$ line decoded.
            Yn_prime (np.ndarray): Bit vector $Y'[n]$ line decoded.

        Returns:
            vt0_prime (np.ndarray): Bit vector $v_{t}^{(0)} \prime$ unscrambled.
            vt1_prime (np.ndarray): Bit vector $v_{t}^{(1)} \prime$ unscrambled.

        Examples:
            - Time Domain Plot Example: ![pageplot](assets/receiver_descrambler_time.svg)
        """

        vt0_prime, vt1_prime = self.unscrambler.unscramble(Xn_prime, Yn_prime)

        if self.output_print:
            print("\n ==== UNSCRAMBLER ==== \n")
            print("vt0':", ''.join(map(str, vt0_prime)))
            print("vt1':", ''.join(map(str, vt1_prime)))
        
        if self.output_plot:
            fig_descrambler, grid_descrambler = create_figure(4, 1, figsize=(16, 12))

            BitsPlot(
                fig_descrambler, grid_descrambler, (0, 0),
                bits_list=[Xn_prime],
                sections=[(r"$X'[n]$", len(Xn_prime))],
                colors=[COLOR_I],
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
                title=I_CHANNEL_TITLE,
            ).plot()

            BitsPlot(
                fig_descrambler, grid_descrambler, (1, 0),
                bits_list=[vt0_prime],
                sections=[(r"$v_t^{(0)} \prime$", len(vt0_prime))],
                colors=[COLOR_I],
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
            ).plot()

            BitsPlot(
                fig_descrambler, grid_descrambler, (2, 0),
                bits_list=[Yn_prime],
                sections=[(r"$Y'[n]$", len(Yn_prime))],
                colors=[COLOR_Q],
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
                title=Q_CHANNEL_TITLE,
            ).plot()

            BitsPlot(
                fig_descrambler, grid_descrambler, (3, 0),
                bits_list=[vt1_prime],
                sections=[(r"$v_t^{(1)} \prime$", len(vt1_prime))],
                colors=[COLOR_Q],
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
            ).plot()

            fig_descrambler.tight_layout()
            save_figure(fig_descrambler, "receiver_descrambler_time.pdf")     

        return vt0_prime, vt1_prime

    def conv_decoder(self, vt0_prime, vt1_prime):
        r"""
        Decodes the bit vectors $v_{t}^{(0)} \prime$ and $v_{t}^{(1)} \prime$, returning the bit vector $u_{t}'$.

        Args:
            vt0_prime (np.ndarray): Bit vector $v_{t}^{(0)} \prime$ unscrambled.
            vt1_prime (np.ndarray): Bit vector $v_{t}^{(1)} \prime$ unscrambled.

        Returns:
            ut_prime (np.ndarray): Bit vector $u_{t}'$ decoded.
        
        Examples:
            - Time Domain Plot Example: ![pageplot](assets/receiver_conv_time.svg)
        """

        ut_prime = self.conv_viterbi.decode(vt0_prime, vt1_prime)

        if self.output_print:
            print("\n ==== VITERBI DECODER ==== \n")
            print("ut':", ''.join(map(str, ut_prime)))
        
        if self.output_plot:
            fig_conv_decoder, grid_conv_decoder = create_figure(3, 1, figsize=(16, 9))

            BitsPlot(
                fig_conv_decoder, grid_conv_decoder, (0, 0),
                bits_list=[vt0_prime],
                sections=[(r"$v_t^{(0)} \prime$", len(vt0_prime))],
                colors=[COLOR_I],
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
                title=I_CHANNEL_TITLE,
            ).plot()

            BitsPlot(
                fig_conv_decoder, grid_conv_decoder, (1, 0),
                bits_list=[vt1_prime],
                sections=[(r"$v_t^{(1)} \prime$", len(vt1_prime))],
                colors=[COLOR_Q],
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
                title=Q_CHANNEL_TITLE,
            ).plot()

            BitsPlot(
                fig_conv_decoder, grid_conv_decoder, (2, 0),
                bits_list=[ut_prime],
                sections=[(r"$u_t'$", len(ut_prime))],
                colors=[COLOR_COMBINED],
                ylabel=BITSTREAM_Y,
                xlabel=BITSTREAM_X,
                xlim=BITSTREAM_XLIM,
                title=OUTPUT_STREAM_TITLE,
            ).plot()

            fig_conv_decoder.tight_layout()
            save_figure(fig_conv_decoder, "receiver_conv_time.pdf")     

        return ut_prime

    
    def datagram_parser(self, ut_prime):
        r"""
        Receives a decoded bit vector $u_{t}'$ and returns a datagram in the ARGOS-3 format, or the bit vector $u_{t}'$ if there is an error.

        Args:
            ut_prime (np.ndarray): Bit vector $u_{t}'$ decoded.

        Returns:
            datagram (np.ndarray): Datagram generated, or the bit vector $u_{t}'$ if there is an error.
            success (bool): Indicates if the operation was successful.

        Examples:
            - Time Domain Plot Example: ![pageplot](assets/receiver_datagram_time.svg)
        """
        try:
            datagramRX = Datagram(streambits=ut_prime)

            if self.output_print:
                print("\n ==== DATAGRAM PARSING ==== \n")
                print("\n",datagramRX.parse_datagram())

            if self.output_plot:
                fig_datagram, grid = create_figure(1, 1, figsize=(16, 5))
                BitsPlot(
                    fig_datagram, grid, (0, 0),
                    bits_list=[datagramRX.msglength, 
                               datagramRX.pcdid, 
                               datagramRX.payload, 
                               datagramRX.tail],
                    sections=[("Message Length", len(datagramRX.msglength)),
                              ("PCD ID", len(datagramRX.pcdid)),
                              ("Dados de App.", len(datagramRX.payload)),
                              ("Tail", len(datagramRX.tail))],
                    colors=[COLOR_AUX1, COLOR_AUX2, COLOR_AUX3, COLOR_AUX4],
                    xlabel=BITSTREAM_X,
                    ylabel=BITSTREAM_Y,
                    title=DATAGRAM_STREAM_TITLE,
                ).plot()
                fig_datagram.tight_layout()
                save_figure(fig_datagram, "receiver_datagram_time.pdf")
            
            return datagramRX, True

        except Exception as e:
            print("Erro ao gerar datagrama:", e)
            return ut_prime, False
    
    def receive(self, s_prime):
        r"""
        Receives a signal $s(t)$ and returns the result of the reception.

        Args:
            s_prime (np.ndarray): Signal $s(t)$ received.

        Returns:
            datagramRX (np.ndarray): Datagram generated, or the bit vector $u_{t}'$ if there is an error.
        """

        t = np.arange(0, len(s_prime)/self.fs, 1/self.fs)

        xI_prime, yQ_prime = self.bandpass_demodulate(s_prime, t)
        dI_prime, dQ_prime= self.low_pass_filter(xI_prime, yQ_prime, t)
        It_prime, Qt_prime = self.matched_filter(dI_prime, dQ_prime, t)
        self.delayI, self.delayQ = self.synchronizer(It_prime, Qt_prime)

        # update sampler delay using the delay calculated by the synchronizer
        self.samplerI.update_sampler(self.delayI, t)
        self.samplerQ.update_sampler(self.delayQ, t)

        In_prime, Qn_prime = self.sampler(It_prime, Qt_prime, t)
        Xn_prime, Yn_prime = self.line_decoder(In_prime, Qn_prime)
        vt0_prime, vt1_prime = self.unscramble(Xn_prime, Yn_prime)
        ut_prime = self.conv_decoder(vt0_prime, vt1_prime)

        datagramRX, success = self.datagram_parser(ut_prime)
        return datagramRX, success 


if __name__ == "__main__":

    fc = np.random.randint(10, 90)*100
    print("\n\n==== SIMULATING TRANSMISSION/RECEPTION WITH FC =", fc)

    datagramTX = Datagram(pcdnum=1234, numblocks=1)
    transmitter = Transmitter(fc=fc, output_print=True, output_plot=True)
    t, s = transmitter.transmit(datagramTX)

    print("\n\n ==== CHANNEL ==== \n")    

    channel = Channel(fs=transmitter.fs, duration=0.335, noise_mode="ebn0", noise_db=20, seed=11)
    channel.add_signal(s, position_factor=1)
    channel.add_noise()

    # Comprimentos (para verificação do canal)
    signal_length = len(s) / transmitter.fs
    print("Signal length:", signal_length)

    signalnoise_length = len(channel.channel) / transmitter.fs
    print("Signal noise length:", signalnoise_length)

    noise_first = (signalnoise_length - signal_length)
    print("Noise length before signal:", noise_first)
    
    receiver = Receiver(fc=fc, output_print=True)
    datagramRX, success = receiver.receive(channel.channel)
        
    if not success:
        bitsTX = datagramTX.streambits 
        bitsRX = datagramRX
        print("Bits TX: ", ''.join(str(b) for b in bitsTX))
        print("Bits RX: ", ''.join(str(b) for b in bitsRX))

        # Calculate Bit Error Rate (BER)
        num_errors = sum(1 for tx, rx in zip(bitsTX, bitsRX) if tx != rx)
        ber = num_errors / len(bitsTX)

        print(f"Number of errors: {num_errors}")
        print(f"Bit Error Rate (BER): {ber:.6f}")

