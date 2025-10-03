# """
# Implementation of a PTT-A3 transmitter with its components.
#
# Author: Arthur Cadore
# Date: 16-08-2025
# """

import numpy as np
from .formatter import Formatter
from .convolutional import EncoderConvolutional
from .datagram import Datagram
from .modulator import Modulator
from .preamble import Preamble
from .scrambler import Scrambler
from .multiplexer import Multiplexer
from .encoder import Encoder
from .data import ExportData
from .plotter import create_figure, save_figure, BitsPlot, ImpulseResponsePlot, TimePlot, FrequencyPlot, ConstellationPlot, PhasePlot, SymbolsPlot
from .env_vars import *

class Transmitter:
    def __init__(self, fc=4000, fs=128_000, Rb=400, carrier_length=0.082, preamble="2BEEEEBF", channel_encode=("nrz", "man"), G=np.array([[0b1111001, 0b1011011]]), output_print=True, output_plot=True):
        r"""
        Encapsulates the entire transmission process in the PTT-A3 standard.
    
        Args:
            fc (float): Carrier frequency in Hz. 
            fs (float): Sampling frequency in Hz. 
            Rb (float): Bit rate in bps.
            carrier_length (float): Prefix duration in seconds.
            preamble (str): Preamble string in hex.
            channel_encode (tuple): Tuple with the type of encoding for channels I and Q respectively.
            G (np.ndarray): Generation matrix for convolutional encoding.
            output_print (bool): If `True`, prints intermediate vectors to the console.
            output_plot (bool): If `True`, generates and saves the graphs of the intermediate processes.

        Raises:
            ValueError: If the sampling frequency is less than or equal to zero.
            ValueError: If the bit rate is less than or equal to zero.
            ValueError: If the carrier length is less than or equal to zero.
            ValueError: If the preamble is empty.
            ValueError: If the channel encoding types are not 'nrz' or 'manchester'.

        Examples: 
            >>> import argos3
            >>> import numpy as np 
            >>> 
            >>> fc = np.random.randint(10,80)*100
            >>> print(fc)
            2400
            >>>
            >>> transmitter = argos3.Transmitter(fc=fc, output_print=False, output_plot=False)
            >>> t, s = transmitter.transmit(argos3.Datagram(pcdnum=1234, numblocks=1))
            >>> 
            >>> receiver = argos3.Receiver(fc=fc, output_print=False, output_plot=False)
            >>> datagramRX, success = receiver.receive(s)
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

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (sections 3.1 and 3.2)
        </div>
        """

        # Validar os valores de channel_encode
        valid_encodings = ["nrz", "man"]
        if channel_encode[0] not in valid_encodings or channel_encode[1] not in valid_encodings:
            raise ValueError("The encoding types must be 'nrz' or 'manchester'.")

        # Parameters
        self.fc = fc
        self.fs = fs
        self.Rb = Rb
        self.output_print = output_print
        self.output_plot = output_plot
        self.prefix_duration = carrier_length

        # Fixed parameters
        self.alpha = 0.8
        self.span = 12
        self.cI_encoder = "nrz"
        self.cQ_encoder = "nrz"

        # Channel encoding
        self.cI_type = channel_encode[0]
        self.cQ_type = channel_encode[1]

        # Encoding configurations
        encoding_params = {
            "nrz": {"format": "RRC", "bits_per_symbol": 1, "Rb_multiplier": 1},
            "man": {"format": "Manchester", "bits_per_symbol": 2, "Rb_multiplier": 2}
        }

        # Channel I and Q parameters
        cI_params = encoding_params[self.cI_type]
        self.cI_format = cI_params["format"]
        self.cI_bits_per_symbol = cI_params["bits_per_symbol"]
        self.cI_Rb = self.Rb
        cQ_params = encoding_params[self.cQ_type]
        self.cQ_format = cQ_params["format"]
        self.cQ_bits_per_symbol = cQ_params["bits_per_symbol"]
        self.cQ_Rb = self.Rb

        # Submodules
        self.encoder = EncoderConvolutional(G=G)
        self.scrambler = Scrambler()
        self.preamble = Preamble(preamble_hex=preamble)
        self.multiplexer = Multiplexer()
        self.c_encoderI = Encoder(self.cI_encoder)
        self.c_encoderQ = Encoder(self.cQ_encoder)
        self.formatterI = Formatter(fs=self.fs, Rb=self.cI_Rb, type=self.cI_format, channel="I", bits_per_symbol=self.cI_bits_per_symbol, prefix_duration=self.prefix_duration, alpha=self.alpha, span=self.span)
        self.formatterQ = Formatter(fs=self.fs, Rb=self.cQ_Rb, type=self.cQ_format, channel="Q", bits_per_symbol=self.cQ_bits_per_symbol, prefix_duration=self.prefix_duration, alpha=self.alpha, span=self.span)
        self.modulator = Modulator(fc=self.fc, fs=self.fs)

    def datagram_build(self, datagram: Datagram):
        r"""
        Prepares the datagram for transmission, returning the bit vector $u_t$.

        Returns:
            ut (np.ndarray): Bit vector of the datagram.

        Examples:
            - Bitstream Plot Example: ![pageplot](assets/transmitter_datagram_time.svg)
        """

        ut = datagram.streambits

        if self.output_print:
            print("\n ==== DATAGRAM BUILDING ==== \n")
            print(datagram.parse_datagram())
            print("\nut:", ''.join(map(str, ut)))

        if self.output_plot:
            fig_datagram, grid = create_figure(1, 1, figsize=(16, 5))

            BitsPlot(
                fig_datagram, grid, (0, 0),
                bits_list=[datagram.msglength, 
                           datagram.pcdid, 
                           datagram.payload, 
                           datagram.tail],
                sections=[("Message Length", len(datagram.msglength)),
                          ("PCD ID", len(datagram.pcdid)),
                          ("Payload", len(datagram.payload)),
                          ("Tail", len(datagram.tail))],
                colors=[COLOR_AUX1, COLOR_AUX2, COLOR_AUX3, COLOR_AUX4],
                xlabel=BITSTREAM_X,
                ylabel=BITSTREAM_Y,
                title="Datagram Stream"
            ).plot()

            fig_datagram.tight_layout()
            save_figure(fig_datagram, "transmitter_datagram_time.pdf")

        return ut

    def conv_encoder(self, ut):
        r"""
        Encodes the bit vector $u_t$ using convolutional encoding, returning the bit vectors $v_t^{(0)}$ and $v_t^{(1)}$.

        Args:
            ut (np.ndarray): Bit vector to be encoded.

        Returns:
            vt0 (np.ndarray): Bit vector of channel I.
            vt1 (np.ndarray): Bit vector of channel Q.

        Examples:
            - Bitstream Plot Example: ![pageplot](assets/transmitter_conv_time.svg)
        """

        vt0, vt1 = self.encoder.encode(ut)

        if self.output_print:
            print("\n ==== CONVOLUTIONAL ENCODER ==== \n")
            print("vt0:", ''.join(map(str, vt0)))
            print("vt1:", ''.join(map(str, vt1)))

        if self.output_plot:
            fig_conv, grid_conv = create_figure(3, 1, figsize=(16, 9))

            BitsPlot(
                fig_conv, grid_conv, (0, 0),
                bits_list=[ut],
                sections=[(r"$u_t$", len(ut))],
                colors=[COLOR_COMBINED],
                title=INPUT_STREAM_TITLE,
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM
            ).plot()

            BitsPlot(
                fig_conv, grid_conv, (1, 0),
                bits_list=[vt0],
                sections=[(r"$v_t^{(0)}$", len(vt0))],
                colors=[COLOR_I],
                title=I_CHANNEL_TITLE,
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM
            ).plot()

            BitsPlot(
                fig_conv, grid_conv, (2, 0),
                bits_list=[vt1],
                sections=[(r"$v_t^{(1)}$", len(vt1))],
                colors=[COLOR_Q],
                title=Q_CHANNEL_TITLE,
                xlabel=BITSTREAM_X,
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM
            ).plot()

            fig_conv.tight_layout()
            save_figure(fig_conv, "transmitter_conv_time.pdf")       
        return vt0, vt1

    def scramble(self, vt0, vt1):
        r"""
        Scrambles the bit vectors $v_t^{(0)}$ and $v_t^{(1)}$, creating the shuffled vectors $X[n]$ and $Y[n]$.

        Args:
            vt0 (np.ndarray): Bit vector of channel I.
            vt1 (np.ndarray): Bit vector of channel Q.

        Returns:
            Xn (np.ndarray): Scrambled bit vector of channel I.
            Yn (np.ndarray): Scrambled bit vector of channel Q.

        Examples:
            - Bitstream Plot Example: ![pageplot](assets/transmitter_scrambler_time.svg)
        """

        X, Y = self.scrambler.scramble(vt0, vt1)

        if self.output_print:
            print("\n ==== SCRAMBLER ==== \n")
            print("Xn:", ''.join(map(str, X)))
            print("Yn:", ''.join(map(str, Y)))
            
        if self.output_plot:
            fig_scrambler, grid_scrambler = create_figure(4, 1, figsize=(16, 12))

            BitsPlot(
                fig_scrambler, grid_scrambler, (0, 0),
                bits_list=[vt0],
                sections=[(r"$v_t^{(0)}$", len(vt0))],
                colors=[COLOR_I],
                ylabel=BITSTREAM_Y,
                title=I_CHANNEL_TITLE,
                xlim=BITSTREAM_XLIM
            ).plot()

            BitsPlot(
                fig_scrambler, grid_scrambler, (1, 0),
                bits_list=[X],
                sections=[(r"$X[n]$", len(X))],
                colors=[COLOR_I],
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM
            ).plot()

            BitsPlot(
                fig_scrambler, grid_scrambler, (2, 0),
                bits_list=[vt1],
                sections=[(r"$v_t^{(1)}$", len(vt1))],
                colors=[COLOR_Q],
                xlim=BITSTREAM_XLIM,
                ylabel=BITSTREAM_Y,
                title=Q_CHANNEL_TITLE
            ).plot()

            BitsPlot(
                fig_scrambler, grid_scrambler, (3, 0),
                bits_list=[Y],
                sections=[(r"$Y[n]$", len(Y))],
                colors=[COLOR_Q], 
                ylabel=BITSTREAM_Y, 
                xlabel=BITSTREAM_X,
                xlim=BITSTREAM_XLIM,
            ).plot()

            fig_scrambler.tight_layout()
            save_figure(fig_scrambler, "transmitter_scrambler_time.pdf")

        return X, Y

    def preamble_build(self):
        r"""
        Generates the preamble vectors $S_I[n]$ and $S_Q[n]$.

        Returns:
            sI (np.ndarray): Preamble vector of channel I.
            sQ (np.ndarray): Preamble vector of channel Q.

        Examples:
            - Bitstream Plot Example: ![pageplot](assets/transmitter_preamble_time.svg)
        """
        
        sI, sQ = self.preamble.generate_preamble()

        if self.output_print:
            print("\n ==== PREAMBLE BUILDING ==== \n")
            print("sI:", ''.join(map(str, sI)))
            print("sQ:", ''.join(map(str, sQ)))

        if self.output_plot:
            fig_preamble, grid_preamble = create_figure(2, 1, figsize=(16, 6))

            BitsPlot(
                fig_preamble, grid_preamble, (0,0),
                bits_list=[sI],
                sections=[(r"$S_I[n]$", len(sI))],
                colors=[COLOR_I],
                ylabel=BITSTREAM_Y,
                title=I_CHANNEL_TITLE
            ).plot()
            
            BitsPlot(
                fig_preamble, grid_preamble, (1,0),
                bits_list=[sQ],
                sections=[(r"$S_Q[n]$", len(sQ))],
                colors=[COLOR_Q], 
                xlabel=BITSTREAM_X, 
                ylabel=BITSTREAM_Y,
                title=Q_CHANNEL_TITLE
            ).plot()

            fig_preamble.tight_layout()
            save_figure(fig_preamble, "transmitter_preamble_time.pdf")

        return sI, sQ

    def mux(self, sI, sQ, X, Y):
        r"""
        Multiplexes the preamble vectors $S_I[n]$ and $S_Q[n]$ with the data vectors $X[n]$ and $Y[n]$, returning the multiplexed vectors $X[n]$ and $Y[n]$.

        Args:
            sI (np.ndarray): Preamble vector of channel I.
            sQ (np.ndarray): Preamble vector of channel Q.
            Xn (np.ndarray): Data vector of channel I.
            Yn (np.ndarray): Data vector of channel Q.
        
        Returns:
            Xn (np.ndarray): Multiplexed vector of channel I.
            Yn (np.ndarray): Multiplexed vector of channel Q.

        Examples:
            - Bitstream Plot Example: ![pageplot](assets/transmitter_mux_time.svg)
        """

        Xn, Yn = self.multiplexer.concatenate(sI, sQ, X, Y)

        if self.output_print:
            print("\n ==== MULTIPLEXER ==== \n")
            print("X[n]:", ''.join(map(str, Xn)))
            print("Yn:", ''.join(map(str, Yn)))

        if self.output_plot:
            fig_mux, grid_mux = create_figure(2, 1, figsize=(16, 9))

            BitsPlot(
                fig_mux, grid_mux, (0,0),
                bits_list=[sI, X],
                sections=[(r"$S_I[n]$", len(sI)),(r"$X[n]$", len(X))],
                colors=[COLOR_AUX1, COLOR_I],
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
                title=I_CHANNEL_TITLE
            ).plot()

            BitsPlot(
                fig_mux, grid_mux, (1,0),
                bits_list=[sQ, Y],
                sections=[(r"$S_Q[n]$", len(sQ)),(r"$Y[n]$", len(Y))],
                colors=[COLOR_AUX1, COLOR_Q],
                xlabel=BITSTREAM_X, 
                ylabel=BITSTREAM_Y,
                xlim=BITSTREAM_XLIM,
                title=Q_CHANNEL_TITLE
            ).plot()

            fig_mux.tight_layout()
            save_figure(fig_mux, "transmitter_mux_time.pdf")   
        return Xn, Yn

    def line_encoder(self, Xn, Yn):
        r"""
        Encodes the bit vectors $X[n]$ and $Y[n]$ using line coding ($NRZ$), returning the encoded symbol vectors $I[n]$ and $Q[n]$.

        Args:
            Xn (np.ndarray): Bit vector of channel I to be encoded.
            Yn (np.ndarray): Bit vector of channel Q to be encoded.
        
        Returns:
            In (np.ndarray): Encoded symbol vector of channel I. 
            Qn (np.ndarray): Encoded symbol vector of channel Q. 

        Examples:
            - Signal Plot Example: ![pageplot](assets/transmitter_encoder_time.svg)
        """

        In = self.c_encoderI.encode(Xn)
        Qn = self.c_encoderQ.encode(Yn)

        if self.output_print:
            print("\n ==== CODING CHANNELS ==== \n")
            print("In:", ' '.join(f"{x:+d}" for x in In[:40]),"...")
            print("Qn:", ' '.join(f"{y:+d}" for y in Qn[:40]),"...")

        if self.output_plot:
            fig_encoder, grid = create_figure(4, 1, figsize=(16, 12))

            BitsPlot(
                fig_encoder, grid, (0, 0),
                bits_list=[Xn],
                sections=[(r"$X[n]$", len(Xn))],
                colors=[COLOR_I],
                xlabel=BITSTREAM_X, 
                ylabel=BITSTREAM_Y, 
                xlim=BITSTREAM_XLIM,
                title=I_CHANNEL_TITLE
            ).plot()

            SymbolsPlot(
                fig_encoder, grid, (1, 0),
                symbols_list=[In],
                samples_per_symbol=1,
                colors=[COLOR_I],
                xlabel=SYMBOLS_X,
                ylabel=SYMBOLS_Y,
                xlim=SYMBOLS_XLIM,
                label=r"$I[n]$"
            ).plot()

            BitsPlot(
                fig_encoder, grid, (2, 0),
                bits_list=[Yn],
                sections=[(r"$Y[n]$", len(Yn))],
                colors=[COLOR_Q],
                xlabel=BITSTREAM_X, 
                ylabel=BITSTREAM_Y, 
                xlim=BITSTREAM_XLIM,
                title=Q_CHANNEL_TITLE
            ).plot()

            SymbolsPlot(
                fig_encoder, grid, (3, 0),
                symbols_list=[Qn],
                samples_per_symbol=1,
                colors=[COLOR_Q],
                xlabel=SYMBOLS_X,
                ylabel=SYMBOLS_Y,
                xlim=SYMBOLS_XLIM,
                label=r"$Q[n]$"
            ).plot()

            fig_encoder.tight_layout()
            save_figure(fig_encoder, "transmitter_encoder_time.pdf")

        return In, Qn

    def pulse_modulate(self, In, Qn):
        r"""
        Formats the line coded $NRZ$ symbol vectors $I[n]$ and $Q[n]$ using RRC/Manchester filters, returning the formatted vectors $d_I(t)$ and $d_Q(t)$.

        Args:
            In (np.ndarray): Signal vector of channel $I[n]$ to be formatted.
            Qn (np.ndarray): Signal vector of channel $Q[n]$ to be formatted.
        
        Returns:
            dI (np.ndarray): Formatted vector of channel I.
            dQ (np.ndarray): Formatted vector of channel Q.

        Examples:
            - Time Domain Plot Example: ![pageplot](assets/transmitter_formatter_time.svg)
            - Frequency Domain Plot Example: ![pageplot](assets/transmitter_formatter_freq.svg)
        """

        dI = self.formatterI.apply_format(In)
        dQ = self.formatterQ.apply_format(Qn)
        
        if self.output_print:
            print("\n ==== PULSE MODULATE ==== \n")
            print("dI(t):", ''.join(map(str, dI[:5])),"...")
            print("dQ(t):", ''.join(map(str, dQ[:5])),"...")
            print("Pd:", self.prefix_duration)
            
        if self.output_plot:
            fig_format, grid_format = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_format, grid_format, (0, 0),
                self.formatterI.t_rc, self.formatterI.g,
                t_unit="ms",
                colors=COLOR_IMPULSE,
                label=r"$g(t)$", 
                xlabel=IMPULSE_X, 
                ylabel=IMPULSE_Y, 
                xlim=IMPULSE_XLIM_400,
                amp_norm=True,
                title=I_CHANNEL_TITLE
            ).plot()

            ImpulseResponsePlot(
                fig_format, grid_format, (0, 1),
                self.formatterQ.t_rc, self.formatterQ.g,
                t_unit="ms",
                colors=COLOR_IMPULSE,
                label=r"$g(t)$", 
                xlabel=IMPULSE_X, 
                ylabel=IMPULSE_Y, 
                xlim=IMPULSE_XLIM_400,
                amp_norm=True,
                title=Q_CHANNEL_TITLE
            ).plot()

            TimePlot(
                fig_format, grid_format, (1,0),
                t= np.arange(len(dI)) / self.formatterI.fs,
                signals=[dI],
                labels=[r"$d_I(t)$"],
                amp_norm=True,
                xlim=TIME_XLIM,
                colors=COLOR_I
            ).plot()

            TimePlot(
                fig_format, grid_format, (1,1),
                t= np.arange(len(dQ)) / self.formatterQ.fs,
                signals=[dQ],
                labels=[r"$d_Q(t)$"],
                xlim=TIME_XLIM,
                amp_norm=True,
                colors=COLOR_Q,
            ).plot()

            fig_format.tight_layout()
            save_figure(fig_format, "transmitter_formatter_time.pdf")

            fig_format_freq, grid_format_freq = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_format_freq, grid_format_freq, (0, 0),
                self.formatterI.t_rc, self.formatterI.g,
                t_unit="ms",
                colors=COLOR_IMPULSE,
                label=r"$g(t)$", 
                xlabel=IMPULSE_X, 
                ylabel=IMPULSE_Y, 
                xlim=IMPULSE_XLIM_400, 
                amp_norm=True,
                title=I_CHANNEL_TITLE
            ).plot()

            ImpulseResponsePlot(
                fig_format_freq, grid_format_freq, (0, 1),
                self.formatterQ.t_rc, self.formatterQ.g,
                t_unit="ms",
                colors=COLOR_IMPULSE,
                label=r"$g(t)$", 
                xlabel=IMPULSE_X, 
                ylabel=IMPULSE_Y, 
                xlim=IMPULSE_XLIM_400, 
                amp_norm=True,
                title=Q_CHANNEL_TITLE
            ).plot()

            FrequencyPlot(
                fig_format_freq, grid_format_freq, (1, 0),
                fs=self.fs,
                signal=dI,
                fc=self.fc,
                labels=[r"$D_I(f)$"],
                xlim=FREQ_COMPONENTS_XLIM,
                colors=COLOR_I,
            ).plot()

            FrequencyPlot(
                fig_format_freq, grid_format_freq, (1, 1),
                fs=self.fs,
                signal=dQ,
                fc=self.fc,
                labels=[r"$D_Q(f)$"],
                xlim=FREQ_COMPONENTS_XLIM,
                colors=COLOR_Q,
            ).plot()

            fig_format_freq.tight_layout()
            save_figure(fig_format_freq, "transmitter_formatter_freq.pdf")

        return dI, dQ

    def bandpass_modulate(self, dI, dQ):
        r"""
        Modulates the signal vectors $d_I(t)$ and $d_Q(t)$ using QPSK modulation, returning the modulated signal $s(t)$.

        Args:
            dI (np.ndarray): Formatted vector of channel I.
            dQ (np.ndarray): Formatted vector of channel Q.
        
        Returns:
            t (np.ndarray): Time vector $t$.
            s (np.ndarray): Modulated signal $s(t)$.

        Examples:
            - Time Domain Plot Example: ![pageplot](assets/transmitter_modulator_time.svg)
            - Phase/Constellation Plot Example: ![pageplot](assets/transmitter_modulator_constellation.svg)
            - Frequency Domain Plot Example: ![pageplot](assets/transmitter_modulator_freq.svg)
            - Pure Carrier Plot Example: ![pageplot](assets/transmitter_modulator_portadora.svg)
        """

        t, s = self.modulator.modulate(dI, dQ)

        if self.output_print:
            print("\n ==== BANDPASS MODULATE ==== \n")
            print("s(t):", ''.join(map(str, s[:5])),"...")
            print("t:   ", ''.join(map(str, t[:5])),"...")

        if self.output_plot:
            fig_time, grid = create_figure(2, 1, figsize=(16, 9))

            TimePlot(
                fig_time, grid, (0, 0),
                t=t,
                signals=[dI, dQ],
                labels=[r"$d_I(t)$", r"$d_Q(t)$"],
                title=IQ_COMPONENTS_TITLE,
                xlim=TIME_XLIM,
                amp_norm=True,
                colors=[COLOR_I, COLOR_Q],
            ).plot()

            TimePlot(
                fig_time, grid, (1, 0),
                t=t,
                signals=[s],
                labels=["$s(t)$"],
                title=MODULATED_STREAM_TITLE,
                xlim=TIME_XLIM,
                amp_norm=True,
                colors=COLOR_COMBINED,
            ).plot()

            fig_time.tight_layout()
            save_figure(fig_time, "transmitter_modulator_time.pdf")

            fig_freq, grid = create_figure(2, 2, figsize=(16, 9))
            FrequencyPlot(
                fig_freq, grid, (0, 0),
                fs=self.fs,
                signal=dI,
                fc=self.fc,
                labels=[r"$D_I(f)$"],
                title=I_CHANNEL_TITLE,
                xlim=FREQ_COMPONENTS_XLIM,
                colors=COLOR_I,
            ).plot()
        
            FrequencyPlot(
                fig_freq, grid, (0, 1),
                fs=self.fs,
                signal=dQ,
                fc=self.fc,
                labels=[r"$D_Q(f)$"],
                title=Q_CHANNEL_TITLE,
                xlim=FREQ_COMPONENTS_XLIM,
                colors=COLOR_Q,
            ).plot()
        
            FrequencyPlot(
                fig_freq, grid, (1, slice(0, 2)),
                fs=self.fs,
                signal=s,
                fc=self.fc,
                labels=[r"$S(f)$"],
                title=MODULATED_STREAM_TITLE,
                xlim=FREQ_COMBINED_XLIM,
                colors=COLOR_COMBINED,
            ).plot()
        
            fig_freq.tight_layout()
            save_figure(fig_freq, "transmitter_modulator_freq.pdf")

            fig_const, grid = create_figure(1, 2, figsize=(16, 8))
            PhasePlot(
                fig_const, grid, (0, 0),
                t=t,
                signals=[dI, dQ],
                labels=[r"Phase $I + jQ$"],
                title=PHASE_TITLE,
                xlim=PHASE_XLIM,
                colors=[COLOR_COMBINED],
            ).plot()

            ConstellationPlot(
                fig_const, grid, (0, 1),
                dI=dI[:40000:5],
                dQ=dQ[:40000:5],
                xlim=CONSTELLATION_XLIM,
                ylim=CONSTELLATION_YLIM,
                rms_norm=True,
                show_ideal_points=False,
                title=IQ_CONSTELLATION_TITLE,
                colors=COLOR_COMBINED,
            ).plot()

            fig_const.tight_layout()
            save_figure(fig_const, "transmitter_modulator_constellation.pdf") 

            fig_portadora, grid = create_figure(1, 2, figsize=(16, 8))
            FrequencyPlot(
                fig_portadora, grid, (0, 0),
                fs=self.fs,
                signal=s[0:(int(round(self.prefix_duration * self.fs)))],
                fc=self.fc,
                labels=[r"$S(f)$"],
                title=(MODULATED_STREAM_TITLE + " - (0 to " + str(self.prefix_duration*1000) + " ms)"),
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_COMBINED,
            ).plot()

            FrequencyPlot(
                fig_portadora, grid, (0, 1),
                fs=self.fs,
                signal=s[(int(round(self.prefix_duration * self.fs))):],
                fc=self.fc,
                labels=[r"$S(f)$"],
                title=MODULATED_STREAM_TITLE,
                xlim=FREQ_MODULATED_XLIM,
                colors=COLOR_COMBINED,
            ).plot()

            fig_portadora.tight_layout()
            save_figure(fig_portadora, "transmitter_modulator_portadora.pdf")

        return t, s

    def transmit(self, datagram: Datagram):
        r"""
        Executes the entire transmission chain for a datagram, returning the modulated signal $s(t)$ and the time vector $t$.

        Args:
            datagram (Datagram): Instance of the datagram to be transmitted.

        Returns:
            t (np.ndarray): Time vector $t$.
            s (np.ndarray): Modulated signal $s(t)$.
        """
        ut = self.datagram_build(datagram)
        vt0, vt1 = self.conv_encoder(ut)
        X, Y = self.scramble(vt0, vt1)
        sI, sQ = self.preamble_build()
        Xn, Yn = self.mux(sI, sQ, X, Y)
        In, Qn = self.line_encoder(Xn, Yn)
        dI, dQ = self.pulse_modulate(In, Qn)
        t, s = self.bandpass_modulate(dI, dQ)
        return t, s


if __name__ == "__main__":

    # Creates a transmitter instance
    transmitter = Transmitter(output_print=True, output_plot=True)

    datagram1 = Datagram(pcdnum=1234, numblocks=1, seed=10)
    datagram2 = Datagram(pcdnum=1234, numblocks=1, seed=10)

    # Transmits the datagram 1
    t1, s1 = transmitter.transmit(datagram1)
    t2, s2 = transmitter.transmit(datagram2)

    # Verifies if the vectors are equal
    if np.array_equal(s1, s2):
        print("S1 == S2")
    else:
        print("S1 != S2")

    # Export the data
    ExportData([s1, t1], "transmitter_st").save()

