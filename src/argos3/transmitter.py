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

    def prepare_datagram(self, datagram: Datagram):
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
                colors=["green", "orange", "red", "blue"],
                xlabel="Bit Index"
            ).plot()

            fig_datagram.tight_layout()
            save_figure(fig_datagram, "transmitter_datagram_time.pdf")

        return ut

    def encode_convolutional(self, ut):
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
                sections=[("$u_t$", len(ut))],
                colors=["darkred"],
                ylabel="$u_t$",
                xlim=(0, 60)
            ).plot()

            BitsPlot(
                fig_conv, grid_conv, (1, 0),
                bits_list=[vt0],
                sections=[("$v_t^{(0)}$", len(vt0))],
                colors=["darkgreen"],
                ylabel="$v_t^{(0)}$",
                xlim=(0, 60)
            ).plot()

            BitsPlot(
                fig_conv, grid_conv, (2, 0),
                bits_list=[vt1],
                sections=[("$v_t^{(1)}$", len(vt1))],
                colors=["navy"],
                ylabel="$v_t^{(1)}$", 
                xlim=(0, 60),
                xlabel="Bit Index"
            ).plot()

            fig_conv.tight_layout()
            save_figure(fig_conv, "transmitter_conv_time.pdf")       
        return vt0, vt1

    def scramble(self, vt0, vt1):
        r"""
        Scrambles the bit vectors $v_t^{(0)}$ and $v_t^{(1)}$, creating the shuffled vectors $X_n$ and $Y_n$.

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
            fig_scrambler, grid_scrambler = create_figure(4, 1, figsize=(16, 9))

            BitsPlot(
                fig_scrambler, grid_scrambler, (0, 0),
                bits_list=[vt0],
                sections=[("$v_t^{0}$", len(vt0))],
                colors=["darkgreen"],
                xlim=(0, 60),
                ylabel="Original"
            ).plot()

            BitsPlot(
                fig_scrambler, grid_scrambler, (1, 0),
                bits_list=[X],
                sections=[("$X_n$", len(X))],
                colors=["darkgreen"],
                ylabel="Scrambled", 
                xlim=(0, 60),
            ).plot()

            BitsPlot(
                fig_scrambler, grid_scrambler, (2, 0),
                bits_list=[vt1],
                sections=[("$v_t^{1}$", len(vt1))],
                colors=["navy"],
                xlim=(0, 60),
                ylabel="Original"
            ).plot()

            BitsPlot(
                fig_scrambler, grid_scrambler, (3, 0),
                bits_list=[Y],
                sections=[("$Y_n$", len(Y))],
                colors=["navy"], 
                ylabel="Scrambled", 
                xlabel="Bit Index",
                xlim=(0, 60),
            ).plot()

            fig_scrambler.tight_layout()
            save_figure(fig_scrambler, "transmitter_scrambler_time.pdf")

        return X, Y

    def generate_preamble(self):
        r"""
        Generates the preamble vectors $S_I$ and $S_Q$.

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
                sections=[("$S_I$", len(sI))],
                colors=["darkgreen"],
                ylabel="Canal $I$"
            ).plot()
            
            BitsPlot(
                fig_preamble, grid_preamble, (1,0),
                bits_list=[sQ],
                sections=[("$S_Q$", len(sQ))],
                colors=["navy"], 
                xlabel="Bit Index", 
                ylabel="Canal $Q$"
            ).plot()

            fig_preamble.tight_layout()
            save_figure(fig_preamble, "transmitter_preamble_time.pdf")

        return sI, sQ

    def multiplex(self, sI, sQ, X, Y):
        r"""
        Multiplexes the preamble vectors $S_I$ and $S_Q$ with the data vectors $X$ and $Y$, returning the multiplexed vectors $X_n$ and $Y_n$.

        Args:
            sI (np.ndarray): Preamble vector of channel I.
            sQ (np.ndarray): Preamble vector of channel Q.
            X (np.ndarray): Data vector of channel I.
            Y (np.ndarray): Data vector of channel Q.
        
        Returns:
            Xn (np.ndarray): Multiplexed vector of channel I.
            Yn (np.ndarray): Multiplexed vector of channel Q.

        Examples:
            - Bitstream Plot Example: ![pageplot](assets/transmitter_mux_time.svg)
        """

        Xn, Yn = self.multiplexer.concatenate(sI, sQ, X, Y)

        if self.output_print:
            print("\n ==== MULTIPLEXER ==== \n")
            print("Xn:", ''.join(map(str, Xn)))
            print("Yn:", ''.join(map(str, Yn)))

        if self.output_plot:
            fig_mux, grid_mux = create_figure(2, 1, figsize=(16, 9))

            BitsPlot(
                fig_mux, grid_mux, (0,0),
                bits_list=[sI, X],
                sections=[("Preambulo $S_I$", len(sI)),
                          ("Canal I $(X_n)$", len(X))],
                colors=["darkred", "darkgreen"],
                ylabel="Canal $I$",
                xlim=(0, 60),
            ).plot()

            BitsPlot(
                fig_mux, grid_mux, (1,0),
                bits_list=[sQ, Y],
                sections=[("Preambulo $S_Q$", len(sQ)),
                          ("Canal Q $(Y_n)$", len(Y))],
                colors=["darkred", "navy"],
                xlabel="Index de Bit", 
                ylabel="Canal $Q$",
                xlim=(0, 60),
            ).plot()

            fig_mux.tight_layout()
            save_figure(fig_mux, "transmitter_mux_time.pdf")   
        return Xn, Yn

    def encode_channels(self, Xn, Yn):
        r"""
        Encodes the bit vectors $X_n$ and $Y_n$ using $NRZ$ and $Manchester$, respectively, returning the encoded signal vectors $X_{NRZ}$ and $Y_{MAN}$.

        Args:
            Xn (np.ndarray): Bit vector of channel $X_n$ to be encoded.
            Yn (np.ndarray): Bit vector of channel $Y_n$ to be encoded.
        
        Returns:
            Xnrz (np.ndarray): Encoded signal vector of channel I $NRZ$. 
            Yman (np.ndarray): Encoded signal vector of channel Q $Manchester$. 

        Examples:
            - Signal Plot Example: ![pageplot](assets/transmitter_encoder_time.svg)
        """

        Xi = self.c_encoderI.encode(Xn)
        Yq = self.c_encoderQ.encode(Yn)

        if self.output_print:
            print("\n ==== CODING CHANNELS ==== \n")
            print("Xi:", ' '.join(f"{x:+d}" for x in Xi[:40]),"...")
            print("Yq:", ' '.join(f"{y:+d}" for y in Yq[:40]),"...")

        if self.output_plot:
            fig_encoder, grid = create_figure(4, 1, figsize=(16, 9))

            BitsPlot(
                fig_encoder, grid, (0, 0),
                bits_list=[Xn],
                sections=[("$X_n$", len(Xn))],
                colors=["darkgreen"],
                xlabel="Index de Bit", 
                ylabel="$X_n$", 
                xlim=(0, 60)
            ).plot()

            SymbolsPlot(
                fig_encoder, grid, (1, 0),
                symbols_list=[Xi],
                samples_per_symbol=1,
                colors=["darkgreen"],
                xlabel="Index de Simbolo",
                xlim=(0, 60),
                ylabel="$X_{NRZ}[n]$"
            ).plot()

            BitsPlot(
                fig_encoder, grid, (2, 0),
                bits_list=[Yn],
                sections=[("$Y_n$", len(Yn))],
                colors=["navy"],
                xlabel="Index de Bit", 
                ylabel="$Y_n$", 
                xlim=(0, 60),
            ).plot()

            SymbolsPlot(
                fig_encoder, grid, (3, 0),
                symbols_list=[Yq],
                samples_per_symbol=1,
                colors=["navy"],
                xlabel="Index de Simbolo",
                xlim=(0, 60),
                ylabel="$Y_{MAN}[n]$"
            ).plot()

            fig_encoder.tight_layout()
            save_figure(fig_encoder, "transmitter_encoder_time.pdf")

        return Xi, Yq

    def format_signals(self, Xi, Yq):
        r"""
        Formats the encoded signal vectors $X_{NRZ}$ and $Y_{MAN}$ using RRC filter, returning the formatted vectors $d_I$ and $d_Q$.

        Args:
            Xnrz (np.ndarray): Signal vector of channel $X_{NRZ}$ to be formatted.
            Yman (np.ndarray): Signal vector of channel $Y_{MAN}$ to be formatted.
        
        Returns:
            dI (np.ndarray): Formatted vector of channel I, $d_I$.
            dQ (np.ndarray): Formatted vector of channel Q, $d_Q$.

        Examples:
            - Time Domain Plot Example: ![pageplot](assets/transmitter_formatter_time.svg)
            - Frequency Domain Plot Example: ![pageplot](assets/transmitter_formatter_freq.svg)
        """

        dI = self.formatterI.apply_format(Xi)
        dQ = self.formatterQ.apply_format(Yq)
        
        if self.output_print:
            print("\n ==== PULSE MODULATE ==== \n")
            print("dI:", ''.join(map(str, dI[:5])),"...")
            print("dQ:", ''.join(map(str, dQ[:5])),"...")
            print("Prefix Duration:", self.prefix_duration)
            
        if self.output_plot:
            fig_format, grid_format = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_format, grid_format, (0, 0),
                self.formatterI.t_rc, self.formatterI.g,
                t_unit="ms",
                colors="darkorange",
                label="$g(t)$", 
                xlabel=r"Tempo ($ms$)", 
                ylabel="Amplitude", 
                xlim=(-15, 15),
                amp_norm=True,
            ).plot()

            ImpulseResponsePlot(
                fig_format, grid_format, (0, 1),
                self.formatterQ.t_rc, self.formatterQ.g,
                t_unit="ms",
                colors="darkorange",
                label="$g(t)$", 
                xlabel=r"Tempo ($ms$)", 
                ylabel="Amplitude", 
                xlim=(-15, 15),
                amp_norm=True,
            ).plot()

            TimePlot(
                fig_format, grid_format, (1,0),
                t= np.arange(len(dI)) / self.formatterI.fs,
                signals=[dI],
                labels=["$d_I(t)$"],
                title="Canal $I$",
                xlim=(40, 200),
                amp_norm=True,
                colors="darkgreen",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            TimePlot(
                fig_format, grid_format, (1,1),
                t= np.arange(len(dQ)) / self.formatterQ.fs,
                signals=[dQ],
                labels=["$d_Q(t)$"],
                title="Canal $Q$",
                xlim=(40, 200),
                amp_norm=True,
                colors="darkblue",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            fig_format.tight_layout()
            save_figure(fig_format, "transmitter_formatter_time.pdf")

            fig_format_freq, grid_format_freq = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_format_freq, grid_format_freq, (0, 0),
                self.formatterI.t_rc, self.formatterI.g,
                t_unit="ms",
                colors="darkorange",
                label="$g(t)$", 
                xlabel=r"Tempo ($ms$)", 
                ylabel="Amplitude", 
                xlim=(-15, 15), 
                amp_norm=True
            ).plot()

            ImpulseResponsePlot(
                fig_format_freq, grid_format_freq, (0, 1),
                self.formatterQ.t_rc, self.formatterQ.g,
                t_unit="ms",
                colors="darkorange",
                label="$g(t)$", 
                xlabel=r"Tempo ($ms$)", 
                ylabel="Amplitude", 
                xlim=(-15, 15), 
                amp_norm=True
            ).plot()

            FrequencyPlot(
                fig_format_freq, grid_format_freq, (1, 0),
                fs=self.fs,
                signal=dI,
                fc=self.fc,
                labels=["$D_I(f)$"],
                title="Canal $I$",
                xlim=(-1.5, 1.5),
                colors="darkgreen",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_format_freq, grid_format_freq, (1, 1),
                fs=self.fs,
                signal=dQ,
                fc=self.fc,
                labels=["$D_Q(f)$"],
                title="Canal $Q$",
                xlim=(-1.5, 1.5),
                colors="navy",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            fig_format_freq.tight_layout()
            save_figure(fig_format_freq, "transmitter_formatter_freq.pdf")

        return dI, dQ

    def modulate(self, dI, dQ):
        r"""
        Modulates the signal vectors $d_I(t)$ and $d_Q(t)$ using QPSK modulation, returning the modulated signal $s(t)$.

        Args:
            dI (np.ndarray): Formatted vector of channel I, $d_I(t)$.
            dQ (np.ndarray): Formatted vector of channel Q, $d_Q(t)$.
        
        Returns:
            t (np.ndarray): Time vector, $t$.
            s (np.ndarray): Modulated signal, $s(t)$.

        Examples:
            - Time Domain Plot Example: ![pageplot](assets/transmitter_modulator_time.svg)
            - Frequency Domain Plot Example: ![pageplot](assets/transmitter_modulator_freq.svg)
            - Pure Carrier Plot Example: ![pageplot](assets/transmitter_modulator_portadora.svg)
            - Phase/Constellation Plot Example: ![pageplot](assets/transmitter_modulator_constellation.svg)
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
                labels=["$d_I(t)$", "$d_Q(t)$"],
                title="Componentes $IQ$ - Demoduladas",
                xlim=(40, 200),
                amp_norm=True,
                colors=["darkgreen", "navy"],
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            TimePlot(
                fig_time, grid, (1, 0),
                t=t,
                signals=[s],
                labels=["$s(t)$"],
                title="Sinal Modulado $IQ$",
                xlim=(40, 200),
                amp_norm=True,
                colors="darkred",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            fig_time.tight_layout()
            save_figure(fig_time, "transmitter_modulator_time.pdf")

            fig_freq, grid = create_figure(2, 2, figsize=(16, 9))
            FrequencyPlot(
                fig_freq, grid, (0, 0),
                fs=self.fs,
                signal=dI,
                fc=self.fc,
                labels=["$D_I(f)$"],
                title="Componente I",
                xlim=(-1.5, 1.5),
                colors="darkgreen",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()
        
            FrequencyPlot(
                fig_freq, grid, (0, 1),
                fs=self.fs,
                signal=dQ,
                fc=self.fc,
                labels=["$D_Q(f)$"],
                title="Componente Q",
                xlim=(-1.5, 1.5),
                colors="navy",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()
        
            FrequencyPlot(
                fig_freq, grid, (1, slice(0, 2)),
                fs=self.fs,
                signal=s,
                fc=self.fc,
                labels=["$S(f)$"],
                title="Sinal Modulado $IQ$",
                xlim=(0, 8),
                colors="darkred",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()
        
            fig_freq.tight_layout()
            save_figure(fig_freq, "transmitter_modulator_freq.pdf")

            fig_const, grid = create_figure(1, 2, figsize=(16, 8))
            PhasePlot(
                fig_const, grid, (0, 0),
                t=t,
                signals=[dI, dQ],
                labels=["Fase $I + jQ$"],
                title="Fase $I + jQ$",
                xlim=(40, 320),
                colors=["darkred"],
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            ConstellationPlot(
                fig_const, grid, (0, 1),
                dI=dI[:40000:5],
                dQ=dQ[:40000:5],
                xlim=(-1.4, 1.4),
                ylim=(-1.4, 1.4),
                rms_norm=True,
                show_ideal_points=False,
                title="Constelação $IQ$",
                colors=["darkred"],
                style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            fig_const.tight_layout()
            save_figure(fig_const, "transmitter_modulator_constellation.pdf") 

            fig_portadora, grid = create_figure(1, 2, figsize=(16, 8))
            FrequencyPlot(
                fig_portadora, grid, (0, 0),
                fs=self.fs,
                signal=s[0:(int(round(0.082 * self.fs)))],
                fc=self.fc,
                labels=["$S(f)$"],
                title="Portadora Pura - $0$ a $80$ms",
                xlim=(-10, 10),
                colors="darkred",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_portadora, grid, (0, 1),
                fs=self.fs,
                signal=s[(int(round(0.082 * self.fs))):],
                fc=self.fc,
                labels=["$S(f)$"],
                title="Sinal Modulado - $80$ms em diante",
                xlim=(-10, 10),
                colors="darkred",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
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
            t (np.ndarray): Time vector, $t$.
            s (np.ndarray): Modulated signal, $s(t)$.
        """
        ut = self.prepare_datagram(datagram)
        vt0, vt1 = self.encode_convolutional(ut)
        X, Y = self.scramble(vt0, vt1)
        sI, sQ = self.generate_preamble()
        Xn, Yn = self.multiplex(sI, sQ, X, Y)
        Xnrz, Yman = self.encode_channels(Xn, Yn)
        dI, dQ = self.format_signals(Xnrz, Yman)
        t, s = self.modulate(dI, dQ)
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

