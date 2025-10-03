# """
# Encoder e Decoder using CCSDS 131.1-G-2 standard polynomials, used in ARGOS-3 system standard.
# 
# Autor: Arthur Cadore
# Data: 28-07-2025
# """

import numpy as np
import komm 
from .plotter import create_figure, save_figure, BitsPlot, TimePlot, SampledSignalPlot, SymbolsPlot
from .formatter import Formatter
from .matchedfilter import MatchedFilter
from .encoder import Encoder
from .sampler import Sampler
from .encoder import Encoder
from .env_vars import *

class EncoderConvolutional: 
    def __init__(self, G=np.array([[0b1111001, 0b1011011]])):
        r"""
        Inicialize the Convolutional Encoder with the given polynomials $G$. The default polynomials are $G_0 = 121_{10}$ and $G_1 = 91_{10}$. 

        $$
        \begin{equation}
            \begin{split}
                G_0 &= 121_{10} \quad \mapsto \quad G_0 = [1, 1, 1, 1, 0, 0, 1] \\
                G_1 &= 91_{10} \quad \mapsto \quad G_1 = [1, 0, 1, 1, 0, 1, 1]
            \end{split}
        \end{equation}
        $$

        The Convolutional Encoder can be represented by the block diagram below that corresponds to the shift register organization.

        ![pageplot](../assets/cod_convolucional.svg)
        
        Args:
            G (np.ndarray): Tuple of generator polynomials $G$.

        Examples: 
            >>> import argos3
            >>> import numpy as np
            >>>
            >>> ut = np.random.randint(0, 2, 30)
            >>> encoder = argos3.EncoderConvolutional(G=np.array([[121, 91]]))
            >>> vt0, vt1 = encoder.encode(ut)
            >>>
            >>> print(ut)
            [1 0 0 0 0 1 0 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 0 0 1 1]
            >>>
            >>> decoder = argos3.DecoderViterbi(G=np.array([[121, 91]]))
            >>> ut_prime = decoder.decode(vt0, vt1)
            >>>
            >>> print(ut_prime)
            >>>
            [1 0 0 0 0 1 0 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 0 0 1 1]
            >>> print("ut = ut': ", np.array_equal(ut, ut_prime))
            ut = ut':  True

            - Bitstream Plot Example: ![pageplot](assets/example_conv_time.svg)

        <div class="referencia">
          <b>Reference:</b>
          <p>AS3-SP-516-274-CNES (seção 3.1.4.4)</p>
          <p>CCSDS 131.1-G-2</p>
        </div>
        """

        # Attributes
        self.G = G
        self.G0 = int(G[0][0])
        self.G1 = int(G[0][1])

        # Calculate the number of bits in the shift register
        self.K = max(self.G0.bit_length(), self.G1.bit_length())

        # Calculate the taps for each generator polynomial
        self.g0_taps = self.calc_taps(self.G0)
        self.g1_taps = self.calc_taps(self.G1)

        # Initialize the shift register
        self.shift_register = np.zeros(self.K, dtype=int)

        # Initialize komm library (for free distance calculation)
        self.komm = komm.ConvolutionalCode(G)

    def calc_taps(self, poly):
        r"""
        Generate the taps for each generator polynomial.

        Args:
            poly (int): Generator polynomial $G_n$ in binary format.

        Returns:
            taps (int): List of active tap indices.
        """

        # Convert the generator polynomial to a binary string
        poly_bin = f"{poly:0{self.K}b}"

        # Generate the taps for each generator polynomial
        taps = [i for i, b in enumerate(poly_bin) if b == '1']
        return taps

    def calc_free_distance(self):
        r"""
        Calculate the free distance $d_{free}$ of the convolutional code, defined as the minimum Hamming distance between any two distinct output sequences.

        Returns:
            dist (int): Free distance $d_{free}$ of the convolutional encoder organized with $G$.
        """

        # Calculate the free distance using the komm library
        return self.komm.free_distance()

    def encode(self, ut):
        r"""
        Encode a binary input sequence $u_t$, returning the output sequences $v_t^{(0)}$ and $v_t^{(1)}$. The encoding process can be represented by the expression below.

        $$
        \begin{equation}
        \begin{bmatrix} v_t^{(0)} & v_t^{(1)} \end{bmatrix}
        =
        \begin{bmatrix}
        u_{(t)} & u_{(t-1)} & u_{(t-2)} & u_{(t-3)} & u_{(t-4)} & u_{(t-5)} & u_{(t-6)}
        \end{bmatrix}
        \cdot
        \begin{bmatrix} G_{0} & G_{1} \end{bmatrix}^{T}
        \end{equation}
        $$

        Where: 
            - $v_t^{(0)}$ and $v_t^{(1)}$: Output channels of the encoder.
            - $u_t$: Input bit vector.
            - $G_{0}$ and $G_{1}$: Generator polynomials of the encoder.

        Args:
            ut (np.ndarray): Input bit vector $u_t$ to be encoded.

        Returns:
                tuple (np.ndarray, np.ndarray): Tuple with the two output channels $v_t^{(0)}$ and $v_t^{(1)}$.
        """
        ut = np.array(ut, dtype=int)
        vt0 = []
        vt1 = []

        # Encode the input sequence
        for bit in ut:

            # Shift the shift register
            self.shift_register = np.insert(self.shift_register, 0, bit)[:self.K]
            
            # Calculate the output bits
            out0 = np.sum(self.shift_register[self.g0_taps]) % 2
            out1 = np.sum(self.shift_register[self.g1_taps]) % 2
            
            # Append the output bits to the output sequences
            vt0.append(out0)
            vt1.append(out1)

        return np.array(vt0, dtype=int), np.array(vt1, dtype=int)

class DecoderViterbi:
    def __init__(self, G=np.array([[0b1111001, 0b1011011]]), decision="hard"):
        r"""
        Initialize the Viterbi decoder, based on a tuple of generator polynomials $G$ that determine the structure of the decoder.

        $$
        \begin{equation}
            \begin{split}
                G_0 &= 121_{10} \quad \mapsto \quad G_0 = [1, 1, 1, 1, 0, 0, 1] \\
                G_1 &= 91_{10} \quad \mapsto \quad G_1 = [1, 0, 1, 1, 0, 1, 1]
            \end{split}
        \end{equation}
        $$

        Args:
            G (np.ndarray): Tuple of generator polynomials $G$.
            decision (str): Decision type, either "hard" or "soft".

        Raises:
            ValueError: If decision is not "hard" or "soft".

        Examples: 
            >>> import argos3
            >>> import numpy as np
            >>>
            >>> ut = np.random.randint(0, 2, 30)
            >>> encoder = argos3.EncoderConvolutional(G=np.array([[121, 91]]))
            >>> vt0, vt1 = encoder.encode(ut)
            >>>
            >>> print(ut)
            [1 0 0 0 0 1 0 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 0 0 1 1]
            >>>
            >>> decoder = argos3.DecoderViterbi(G=np.array([[121, 91]]))
            >>> ut_prime = decoder.decode(vt0, vt1)
            >>>
            >>> print(ut_prime)
            >>>
            [1 0 0 0 0 1 0 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 0 0 1 1]
            >>> print("ut = ut': ", np.array_equal(ut, ut_prime))
            ut = ut':  True

            - Bitstream Plot Example: ![pageplot](assets/example_conv_time.svg)

        <div class="referencia">
          <b>References:</b>
          <p>https://rwnobrega.page/apontamentos/codigos-convolucionais/</p>
          <p>AS3-SP-516-274-CNES (seção 3.1.4.4)</p>
          <p>https://dsplog.com/2009/01/14/soft-viterbi/</p>
        </div>
        """
        
        # Attributes
        self.G = G
        self.G0 = int(G[0][0])
        self.G1 = int(G[0][1])

        # Calculate the number of bits in the shift register
        self.K = max(self.G0.bit_length(), self.G1.bit_length())
        self.num_states = 2**(self.K - 1)

        # Build the trellis
        self.trellis = self.build_trellis()

        # Set the decision type
        self.decision_type = decision.lower()

    def build_trellis(self):
        r"""
        Build the trellis for the Viterbi decoder based on the generator polynomials $G$.

        Returns:
            trellis (dict): Trellis for the Viterbi decoder.
        """
        trellis = {}

        # Build the trellis
        for state in range(self.num_states):
            trellis[state] = {}
            # for each bit, calculate the output bits and the next state
            for bit in [0, 1]:

                # Calculate the shift register
                sr = [bit] + [int(b) for b in format(state, f'0{self.K - 1}b')]
                
                # Calculate the output bits
                out0 = sum([sr[i] for i in range(self.K) if (self.G0 >> (self.K - 1 - i)) & 1]) % 2
                out1 = sum([sr[i] for i in range(self.K) if (self.G1 >> (self.K - 1 - i)) & 1]) % 2

                # Calculate the next state
                next_state = int(''.join(str(b) for b in sr[:-1]), 2)
                
                # Add the transition to the trellis
                trellis[state][bit] = (next_state, [out0, out1])
        return trellis

    def branch_metric(self, trellis_symbol, received_symbol):
        r"""
        Calculate the branch metric depending on the decision type for the Viterbi decoder.

        For Hard Decision uses Hamming distance, given by the expression below.

        $$
        \begin{equation}
            \begin{split}
                d_{hamming} &= \sum_{i=0}^{N-1} |r_i - s_i|
            \end{split}
        \end{equation}
        $$
        
        Where: 
            - $d_{hamming}$: Hamming distance.
            - $r_i$: Received bit.
            - $s_i$: Expected trellis output bit.

        Examples:
            - Bitstream Plot Example: ![pageplot](assets/example_conv_time.svg)

        For Soft Decision uses Euclidean distance, given by the expression below.

        $${
        \begin{equation}
            \begin{split}
                d_{euclidean} &= \sum_{i=0}^{N-1} (r_i - s_i)^2 
            \end{split}
        \end{equation}
        }$$

        Where: 
            - $d_{euclidean}$: Euclidean distance.
            - $r_i$: Received symbol.
            - $s_i$: Expected trellis output symbol.

        Examples:
            Soft Decision Example: 

            - Stream: ![pageplot](assets/example_conv_time_soft.svg)
            - Sampling: ![pageplot](assets/example_conv_time_soft_sampled.svg)
            - Decoding: ![pageplot](assets/example_conv_time_soft_quantized.svg)
            
    
        Args:
            trellis_symbol (np.ndarray): Expected trellis output symbol, based on the trellis state. 
            received_symbol (np.ndarray): Received symbol.

        Returns:
            metric (float): Branch metric.
        """
        if self.decision_type == "hard":
            # Hamming between expected and rounded received
            return (trellis_symbol[0] != int(round(received_symbol[0]))) + \
                   (trellis_symbol[1] != int(round(received_symbol[1])))
        elif self.decision_type == "soft":

            # Convert expected trellis output to NRZ
            nrz_symbol = 2*np.array(trellis_symbol, dtype=float) - 1.0
            
            # squared euclidean distance (cost): (r - s)^2 summed
            return np.sum((received_symbol - nrz_symbol)**2)
        else:
            raise ValueError("decision_type deve ser 'hard' ou 'soft'.")

    def decode(self, vt0, vt1):
        r"""
        Decode the bits of the input $v_t^{(0)}$ and $v_t^{(1)}$, returning the decoded bits $u_t$.

        Args:
            vt0 (np.ndarray): Bits of the input channel I.
            vt1 (np.ndarray): Bits of the input channel Q.

        Returns:
            ut_hat (np.ndarray): Decoded bits.
        """

        # Convert the input to numpy arrays
        vt0 = np.array(vt0, dtype=float)
        vt1 = np.array(vt1, dtype=float)
        T = len(vt0)

        # Initialize the path metrics
        path_metrics = np.full((T + 1, self.num_states), np.inf, dtype=float)
        path_metrics[0, 0] = 0.0

        # Initialize the previous state and input
        prev_state = np.full((T + 1, self.num_states), -1, dtype=int)
        prev_input = np.full((T + 1, self.num_states), -1, dtype=int)

        # Initialize the minimum metric per bit
        bit_metric = np.full((T, 2), np.inf, dtype=float)

        # for each time step
        for t in range(T):
            # for each state
            for state in range(self.num_states):
                # if the path metric is not infinite
                if path_metrics[t, state] < np.inf:
                    # for each bit
                    for bit in [0, 1]:

                        # get the next state and expected output
                        next_state, expected_out = self.trellis[state][bit]
                        
                        # get the received symbol
                        received = np.array([vt0[t], vt1[t]])
                        
                        # calculate the branch metric
                        bm = self.branch_metric(expected_out, received)
                        
                        # calculate the metric
                        metric = path_metrics[t, state] + bm

                        # update the minimum metric per bit
                        if metric < bit_metric[t, bit]:
                            bit_metric[t, bit] = metric

                        # update the path metrics
                        if metric < path_metrics[t + 1, next_state]:
                            path_metrics[t + 1, next_state] = metric
                            prev_state[t + 1, next_state] = state
                            prev_input[t + 1, next_state] = bit

        # get the final state
        state = np.argmin(path_metrics[T])
        
        # backtracking
        ut_hat_rev = []
        for t in range(T, 0, -1):
            bit = prev_input[t, state]
            if bit == -1:
                bit = 0
            ut_hat_rev.append(bit)
            state = prev_state[t, state]

        # reverse the decoded bits
        ut_hat = np.array(ut_hat_rev[::-1], dtype=int)

        # hard decision
        if self.decision_type == "hard":
            return ut_hat
        
        # calculate the log-likelihood ratios
        llrs = bit_metric[:, 0] - bit_metric[:, 1]
        return llrs


if __name__ == "__main__":

    print("\n\n ==== HARD DECISION ==== \n\n")

    encoder = EncoderConvolutional()
    print("Free Distance:", encoder.calc_free_distance())
    print("G0:  ", format(encoder.G0, 'b'), " |  Taps: ", ''.join(str(b) for b in encoder.g0_taps))
    print("G1:  ", format(encoder.G1, 'b'), " |  Taps: ", ''.join(str(b) for b in encoder.g1_taps))

    ut = np.random.randint(0, 2, 40)
    vt0, vt1 = encoder.encode(ut)
    print("ut:  ", ''.join(str(b) for b in ut))
    print("vt0: ", ''.join(str(b) for b in vt0))
    print("vt1: ", ''.join(str(b) for b in vt1))
    
    fig_conv, grid_conv = create_figure(3, 1, figsize=(16, 9))
    
    BitsPlot(
        fig_conv, grid_conv, (0, 0),
        bits_list=[ut],
        sections=[("$u_t$", len(ut))],
        colors=[COLOR_COMBINED],
        title=INPUT_STREAM_TITLE,
        ylabel=BITSTREAM_Y
    ).plot()

    BitsPlot(
        fig_conv, grid_conv, (1, 0),
        bits_list=[vt0],
        sections=[("$v_t^{(0)}$", len(vt0))],
        colors=[COLOR_I],
        title=I_CHANNEL_TITLE,
        ylabel=BITSTREAM_Y
    ).plot()

    BitsPlot(
        fig_conv, grid_conv, (2, 0),
        bits_list=[vt1],
        sections=[("$v_t^{(1)}$", len(vt1))],
        colors=[COLOR_Q],
        title=Q_CHANNEL_TITLE,
        xlabel=BITSTREAM_X,
        ylabel=BITSTREAM_Y 
    ).plot()

    fig_conv.tight_layout()
    save_figure(fig_conv, "example_conv_time.pdf")

    decoder = DecoderViterbi()
    ut_prime = decoder.decode(vt0, vt1)

    print("ut': ", ''.join(str(b) for b in ut_prime))
    print("ut = ut': ", np.array_equal(ut, ut_prime))

    print("\n\n ==== SOFT DECISION ==== \n\n")

    encoder = EncoderConvolutional()
    print("Free Distance:", encoder.calc_free_distance())
    print("G0:  ", format(encoder.G0, 'b'), " |  Taps: ", ''.join(str(b) for b in encoder.g0_taps))
    print("G1:  ", format(encoder.G1, 'b'), " |  Taps: ", ''.join(str(b) for b in encoder.g1_taps))

    ut = np.random.randint(0, 2, 40)
    vt0, vt1 = encoder.encode(ut)
    print("ut:  ", ''.join(str(b) for b in ut))
    print("vt0: ", ''.join(str(b) for b in vt0))
    print("vt1: ", ''.join(str(b) for b in vt1))

    encoder_NRZ = Encoder()

    X = encoder_NRZ.encode(vt0)
    Y = encoder_NRZ.encode(vt1)

    formatterI = Formatter(type="RRC", channel="I", bits_per_symbol=1, Rb=1000, fs=128000, alpha=0.8, span=12, prefix_duration=0.01)
    formatterQ = Formatter(type="Manchester", channel="Q", bits_per_symbol=2, Rb=1000, fs=128000, alpha=0.8, span=12, prefix_duration=0.01)

    dX = formatterI.apply_format(X, add_prefix=True)
    dY = formatterQ.apply_format(Y, add_prefix=True)

    mfI = MatchedFilter(alpha=0.8, fs=128000, Rb=1000, span=12, type="RRC-Inverted", channel="I", bits_per_symbol=1)
    mfQ = MatchedFilter(alpha=0.8, fs=128000, Rb=1000, span=12, type="Manchester-Inverted", channel="Q", bits_per_symbol=2)

    noise = np.random.normal(0, 1, len(dX)) * 0.5

    dX_prime = mfI.apply_filter(dX + noise)
    dY_prime = mfQ.apply_filter(dY + noise)
    
    fig_time, grid_time = create_figure(3, 2, figsize=(16, 9))

    BitsPlot(
        fig_time, grid_time, (0, 0),
        bits_list=[vt0],
        sections=[("$v_t^{(0)}$", len(vt0))],
        colors=[COLOR_I],
        xlabel=BITSTREAM_X,
        ylabel=BITSTREAM_Y,
        title=I_CHANNEL_TITLE,
    ).plot()

    BitsPlot(
        fig_time, grid_time, (0, 1),
        bits_list=[vt1],
        sections=[("$v_t^{(1)}$", len(vt1))],
        colors=[COLOR_Q],
        xlabel=BITSTREAM_X,
        title=Q_CHANNEL_TITLE, 
    ).plot()

    TimePlot(
        fig_time, grid_time, (1, 0),
        t = np.arange(len(dX)) / formatterI.fs,
        signals=[dX],
        labels=[r"$d_I(t)$"],
        colors=[COLOR_I],
    ).plot()

    TimePlot(
        fig_time, grid_time, (1, 1),
        t = np.arange(len(dY)) / formatterQ.fs,
        signals=[dY],
        labels=[r"$d_Q(t)$"],
        colors=[COLOR_Q],
    ).plot()

    TimePlot(
        fig_time, grid_time, (2, 0),
        t = np.arange(len(dX_prime)) / formatterI.fs,
        signals=[dX_prime],
        labels=[r"$d_I'(t)$"],
        colors=[COLOR_I],
    ).plot()

    TimePlot(
        fig_time, grid_time, (2, 1),
        t = np.arange(len(dY_prime)) / formatterQ.fs,
        signals=[dY_prime],
        labels=[r"$d_Q'(t)$"],
        colors=[COLOR_Q],
    ).plot()

    fig_time.tight_layout()
    save_figure(fig_time, "example_conv_time_soft.pdf")
    
    print("dX len: ", len(dX_prime))
    print("dY len: ", len(dY_prime))

    tI = np.arange(len(dX_prime)) / formatterI.fs
    tQ = np.arange(len(dY_prime)) / formatterQ.fs

    print("tI len: ", len(tI))
    print("tQ len: ", len(tQ))
    
    samplerI = Sampler(fs=128000, Rb=1000, t=tI, delay=0.01)
    samplerQ = Sampler(fs=128000, Rb=1000, t=tQ, delay=0.01)

    X_prime = samplerI.sample(dX_prime)
    Y_prime = samplerQ.sample(dY_prime)
    tX = samplerI.sample(tI)
    tY = samplerQ.sample(tQ)

    print("X len: ", len(X_prime))
    print("Y len: ", len(Y_prime))
    print("tx len: ", len(tX))
    print("ty len: ", len(tY))

    decoder = DecoderViterbi(decision="soft")
    ut_prime = decoder.decode(X_prime, Y_prime)

    fig_time, grid_time = create_figure(2, 2, figsize=(16, 9))

    SampledSignalPlot(
        fig_time, grid_time, (0, 0),
        tI,
        dX_prime, 
        tX,
        X_prime,
        colors=COLOR_I,
        label_signal=r"$d_I'(t)$", 
        label_samples=r"Samples $I[n]'$",
        title=I_CHANNEL_TITLE
    ).plot()
        
    SampledSignalPlot(
        fig_time, grid_time, (0, 1),
        tQ,
        dY_prime,
        tY,
        Y_prime,
        colors=COLOR_Q,
        label_signal=r"$d_Q'(t)$", 
        label_samples=r"Samples $Q[n]'$",
        title=Q_CHANNEL_TITLE
    ).plot()

    SymbolsPlot(
        fig_time, grid_time, (1, 0),
        symbols_list=[X_prime],
        samples_per_symbol=1,
        colors=[COLOR_I],
        xlabel=SYMBOLS_X,
        ylabel=SYMBOLS_Y,
        label=r"$I'[n]$",
        ylim=[min(X_prime)*1.1, max(X_prime)*1.1],
        x_axis_label=(min(X_prime), max(X_prime)),
        show_symbol_values=False
    ).plot()

    SymbolsPlot(
        fig_time, grid_time, (1, 1),
        symbols_list=[Y_prime],
        samples_per_symbol=1,
        colors=[COLOR_Q],
        xlabel=SYMBOLS_X,
        ylabel=SYMBOLS_Y,
        label=r"$Q'[n]$",
        ylim=[min(Y_prime)*1.1, max(Y_prime)*1.1],
        x_axis_label=(min(Y_prime), max(Y_prime)),
        show_symbol_values=False
    ).plot()

    fig_time.tight_layout()
    save_figure(fig_time, "example_conv_time_soft_sampled.pdf")

    print("ut': ", ''.join(str(b) for b in ut_prime))

    encoder_NRZ = Encoder(method="NRZ")
    ut_nrz = encoder_NRZ.decode(ut_prime)


    fig_time, grid_time = create_figure(3, 2, figsize=(16, 9))

    SymbolsPlot(
        fig_time, grid_time, (0, 0),
        symbols_list=[X_prime],
        samples_per_symbol=1,
        colors=[COLOR_I],
        xlabel=SYMBOLS_X,
        ylabel=SYMBOLS_Y,
        label=r"$I'[n]$",
        show_symbol_values=False,
        ylim=[min(X_prime)*1.1, max(X_prime)*1.1],
        x_axis_label=(min(X_prime), max(X_prime)),
        title=I_CHANNEL_TITLE
    ).plot()

    SymbolsPlot(
        fig_time, grid_time, (0, 1),
        symbols_list=[Y_prime],
        samples_per_symbol=1,
        colors=[COLOR_Q],
        xlabel=SYMBOLS_X,
        ylabel=SYMBOLS_Y,
        label=r"$Q'[n]$",
        show_symbol_values=False,
        ylim=[min(Y_prime)*1.1, max(Y_prime)*1.1],
        x_axis_label=(min(Y_prime), max(Y_prime)),
        title=Q_CHANNEL_TITLE
    ).plot()

    SymbolsPlot(
        fig_time, grid_time, (1, slice(0, 2)),
        symbols_list=[ut_prime],
        samples_per_symbol=1,
        colors=[COLOR_COMBINED],
        xlabel=SYMBOLS_X,
        ylabel=SYMBOLS_Y,
        label=r"$U'[n]$",
        show_symbol_values=False,
        ylim=[min(ut_prime)*1.1, max(ut_prime)*1.1],
        x_axis_label=(min(ut_prime), max(ut_prime)),
        title=OUTPUT_STREAM_TITLE
    ).plot()

    BitsPlot(
        fig_time, grid_time, (2, slice(0, 2)),
        bits_list=[ut_nrz],
        sections=[(r"$u_t^{(0)}$", len(ut_nrz))],
        colors=[COLOR_COMBINED],
        xlabel=BITSTREAM_X,
        ylabel=BITSTREAM_Y,
    ).plot()

    fig_time.tight_layout()
    save_figure(fig_time, "example_conv_time_soft_quantized.pdf")

    print("ut:  ", ''.join(str(b) for b in ut))
    print("ut': ", ''.join(str(b) for b in ut_nrz))
    print("ut = ut': ", np.array_equal(ut, ut_nrz))