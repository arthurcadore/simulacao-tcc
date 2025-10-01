# """
# Codificação e decodificação convolucional segundo o padrão CCSDS 131.1-G-2, utilizado no sistema PTT-A3.

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

class EncoderConvolutional: 
    def __init__(self, G=np.array([[0b1111001, 0b1011011]])):
        r"""
        Inicializa o codificador convolucional, com base em uma tupla de polinômios geradores $G$ que determinam a estrutura do codificador.

        $$
        \begin{equation}
            \begin{split}
                G_0 &= 121_{10} \quad \mapsto \quad G_0 = [1, 1, 1, 1, 0, 0, 1] \\
                G_1 &= 91_{10} \quad \mapsto \quad G_1 = [1, 0, 1, 1, 0, 1, 1]
            \end{split}
        \end{equation}
        $$

        O codificador convolucional, considerando $G_0$ e $G_1$ pode ser representado pelo diagrama de blocos abaixo.

        ![pageplot](../assets/cod_convolucional.svg)
        
        Args:
            G (np.ndarray): Tupla de polinômios geradores $G$.

        Example: 
            ![pageplot](assets/example_conv_time.svg)

        <div class="referencia">
          <b>Referência:</b>
          <p>AS3-SP-516-274-CNES (seção 3.1.4.4)</p>
          <p>CCSDS 131.1-G-2</p>
        </div>
        """
        self.G = G
        self.G0 = int(G[0][0])
        self.G1 = int(G[0][1])
        self.K = max(self.G0.bit_length(), self.G1.bit_length())
        self.g0_taps = self.calc_taps(self.G0)
        self.g1_taps = self.calc_taps(self.G1)
        self.shift_register = np.zeros(self.K, dtype=int)
        self.komm = komm.ConvolutionalCode(G)

    def calc_taps(self, poly):
        r"""
        Calcula os índices dos bits ativos ($'1'$), ou taps, do polinômio gerador $G_n$.

        Args:
            poly (int): Polinômio gerador $G_n$ em formato binário. 

        Returns:
            taps (int): Lista com os índices dos taps ativos.
        """
        bin_str = f"{poly:0{self.K}b}"
        taps = [i for i, b in enumerate(bin_str) if b == '1']
        return taps

    def calc_free_distance(self):
        r"""
        Calcula a distância livre $d_{free}$ do código convolucional, definida como a menor distância de Hamming entre quaisquer duas sequências de saída distintas.

        Returns:
            dist (int): Distância livre $d_{free}$ do codificador convolucional organizado com $G$.
        """
        return self.komm.free_distance()

    def encode(self, ut):
        r"""
        Codifica uma sequência binária de entrada $u_t$, retornando as sequências de saida $v_t^{(0)}$ e $v_t^{(1)}$. O processo de codificação pode ser representado pela expressão abaixo.

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

        Sendo: 
            - $v_t^{(0)}$ e $v_t^{(1)}$: Canais de saída do codificador.
            - $u_t$: Vetor de bits de entrada.
            - $G_{0}$ e $G_{1}$: Polinômios geradores do codificador.

        Args:
            ut (np.ndarray): Vetor de bits $u_t$ de entrada a serem codificados.

        Returns:
                tuple (np.ndarray, np.ndarray): Tupla com os dois canais de saída $v_t^{(0)}$ e $v_t^{(1)}$.
        """
        ut = np.array(ut, dtype=int)
        vt0 = []
        vt1 = []

        for bit in ut:
            self.shift_register = np.insert(self.shift_register, 0, bit)[:self.K]
            out0 = np.sum(self.shift_register[self.g0_taps]) % 2
            out1 = np.sum(self.shift_register[self.g1_taps]) % 2
            vt0.append(out0)
            vt1.append(out1)

        return np.array(vt0, dtype=int), np.array(vt1, dtype=int)


class DecoderViterbi:
    def __init__(self, G=np.array([[0b1111001, 0b1011011]]), decision="hard"):
        r"""
        Inicializa o decodificador convolucional (algoritmo Viterbi), com base em uma tupla de polinômios geradores $G$ que determinam a estrutura do decodificador.

        $$
        \begin{equation}
            \begin{split}
                G_0 &= 121_{10} \quad \mapsto \quad G_0 = [1, 1, 1, 1, 0, 0, 1] \\
                G_1 &= 91_{10} \quad \mapsto \quad G_1 = [1, 0, 1, 1, 0, 1, 1]
            \end{split}
        \end{equation}
        $$

        Args:
            G (np.ndarray): Tupla de polinômios geradores $G$.

        <div class="referencia">
          <b>Referência:</b>
          <p>https://rwnobrega.page/apontamentos/codigos-convolucionais/</p>
          <p>AS3-SP-516-274-CNES (seção 3.1.4.4)</p>
        </div>
        """
        
        self.G = G
        self.G0 = int(G[0][0])
        self.G1 = int(G[0][1])
        self.K = max(self.G0.bit_length(), self.G1.bit_length())
        self.num_states = 2**(self.K - 1)
        self.trellis = self.build_trellis()
        self.decision_type = decision.lower()

    def build_trellis(self):
        r"""
        Constroi a trelica do decodificador Viterbi.

        Returns:
            trellis (dict): Trelica do decodificador Viterbi.
        """
        trellis = {}
        for state in range(self.num_states):
            trellis[state] = {}
            for bit in [0, 1]:
                sr = [bit] + [int(b) for b in format(state, f'0{self.K - 1}b')]
                out0 = sum([sr[i] for i in range(self.K) if (self.G0 >> (self.K - 1 - i)) & 1]) % 2
                out1 = sum([sr[i] for i in range(self.K) if (self.G1 >> (self.K - 1 - i)) & 1]) % 2
                trellis[state][bit] = (int(''.join(str(b) for b in sr[:-1]), 2), [out0, out1])
        return trellis

    def branch_metric(self, expected_out, received):
        """
        Calcula a métrica de ramo dependendo do tipo de decisão.
        - Hard: distância de Hamming
        - Soft: distância euclidiana
        """
        if self.decision_type == "hard":
            return (expected_out[0] != int(round(received[0]))) + \
                   (expected_out[1] != int(round(received[1])))
        elif self.decision_type == "soft":
            return np.sum((np.array(expected_out, dtype=float) - received)**2)
        else:
            raise ValueError("decision_type deve ser 'hard' ou 'soft'.")

    def decode(self, vt0, vt1):
        r"""
        Decodifica os bits de entrada $v_t^{(0)}$ e $v_t^{(1)}$, retornando os bits decodificados $u_t$.

        Args:
            vt0 (np.ndarray): Bits de entrada do canal I.
            vt1 (np.ndarray): Bits de entrada do canal Q.

        Returns:
            ut_hat (np.ndarray): Bits decodificados.
        """
        vt0 = np.array(vt0, dtype=float)
        vt1 = np.array(vt1, dtype=float)
        T = len(vt0)

        path_metrics = np.full((T + 1, self.num_states), np.inf)
        path_metrics[0][0] = 0
        prev_state = np.full((T + 1, self.num_states), -1, dtype=int)
        prev_input = np.full((T + 1, self.num_states), -1, dtype=int)

        for t in range(T):
            for state in range(self.num_states):
                if path_metrics[t, state] < np.inf:
                    for bit in [0, 1]:
                        next_state, expected_out = self.trellis[state][bit]
                        received = np.array([vt0[t], vt1[t]])
                        dist = self.branch_metric(expected_out, received)
                        metric = path_metrics[t, state] + dist
                        if metric < path_metrics[t + 1, next_state]:
                            path_metrics[t + 1, next_state] = metric
                            prev_state[t + 1, next_state] = state
                            prev_input[t + 1, next_state] = bit

        # Traceback
        state = np.argmin(path_metrics[T])
        ut_hat = []
        for t in range(T, 0, -1):
            bit = prev_input[t, state]
            ut_hat.append(bit)
            state = prev_state[t, state]

        return np.array(ut_hat[::-1], dtype=int)


if __name__ == "__main__":

    print("\n\n ==== HARD DECISION ==== \n\n")

    encoder = EncoderConvolutional()
    print("Distância livre:", encoder.calc_free_distance())
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
        colors=["darkred"],
        ylabel="$u_t$"
    ).plot()

    BitsPlot(
        fig_conv, grid_conv, (1, 0),
        bits_list=[vt0],
        sections=[("$v_t^{(0)}$", len(vt0))],
        colors=["darkgreen"],
        ylabel="$v_t^{(0)}$"
    ).plot()

    BitsPlot(
        fig_conv, grid_conv, (2, 0),
        bits_list=[vt1],
        sections=[("$v_t^{(1)}$", len(vt1))],
        colors=["navy"],
        xlabel="Index de Bit", 
        ylabel="$v_t^{(1)}$"
    ).plot()

    fig_conv.tight_layout()
    save_figure(fig_conv, "example_conv_time.pdf")

    decoder = DecoderViterbi()
    ut_prime = decoder.decode(vt0, vt1)

    print("ut': ", ''.join(str(b) for b in ut_prime))
    print("ut = ut': ", np.array_equal(ut, ut_prime))

    print("\n\n ==== SOFT DECISION ==== \n\n")

    encoder = EncoderConvolutional()
    print("Distância livre:", encoder.calc_free_distance())
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

    noise = np.random.normal(0, 1, len(dX)) * 0.3

    dX_prime = mfI.apply_filter(dX + noise)
    dY_prime = mfQ.apply_filter(dY + noise)
    
    fig_time, grid_time = create_figure(3, 2, figsize=(16, 9))

    BitsPlot(
        fig_time, grid_time, (0, 0),
        bits_list=[vt0],
        sections=[("$v_t^{(0)}$", len(vt0))],
        colors=["darkgreen"],
        ylabel="$v_t^{(0)}$"
    ).plot()

    BitsPlot(
        fig_time, grid_time, (0, 1),
        bits_list=[vt1],
        sections=[("$v_t^{(1)}$", len(vt1))],
        colors=["navy"],
        xlabel="Index de Bit", 
        ylabel="$v_t^{(1)}$"
    ).plot()

    TimePlot(
        fig_time, grid_time, (1, 0),
        t = np.arange(len(dX)) / formatterI.fs,
        signals=[dX],
        labels=[r"$d_t^{(0)}$"],
        colors=["darkgreen"],
    ).plot()

    TimePlot(
        fig_time, grid_time, (1, 1),
        t = np.arange(len(dY)) / formatterQ.fs,
        signals=[dY],
        labels=[r"$d_t^{(1)}$"],
        colors=["navy"],
    ).plot()

    TimePlot(
        fig_time, grid_time, (2, 0),
        t = np.arange(len(dX_prime)) / formatterI.fs,
        signals=[dX_prime],
        labels=[r"$d_t^{(0)}$"],
        colors=["darkgreen"],
    ).plot()

    TimePlot(
        fig_time, grid_time, (2, 1),
        t = np.arange(len(dY_prime)) / formatterQ.fs,
        signals=[dY_prime],
        labels=[r"$d_t^{(1)}$"],
        colors=["navy"],
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
    print("ut': ", ''.join(str(b) for b in ut_prime))
    print("ut = ut': ", np.array_equal(ut, ut_prime))

    fig_time, grid_time = create_figure(3, 2, figsize=(16, 9))

    SampledSignalPlot(
        fig_time, grid_time, (0, 0),
        tI,
        dX_prime, 
        tX,
        X_prime,
        colors="darkgreen",
        label_signal="Sinal original", 
        label_samples="Amostras"    
    ).plot()
        
    SampledSignalPlot(
        fig_time, grid_time, (0, 1),
        tQ,
        dY_prime,
        tY,
        Y_prime,
        colors="navy",
        label_signal="Sinal original", 
        label_samples="Amostras"
    ).plot()

    SymbolsPlot(
        fig_time, grid_time, (1, 0),
        symbols_list=[X_prime],
        samples_per_symbol=1,
        colors=["darkgreen"],
        xlabel="Index",
        ylabel="$X_{t}^{(0)}$",
        label="$X_{t}^{(0)}$",
        show_symbol_values=False
    ).plot()

    SymbolsPlot(
        fig_time, grid_time, (1, 1),
        symbols_list=[Y_prime],
        samples_per_symbol=1,
        colors=["navy"],
        xlabel="Index",
        ylabel="$Y_{t}^{(1)}$",
        label="$Y_{t}^{(1)}$",
        show_symbol_values=False
    ).plot()

    SymbolsPlot(
        fig_time, grid_time, (2, slice(0, 2)),
        symbols_list=[ut_prime],
        samples_per_symbol=1,
        colors=["darkred"],
        xlabel="Index",
        ylabel="$U_{t}^{(0)}$",
        label="$U_{t}^{(0)}$",
        show_symbol_values=False
    ).plot()

    fig_time.tight_layout()
    save_figure(fig_time, "example_conv_time_soft_sampled.pdf")
    

    
