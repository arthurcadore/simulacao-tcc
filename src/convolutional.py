"""
Codificação e decodificação convolucional segundo o padrão CCSDS 131.1-G-2, utilizado no sistema PTT-A3.

Referência:
    AS3-SP-516-274-CNES (seção 3.1.4.4)

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
import komm 
from plotter import create_figure, save_figure, BitsPlot, TrellisPlot

class EncoderConvolutional: 
    def __init__(self, G=np.array([[0b1111001, 0b1011011]])):
        r"""
        Inicializa o codificador convolucional com matriz de geradores.

        Referência:
            - AS3-SP-516-274-CNES (seção 3.1.4.4)
            - CCSDS 131.1-G-2

        Args:
            G (np.ndarray): Matriz de polinômios geradores em formato binário.
        
        Nota: 
            - O polinômio gerador $G_0$ é representado por `G[0][0]` e $G_1$ por `G[0][1]`.
            - $G_0$ é definido como $1111001_{2}$ ou $121_{10}$
            - $G_1$ é definido como $1011011_{2}$ ou $91_{10}$

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
        Calcula os índices ("taps") dos bits ativos (ou seja, bits $'1'$) no polinômio gerador.

        Args:
            poly (int): Polinômio em formato binário. 

        Returns:
            taps (int): Lista com os índices dos taps ativos.
        """
        bin_str = f"{poly:0{self.K}b}"
        taps = [i for i, b in enumerate(bin_str) if b == '1']
        return taps

    def calc_free_distance(self):
        r"""
        Calcula a distância livre do código convolucional, definida como a menor
        distância de Hamming entre quaisquer duas sequências de saída distintas.

        Returns:
            dist (int): Distância livre do código.
        """
        return self.komm.free_distance()

    def encode(self, input_bits):
        r"""
        Codifica uma sequência binária de entrada $u_t$ utilizando os registradores deslizantes e os taps.

        Args:
            input_bits (np.ndarray): Vetor de bits $u_t$ de entrada a serem codificados.

        Returns:
            tuple (np.ndarray, np.ndarray): Tupla com os dois canais de saída $v_t^{(0)}$ e $v_t^{(1)}$.
        """
        input_bits = np.array(input_bits, dtype=int)
        vt0 = []
        vt1 = []

        for bit in input_bits:
            self.shift_register = np.insert(self.shift_register, 0, bit)[:self.K]
            out0 = np.sum(self.shift_register[self.g0_taps]) % 2
            out1 = np.sum(self.shift_register[self.g1_taps]) % 2
            vt0.append(out0)
            vt1.append(out1)

        return np.array(vt0, dtype=int), np.array(vt1, dtype=int)


class DecoderViterbi:
    r"""
    Implementa o decodificador Viterbi, no padrão CCSDS 131.1-G-2, utilizado no PTT-A3.

    Referência:
        AS3-SP-516-274-CNES (3.1.4.4)
    """
    def __init__(self, G=np.array([[0b1111001, 0b1011011]])):
        r"""
        Inicializa o decodificador Convolucional.

        Args:
            G (np.ndarray): Matriz de polinômios geradores.
        """
        self.G = G
        self.G0 = int(G[0][0])
        self.G1 = int(G[0][1])
        self.K = max(self.G0.bit_length(), self.G1.bit_length())
        self.num_states = 2**(self.K - 1)
        self.trellis = self.build_trellis()

    def build_trellis(self):
        r"""
        Constroi a trelica do decodificador Viterbi.

        Returns:
            dict: Trelica do decodificador Viterbi.
        """
        trellis = {}
        for state in range(self.num_states):
            trellis[state] = {}
            for bit in [0, 1]:
                # reconstruir o shift register (bit atual + estado anterior)
                sr = [bit] + [int(b) for b in format(state, f'0{self.K - 1}b')]
                out0 = sum([sr[i] for i in range(self.K) if (self.G0 >> (self.K - 1 - i)) & 1]) % 2
                out1 = sum([sr[i] for i in range(self.K) if (self.G1 >> (self.K - 1 - i)) & 1]) % 2
                out = [out0, out1]

                # remove último bit para próximo estado
                next_state = int(''.join(str(b) for b in sr[:-1]), 2)  
                trellis[state][bit] = (next_state, out)
        return trellis

    def decode(self, vt0, vt1):
        r"""
        Decodifica os bits de entrada.

        Args:
            vt0 (np.ndarray): Bits de entrada do canal I.
            vt1 (np.ndarray): Bits de entrada do canal Q.

        Returns:
            np.ndarray: Bits decodificados.
        """
        vt0 = np.array(vt0, dtype=int)
        vt1 = np.array(vt1, dtype=int)
        T = len(vt0)

        # Inicializar métricas
        path_metrics = np.full((T + 1, self.num_states), np.inf)
        path_metrics[0][0] = 0
        prev_state = np.full((T + 1, self.num_states), -1, dtype=int)
        prev_input = np.full((T + 1, self.num_states), -1, dtype=int)

        # Viterbi
        for t in range(T):
            for state in range(self.num_states):
                if path_metrics[t, state] < np.inf:
                    for bit in [0, 1]:
                        next_state, expected_out = self.trellis[state][bit]
                        dist = (expected_out[0] != vt0[t]) + (expected_out[1] != vt1[t])
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
        colors=["darkred"]
    ).plot(ylabel="$u_t$")

    BitsPlot(
        fig_conv, grid_conv, (1, 0),
        bits_list=[vt0],
        sections=[("$v_t^{(0)}$", len(vt0))],
        colors=["darkgreen"]
    ).plot(ylabel="$v_t^{(0)}$")

    BitsPlot(
        fig_conv, grid_conv, (2, 0),
        bits_list=[vt1],
        sections=[("$v_t^{(1)}$", len(vt1))],
        colors=["navy"]
    ).plot(ylabel="$v_t^{(1)}$")

    fig_conv.tight_layout()
    save_figure(fig_conv, "example_conv_time.pdf")

    decoder = DecoderViterbi()
    ut_prime = decoder.decode(vt0, vt1)

    fig_trellis, grid_trellis = create_figure(1, 1, figsize=(16, 9))

    TrellisPlot(
        fig_trellis, grid_trellis, (0, 0),
        decoder.trellis,
        num_steps=10,
        initial_state=0,
        colors=["darkred", "darkgreen", "navy"]
    ).plot()

    fig_trellis.tight_layout()
    save_figure(fig_trellis, "example_conv_trellis.pdf")

    print("ut': ", ''.join(str(b) for b in ut_prime))
    print("ut = ut': ", np.array_equal(ut, ut_prime))
    
