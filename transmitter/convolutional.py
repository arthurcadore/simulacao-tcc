import numpy as np
import matplotlib.pyplot as plt
import json
import os
import scienceplots
import komm 
from collections import defaultdict

# Estilo science
plt.style.use('science')
plt.rcParams["figure.figsize"] = (16, 9)
plt.rc('font', size=16)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=22)
plt.rc('legend', frameon=True, edgecolor='black', facecolor='white', fancybox=True, fontsize=12)


class ConvolutionalEncoder: 
    def __init__(self, G):
        self.G = G
        self.G0 = int(G[0][0])
        self.G1 = int(G[0][1])
        self.K = max(self.G0.bit_length(), self.G1.bit_length())
        self.g0_taps = self._get_taps(self.G0)
        self.g1_taps = self._get_taps(self.G1)
        self.shift_register = np.zeros(self.K, dtype=int)
        self.komm = komm.ConvolutionalCode(G)


    def _get_taps(self, poly):
        bin_str = f"{poly:0{self.K}b}"
        taps = [i for i, b in enumerate(bin_str) if b == '1']
        return taps

    def encode(self, input_bits):
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

    def calc_free_distance(self):
        return self.komm.free_distance()

    def plot_encode(U, V0, V1, save_path=None):

        # Superamostragem do sinal para o plot
        U_up = np.repeat(U, 2)
        V0_up = np.repeat(V0, 2)
        V1_up = np.repeat(V1, 2)
        x = np.arange(len(V0_up))
        bit_edges = np.arange(0, len(V0_up) + 1, 2)

        # Configuração do gráfico
        fig, axs = plt.subplots(3, 1, sharex=True)
        def setup_grid(ax):
            ax.set_xlim(0, len(V0_up))
            ax.set_ylim(-0.2, 1.4)
            ax.grid(False)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
        
        # Canal I
        axs[0].step(x, U_up, where='post', label=r"Entrada $(u_t)$", color='darkgreen', linewidth=3)
        for i, bit in enumerate(U):
            axs[0].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[0].set_ylabel(r"$Entrada$")
        leg0 = axs[0].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        axs[0].set_yticks([0, 1])
        setup_grid(axs[0])

        # Canal I
        axs[1].step(x, V0_up, where='post', label=r"Canal I $(v_t^{(0)})$", color='navy', linewidth=3)
        for i, bit in enumerate(V0):
            axs[1].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[1].set_ylabel(r"Canal I $v_t^{(0)}$")
        leg0 = axs[1].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        axs[1].set_yticks([0, 1])
        setup_grid(axs[1])

        # Canal Q
        axs[2].step(x, V1_up, where='post', label=r"Canal Q $(v_t^{(1)})$", color='darkred', linewidth=3)
        for i, bit in enumerate(V1):
            axs[2].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[2].set_ylabel(r"Canal Q $v_t^{(1)}$")
        leg1 = axs[2].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)
        axs[2].set_yticks([0, 1])
        setup_grid(axs[2])

        # Configuração do layout
        plt.xlabel('Bits')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Salvar ou mostrar o gráfico
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

class ViterbiDecoder:
    def __init__(self, G):
        self.G0 = int(G[0][0])
        self.G1 = int(G[0][1])
        self.K = max(self.G0.bit_length(), self.G1.bit_length())
        self.num_states = 2**(self.K - 1)
        self.trellis = self._build_trellis()

    def _int_to_bits(self, x, width):
        return [int(b) for b in format(x, f'0{width}b')]

    def _build_trellis(self):
        trellis = {}
        for state in range(self.num_states):
            trellis[state] = {}
            for bit in [0, 1]:
                # reconstruir o shift register (bit atual + estado anterior)
                sr = [bit] + self._int_to_bits(state, self.K - 1)
                out0 = sum([sr[i] for i in range(self.K) if (self.G0 >> (self.K - 1 - i)) & 1]) % 2
                out1 = sum([sr[i] for i in range(self.K) if (self.G1 >> (self.K - 1 - i)) & 1]) % 2
                out = [out0, out1]
                next_state = int(''.join(str(b) for b in sr[:-1]), 2)  # remove último bit para próximo estado
                trellis[state][bit] = (next_state, out)
        return trellis

    def decode(self, vt0, vt1):
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

    def plot_trellis_module(self, num_steps=5, initial_state=0, save_path=None):
        states_per_time = defaultdict(set)
        states_per_time[0].add(initial_state)
        branches = []
        for t in range(num_steps):
            for state in states_per_time[t]:
                for bit in [0, 1]:
                    next_state, out = self.trellis[state][bit]
                    module = sum(np.abs(out))
                    branches.append((t, state, bit, next_state, module, out))
                    states_per_time[t+1].add(next_state)

        all_states = sorted(set(s for states in states_per_time.values() for s in states))
        state_to_x = {s: i for i, s in enumerate(all_states)}
        num_states = len(all_states)

        plt.figure(figsize=(0.35*num_states, 1*num_steps))
        ax = plt.gca()
        ax.set_xlabel('Estado')
        ax.set_ylabel('Tempo')
        ax.set_xticks(range(num_states))
        ax.set_xticklabels([hex(s)[2:].upper() for s in all_states])
        ax.set_yticks(range(num_steps+1))
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax.grid(True, axis='y', linestyle=':', alpha=0.2)
        ax.invert_yaxis()

        # Plotar os ramos
        for t, state, bit, next_state, module, out in branches:
            x = [state_to_x[state], state_to_x[next_state]]
            y = [t, t+1]
            color = 'C0' if bit == 0 else 'C1'
            ax.plot(x, y, color=color, lw=2, alpha=0.8)

        # Legenda das cores dos ramos
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='C0', lw=2, label='Bit de entrada 0'),
            Line2D([0], [0], color='C1', lw=2, label='Bit de entrada 1')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True)

        # Plotar os nós (estados em cada tempo)
        for t in range(num_steps+1):
            for state in states_per_time[t]:
                ax.plot(state_to_x[state], t, 'o', color='k', markersize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Treliça salva em {save_path}")
        else:
            plt.show()




if __name__ == "__main__":

    G = np.array([[0b1111001, 0b1011011]])
    encoder = ConvolutionalEncoder(G)
    distance = encoder.calc_free_distance()

    print("Polinômios geradores:")
    print("G0:", ''.join(str(b) for b in bin(encoder.G0)))
    print("G1:", ''.join(str(b) for b in bin(encoder.G1)))
    print("Taps G0:", ''.join(str(b) for b in encoder.g0_taps))
    print("Taps G1:", ''.join(str(b) for b in encoder.g1_taps))
    print("Distância livre:", distance)
    print("Tamanho do registro de deslocamento:", encoder.K)

    ut = np.random.randint(0, 2, 80)
    vt0, vt1 = encoder.encode(ut)
    print("Entrada ut:", ''.join(str(b) for b in ut))
    print("Saída vt0: ", ''.join(str(b) for b in vt0))
    print("Saída vt1: ", ''.join(str(b) for b in vt1))

    output_path = os.path.join("out", "example_convolutional.pdf")
    ConvolutionalEncoder.plot_encode(ut, vt0, vt1, save_path=output_path)
    
    decoder = ViterbiDecoder(G)
    ut_hat = decoder.decode(vt0, vt1)
    output_path = os.path.join("out", "example_trelica.pdf")
    decoder.plot_trellis_module(num_steps=10, initial_state=0, save_path=output_path)

    print("Entrada original ut: ", ''.join(str(b) for b in ut))
    print("Decodificada ut_hat: ", ''.join(str(b) for b in ut_hat))
