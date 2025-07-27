import numpy as np
import komm 
from plots import Plotter

class EncoderConvolutional: 
    def __init__(self, G=np.array([[0b1111001, 0b1011011]])):
        self.G = G
        self.G0 = int(G[0][0])
        self.G1 = int(G[0][1])
        self.K = max(self.G0.bit_length(), self.G1.bit_length())
        self.g0_taps = self.calc_taps(self.G0)
        self.g1_taps = self.calc_taps(self.G1)
        self.shift_register = np.zeros(self.K, dtype=int)
        self.komm = komm.ConvolutionalCode(G)

    def calc_taps(self, poly):
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

class DecoderViterbi:
    def __init__(self, G=np.array([[0b1111001, 0b1011011]])):
        self.G = G
        self.G0 = int(G[0][0])
        self.G1 = int(G[0][1])
        self.K = max(self.G0.bit_length(), self.G1.bit_length())
        self.num_states = 2**(self.K - 1)
        self.trellis = self.build_trellis()

    def build_trellis(self):
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
    distance = encoder.calc_free_distance()
    print("Distância livre:", distance)
    print("G0:  ", format(encoder.G0, 'b'), " |  Taps: ", ''.join(str(b) for b in encoder.g0_taps))
    print("G1:  ", format(encoder.G1, 'b'), " |  Taps: ", ''.join(str(b) for b in encoder.g1_taps))

    ut = np.random.randint(0, 2, 40)
    vt0, vt1 = encoder.encode(ut)
    print("ut:  ", ''.join(str(b) for b in ut))
    print("vt0: ", ''.join(str(b) for b in vt0))
    print("vt1: ", ''.join(str(b) for b in vt1))

    plotter = Plotter()
    plotter.plot_conv(ut, 
                      vt0, 
                      vt1, 
                      "Entrada $u_t$", 
                      "Canal I $v_t^{(0)}$", 
                      "Canal Q $v_t^{(1)}$", 
                      "$u_t$", 
                      "$v_t^{(0)}$", 
                      "$v_t^{(1)}$", 
                      save_path="../out/example_convolutional.pdf"
    )
    
    decoder = DecoderViterbi()
    ut_prime = decoder.decode(vt0, vt1)
    plotter.plot_trellis(decoder.trellis, 
                         num_steps=10, 
                         initial_state=0, 
                         save_path="../out/example_trelica.pdf"
    )

    print("ut': ", ''.join(str(b) for b in ut_prime))
    print("ut = ut': ", np.array_equal(ut, ut_prime))
    
