import numpy as np
from plots import Plotter

class Preamble:
    def __init__(self, preamble_hex="2BEEEEBF"):
        self.preamble_hex = preamble_hex
        self.preamble_bits = self.hex_to_bits(self.preamble_hex)

    def hex_to_bits(self, hex_string):
        return format(int(hex_string, 16), '032b')[2:] 
    
    def generate_preamble(self):
        Si = np.array([int(bit) for bit in self.preamble_bits[::2]])
        Sq = np.array([int(bit) for bit in self.preamble_bits[1::2]])
        return Si, Sq

if __name__ == "__main__":

    preamble = Preamble(preamble_hex="2BEEEEBF")
    S_i, S_q = preamble.generate_preamble()

    print("S_i: ", ''.join(str(int(b)) for b in S_i))
    print("S_q: ", ''.join(str(int(b)) for b in S_q))

    plot = Plotter()
    plot.plot_preamble(S_i, 
                       S_q, 
                       r"$S_i$", 
                       r"$S_q$", 
                       r"Canal $I$", 
                       r"Canal $Q$", 
                       save_path="../out/example_preamble.pdf"
    )