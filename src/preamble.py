"""
Implementa uma palavra de sincronismo compatível com o padrão PPT-A3.

Referência:
    AS3-SP-516-274-CNES (seção 3.1.4.6)

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from plots import Plotter

class Preamble:

    def __init__(self, preamble_hex="2BEEEEBF"):
        r"""
        Inicializa uma instância de palavra de sincronismo. A palavra de sincronismo $S$ é composta por 30 bits, $S = 2BEEEEBF_{16}$, conforme o padrão PPT-A3.

        Referência:
            AS3-SP-516-274-CNES (seção 3.1.4.6)

        Args:
            preamble_hex (str, opcional): Hexadecimal da palavra de sincronismo.
        
        Raises:
            ValueError: Se a palavra de sincronismo não contiver 30 bits.
            ValueError: Se o hexadecimal não for válido ou não puder ser convertido para 30 bits
        """

        if not isinstance(preamble_hex, str) or len(preamble_hex) != 8:
            raise ValueError("O hexadecimal da palavra de sincronismo deve ser uma string de 8 caracteres.")

        self.preamble_hex = preamble_hex
        self.preamble_bits = self.hex_to_bits(self.preamble_hex)

        if len(self.preamble_bits) != 30:
            raise ValueError("A palavra de sincronismo deve conter 30 bits.")

        self.preamble_sI, self.preamble_sQ = self.generate_preamble()

    def hex_to_bits(self, hex_string):
        r"""
        Converte uma string hexadecimal em uma string de bits de 30 bits.

        Args:
            hex_string (str): String hexadecimal a ser convertida.

        Returns:
            bin_str (str): String de bits de 30 bits.
        """
        return format(int(hex_string, 16), '032b')[2:] 
    
    def generate_preamble(self):
        r"""
        Gera os vetores I e Q da palavra de sincronismo, com base no vetor $S$ passado no construtor.

        Definição dos vetores:
        $$
        \begin{align}
            S_I &= [S_0, S_2, S_4, \dots, S_{28}]
        \end{align}
        $$
        $$
        \begin{align}
            S_Q &= [S_1, S_3, S_5, \dots, S_{29}]
        \end{align}
        $$

        Returns:
            tuple (np.ndarray, np.ndarray): Vetores $S_I$ e $S_Q$.
        """
        Si = np.array([int(bit) for bit in self.preamble_bits[::2]])
        Sq = np.array([int(bit) for bit in self.preamble_bits[1::2]])
        return Si, Sq

if __name__ == "__main__":

    preamble = Preamble(preamble_hex="2BEEEEBF")
    Si = preamble.preamble_sI
    Sq = preamble.preamble_sQ

    print("Si: ", ''.join(str(int(b)) for b in Si))
    print("Sq: ", ''.join(str(int(b)) for b in Sq))

    plot = Plotter()
    plot.plot_preamble(Si, 
                       Sq, 
                       r"$S_i$", 
                       r"$S_q$", 
                       r"Canal $I$", 
                       r"Canal $Q$", 
                       save_path="../out/example_preamble.pdf"
    )