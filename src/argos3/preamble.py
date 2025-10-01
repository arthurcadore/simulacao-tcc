# """
# Implements a preamble compatible with the PPT-A3 standard.
# 
# Autor: Arthur Cadore
# Data: 28-07-2025
# """

import numpy as np
from .plotter import save_figure, create_figure, BitsPlot

class Preamble:
    def __init__(self, preamble_hex="2BEEEEBF"):
        r"""
        Criate the preamble bits sequence $S$ from the hexadecimal string. On ARGOS-3 standard, the preamble is $S = 2BEEEEBF_{16}$. The preamble bits sequence $S$ is intercalated to form the vectors $S_I$ and $S_Q$ for each channel, as shown below.

        $$
        \begin{align}
        S_I &= [S_0,\, S_2,\, S_4,\, \dots,\, S_{28}] && \mapsto \quad S_I = [1111,\, 1111,\, 1111,\, 111] \\
        S_Q &= [S_1,\, S_3,\, S_5,\, \dots,\, S_{29}] && \mapsto \quad S_Q = [0011,\, 0101,\, 0100,\, 111]
        \end{align}
        $$

        Where:
            - $S$: Preamble bits sequence.
            - $S_I$ and $S_Q$: Vectors $S_I$ and $S_Q$ corresponding to the I and Q channels, respectively.

        Args:
            preamble_hex (str, opcional): Hexadecimal of the preamble.
        
        Raises:
            ValueError: If the preamble $S$ has a different length from 8 characters. 
            ValueError: If the hexadecimal is not valid or cannot be converted.

        Examples:
            >>> import argos3
            >>> 
            >>> Si, Sq = argos3.Preamble(preamble_hex="2BEEEEBF").generate_preamble()
            >>> 
            >>> print(Si)
            [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
            >>> 
            >>> print(Sq)
            [0 0 1 1 0 1 0 1 0 1 0 0 1 1 1]

            - Bitstream Plot Example: ![pageplot](assets/example_preamble.svg)

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (seção 3.1.4.6)
        </div>
        """

        # Validate the preamble
        if not isinstance(preamble_hex, str) or len(preamble_hex) != 8:
            raise ValueError("The preamble S must be a string of 8 characters.")

        # Attributes
        self.preamble_hex = preamble_hex
        self.preamble_bits = self.hex_to_bits(self.preamble_hex)

        if len(self.preamble_bits) != 30:
            raise ValueError("The preamble S must contain 30 bits.")

        # Generate the preamble
        self.preamble_sI, self.preamble_sQ = self.generate_preamble()

    def hex_to_bits(self, hex_string):
        return format(int(hex_string, 16), '032b')[2:] 
    
    def generate_preamble(self):
        r"""
        Generate the vectors $S_I$ and $S_Q$ from the preamble bits sequence $S$.

        Returns:
            tuple (np.ndarray, np.ndarray): Vectors $S_I$ and $S_Q$.
        """

        # Generate the preamble
        Si = np.array([int(bit) for bit in self.preamble_bits[::2]])
        Sq = np.array([int(bit) for bit in self.preamble_bits[1::2]])
        return Si, Sq

if __name__ == "__main__":

    preamble = Preamble(preamble_hex="2BEEEEBF")
    Si = preamble.preamble_sI
    Sq = preamble.preamble_sQ

    print("Si: ", ''.join(str(int(b)) for b in Si))
    print("Sq: ", ''.join(str(int(b)) for b in Sq))

    fig_preamble, grid_preamble = create_figure(2, 1, figsize=(16, 9))

    BitsPlot(
        fig_preamble, grid_preamble, (0,0),
        bits_list=[Si],
        sections=[("$S_I$", len(Si))],
        colors=["darkgreen"],
        ylabel="Channel $I$"
    ).plot()
    
    BitsPlot(
        fig_preamble, grid_preamble, (1,0),
        bits_list=[Sq],
        sections=[("$S_Q$", len(Sq))],
        colors=["navy"],
        xlabel="Bit Index", 
        ylabel="Channel $Q$"
    ).plot()
    
    fig_preamble.tight_layout()
    save_figure(fig_preamble, "example_preamble.pdf")
        
    