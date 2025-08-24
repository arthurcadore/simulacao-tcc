"""
Implementa uma palavra de sincronismo compatível com o padrão PPT-A3.

Referência:
    AS3-SP-516-274-CNES (seção 3.1.4.6)

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from .plotter import save_figure, create_figure, BitsPlot

class Preamble:

    def __init__(self, preamble_hex="2BEEEEBF"):
        r"""
        Gera uma palavra de sincronismo, $S = 2BEEEEBF_{16}$ no padrão ARGOS-3. A palavra de sincronismo composta por 30 bits, $S = [S_0, S_1, S_2, \dots, S_{29}]$ que são intercalados para formar os vetores $S_I$ e $S_Q$ de cada canal, conforme apresentado abaixo.

        $$
        \begin{align}
        S_I &= [S_0,\, S_2,\, S_4,\, \dots,\, S_{28}] && \mapsto \quad S_I = [1111,\, 1111,\, 1111,\, 111] \\
        S_Q &= [S_1,\, S_3,\, S_5,\, \dots,\, S_{29}] && \mapsto \quad S_Q = [0011,\, 0101,\, 0100,\, 111]
        \end{align}
        $$

        Sendo:
            - $S$: Palavra de sincronismo original.
            - $S_I$ e $S_Q$: Vetores de saida correspondentes aos canais I e Q, respectivamente.

        Args:
            preamble_hex (str, opcional): Hexadecimal da palavra de sincronismo.
        
        Raises:
            ValueError: Se a palavra de sincronismo $S$ tiver comprimento diferente de 8 caracteres. 
            ValueError: Se o hexadecimal não for válido ou não puder ser convertido.

        Exemplo: 
            ![pageplot](assets/example_preamble.svg)

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.1.4.6)
        </div>
        """

        if not isinstance(preamble_hex, str) or len(preamble_hex) != 8:
            raise ValueError("O hexadecimal da palavra de sincronismo deve ser uma string de 8 caracteres.")

        self.preamble_hex = preamble_hex
        self.preamble_bits = self.hex_to_bits(self.preamble_hex)

        if len(self.preamble_bits) != 30:
            raise ValueError("A palavra de sincronismo deve conter 30 bits.")

        self.preamble_sI, self.preamble_sQ = self.generate_preamble()

    def hex_to_bits(self, hex_string):
        return format(int(hex_string, 16), '032b')[2:] 
    
    def generate_preamble(self):
        r"""
        Gera os vetores $S_I$ e $S_Q$ da palavra de sincronismo, com base no vetor $S$ passado no construtor.

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

    fig_preamble, grid_preamble = create_figure(2, 1, figsize=(16, 9))

    BitsPlot(
        fig_preamble, grid_preamble, (0,0),
        bits_list=[Si],
        sections=[("Preambulo $S_I$", len(Si))],
        colors=["darkgreen"]
    ).plot(ylabel="Canal $I$")
    
    BitsPlot(
        fig_preamble, grid_preamble, (1,0),
        bits_list=[Sq],
        sections=[("Preambulo $S_Q$", len(Sq))],
        colors=["navy"]
    ).plot(xlabel="Index de Bit", ylabel="Canal $Q$")
    
    fig_preamble.tight_layout()
    save_figure(fig_preamble, "example_preamble.pdf")
        
    