"""
Implementação do multiplexador. O multiplexador concatena os vetores I e Q de dois canais, conforme o padrão PPT-A3.

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from .plotter import create_figure, save_figure, BitsPlot

class Multiplexer:
    def __init__(self):
        r"""
        Inicializa o multiplexador no padrão ARGOS-3.
        
        Exemplo: 
            ![pageplot](assets/example_mux.svg)
        """
        pass

    def concatenate(self, SI, SQ, Xn, Yn):
        r"""
        Concatena os vetores $X_n$ e $Y_n$ de entrada, com $S_I$ e $S_Q$, retornando os vetores concatenados $X_n$ e $Y_n$. O processo de multiplexação é dado pela expressão abaixo.

        $$
        \begin{align}
        X_n = S_I \oplus X_n \text{ , } \quad Y_n = S_Q \oplus Y_n
        \end{align}
        $$

        Args:
            SI (np.ndarray): Vetor de entrada $S_I$.
            SQ (np.ndarray): Vetor de entrada $S_Q$.
            Xn (np.ndarray): Vetor de entrada $X_n$.
            Yn (np.ndarray): Vetor de entrada $Y_n$.

        Returns:
            Xn (np.ndarray): Vetor $X_n$ concatenado.
            Yn (np.ndarray): Vetor $Y_n$ concatenado.
        
        Raises:
            AssertionError: Se os vetores I e Q não tiverem o mesmo comprimento em ambos os canais.
        """
        assert len(SI) == len(SQ) and len(Xn) == len(Yn), "Os vetores I e Q devem ter o mesmo comprimento em ambos os canais."

        Xn = np.concatenate((SI, Xn))
        Yn = np.concatenate((SQ, Yn))

        return Xn, Yn

# Exemplo de uso
if __name__ == "__main__":

    mux = Multiplexer()

    SI = np.random.randint(0, 2, 15)
    SQ = np.random.randint(0, 2, 15)
    X = np.random.randint(0, 2, 60)
    Y = np.random.randint(0, 2, 60)
    print("SI:", ''.join(str(int(b)) for b in SI))
    print("SQ:", ''.join(str(int(b)) for b in SQ))
    print("X: ", ''.join(str(int(b)) for b in X))
    print("Y: ", ''.join(str(int(b)) for b in Y))

    fig_mux, grid_mux = create_figure(2, 1, figsize=(16, 9))

    BitsPlot(
        fig_mux, grid_mux, (0,0),
        bits_list=[SI, X],
        sections=[("Preambulo $S_I$", len(SI)),
                  ("Canal I $(X_n)$", len(X))],
        colors=["blue", "purple"]
    ).plot(ylabel="Canal $I$")
    
    Xn, Yn = mux.concatenate(SI, SQ, X, Y)

    BitsPlot(
        fig_mux, grid_mux, (1,0),
        bits_list=[SQ, Y],
        sections=[("Preambulo $S_Q$", len(SQ)),
                  ("Canal Q $(Y_n)$", len(Y))],
        colors=["blue", "purple"]
    ).plot(xlabel="Index de Bit", ylabel="Canal $Q$")

    fig_mux.tight_layout()
    save_figure(fig_mux, "example_mux.pdf")

    print("Xn:", ''.join(str(int(b)) for b in Xn))
    print("Yn:", ''.join(str(int(b)) for b in Yn))
