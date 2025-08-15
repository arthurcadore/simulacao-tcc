"""
Implementação do multiplexador. O multiplexador concatena os vetores I e Q de dois canais, conforme o padrão PPT-A3.

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from plots import Plotter

class Multiplexer:
    def __init__(self):
        r"""
        Inicializa uma instância do multiplexador.
        """
        pass

    def concatenate(self, I1, Q1, I2, Q2):
        r"""
        Concatena os vetores I e Q de dois canais, retornando os vetores concatenados.

        Args:
            I1 (np.ndarray): Vetor I do primeiro canal.
            Q1 (np.ndarray): Vetor Q do primeiro canal.
            I2 (np.ndarray): Vetor I do segundo canal.
            Q2 (np.ndarray): Vetor Q do segundo canal.
            
        Returns:
            I (np.ndarray): Vetor I concatenado.
            Q (np.ndarray): Vetor Q concatenado.
        
        Raises:
            AssertionError: Se os vetores I e Q não tiverem o mesmo comprimento em ambos os canais.
        """
        assert len(I1) == len(Q1) and len(I2) == len(Q2), "Os vetores I e Q devem ter o mesmo comprimento em ambos os canais."

        I = np.concatenate((I1, I2))
        Q = np.concatenate((Q1, Q2))

        return I, Q

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

    Xn, Yn = mux.concatenate(SI, SQ, X, Y)

    plotter = Plotter()
    plotter.plot_mux(SI, 
                     SQ, 
                     X, 
                     Y, 
                     "Preambulo $S_I$",
                     "Canal I $(X_n)$",
                     "Preambulo $S_Q$",
                     "Canal Q $(Y_n)$",
                     "$X_n$",
                     "$Y_n$",
                     save_path="../out/example_multiplexing.pdf"
                     )

    print("Xn:", ''.join(str(int(b)) for b in Xn))
    print("Yn:", ''.join(str(int(b)) for b in Yn))
