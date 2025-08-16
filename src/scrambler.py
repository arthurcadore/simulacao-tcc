"""
Implementação do embaralhador e desembaralhador compatível com o padrão PPT-A3.

Referência:
    AS3-SP-516-274-CNES (3.1.4.5)

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from plots import Plotter

class Scrambler:
    def __init__(self):
        r"""
        Implementação do embaralhador e desembaralhador compatível com o padrão PPT-A3.
        """
        pass

    def scramble(self, X, Y):
        r"""
        Embaralha os vetores X e Y de mesmo comprimento, retornando os vetores embaralhados.

        Args:
            X (np.ndarray): Vetor de entrada $v_t^{(0)}$.
            Y (np.ndarray): Vetor de entrada $v_t^{(1)}$.

        Returns:
            X_scrambled (np.ndarray): Vetor $X_n$ embaralhado.
            Y_scrambled (np.ndarray): Vetor $Y_n$ embaralhado.

        Raises:
            AssertionError: Se os vetores X e Y não tiverem o mesmo comprimento.
        """
        assert len(X) == len(Y), "Vetores X e Y devem ter o mesmo comprimento"
        X_scrambled = []
        Y_scrambled = []

        for i in range(0, len(X), 3):
            x_blk = X[i:i+3]
            y_blk = Y[i:i+3]
            n = len(x_blk)

            if n == 3:
                # Embaralhamento do bloco [x1, x2, x3], [y1, y2, y3]
                x1, x2, x3 = x_blk
                y1, y2, y3 = y_blk
                X_scrambled += [y1, x2, y2]
                Y_scrambled += [x1, x3, y3]
            elif n == 2:
                # Embaralhamento do bloco [x1, x2], [y1, y2]
                x1, x2 = x_blk
                y1, y2 = y_blk
                X_scrambled += [y1, x2]
                Y_scrambled += [x1, y2]
            elif n == 1:
                # Embaralhamento do bloco [x1], [y1]
                x1 = x_blk[0]
                y1 = y_blk[0]
                X_scrambled += [y1]
                Y_scrambled += [x1]

        return X_scrambled, Y_scrambled

    def descramble(self, X, Y):
        r"""
        Restaura os vetores X e Y embaralhados ao seu estado original.

        Args:
            X (np.ndarray): Vetor $X'_{n}$ embaralhado.
            Y (np.ndarray): Vetor $Y'_{n}$ embaralhado.

        Returns:
           msg_length (np.ndarray): Vetor de 4 bits representando o campo Message Length.

        Returns:
            X_original (np.ndarray): Vetor $v_t^{(0)}$ restaurado.
            Y_original (np.ndarray): Vetor $v_t^{(1)}$ restaurado.
        
        Raises:
            AssertionError: Se os vetores X e Y não tiverem o mesmo comprimento.
        """
        assert len(X) == len(Y), "Vetores X e Y devem ter o mesmo comprimento"
        X_original = []
        Y_original = []

        for i in range(0, len(X), 3):
            x_blk = X[i:i+3]
            y_blk = Y[i:i+3]
            n = len(x_blk)

            if n == 3:
                # Desembaralhamento do bloco [y1, x2, y2], [x1, x3, y3]
                x1, x2, x3 = y_blk[0], x_blk[1], y_blk[1]
                y1, y2, y3 = x_blk[0], x_blk[2], y_blk[2]
                X_original.extend([x1, x2, x3])
                Y_original.extend([y1, y2, y3])
            elif n == 2:
                # Desembaralhamento do bloco [y1, x2], [x1, y2]
                x1, x2 = y_blk[0], x_blk[1]
                y1, y2 = x_blk[0], y_blk[1]
                X_original.extend([x1, x2])
                Y_original.extend([y1, y2])
            elif n == 1:
                # Desembaralhamento do bloco [y1], [x1]
                x1 = y_blk[0]
                y1 = x_blk[0]
                X_original.append(x1)
                Y_original.append(y1)

        return X_original, Y_original



if __name__ == "__main__":
    vt0 = np.random.randint(0, 2, 30)
    vt1 = np.random.randint(0, 2, 30)
    idx_vt0 = [f"X{i+1}" for i in range(len(vt0))]
    idx_vt1 = [f"Y{i+1}" for i in range(len(vt1))]

    # Embaralha o conteúdo dos vetores e os indices
    scrambler = Scrambler()
    Xn, Yn = scrambler.scramble(vt0, vt1)
    idx_Xn, idx_Yn = scrambler.scramble(idx_vt0, idx_vt1)

    print("\nSequência original:")
    print("vt0: ", ''.join(str(b) for b in vt0))
    print("vt1: ", ''.join(str(b) for b in vt1))
    print("idx_vt0:", idx_vt0[:12])
    print("idx_vt1:", idx_vt1[:12])

    print("\nSequência embaralhada:")
    print("Xn  :", ''.join(str(int(b)) for b in Xn))
    print("Yn  :", ''.join(str(int(b)) for b in Yn))
    print("idx_Xn: ", idx_Xn[:12])
    print("idx_Yn: ", idx_Yn[:12])

    # Desembaralha o conteúdo dos vetores e os indices
    vt0_prime, vt1_prime = scrambler.descramble(Xn, Yn)
    idx_vt0_prime, idx_vt1_prime = scrambler.descramble(idx_Xn, idx_Yn)

    print("\nVerificação:")
    print("vt0':", ''.join(str(int(b)) for b in vt0_prime))
    print("vt1':", ''.join(str(int(b)) for b in vt1_prime))
    print("idx_vt0': ", idx_vt0_prime[:12])
    print("idx_vt1': ", idx_vt1_prime[:12])
    print("vt0 = vt0': ", np.array_equal(vt0, vt0_prime))
    print("vt1 = vt1': ", np.array_equal(vt1, vt1_prime))
    print("idx_vt0 = idx_vt0': ", np.array_equal(idx_vt0, idx_vt0_prime))
    print("idx_vt1 = idx_vt1': ", np.array_equal(idx_vt1, idx_vt1_prime))
    
    plotter = Plotter()
    plotter.plot_scrambler(vt0, 
                           vt1, 
                           Xn, 
                           Yn, 
                           vt0_prime, 
                           vt1_prime,
                           "Canal I $v_t^{(0)}$", 
                           "Canal Q $v_t^{(1)}$", 
                           "Canal I $(X_n)$", 
                           "Canal Q $(Y_n)$", 
                           "Canal I $v_t^{(0)'} $", 
                           "Canal Q $v_t^{(1)'} $",
                           save_path="../out/example_scrambling.pdf"
    )

