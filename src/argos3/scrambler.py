"""
Implementação do embaralhador e desembaralhador compatível com o padrão PPT-A3.

Referência:
    AS3-SP-516-274-CNES (3.1.4.5)

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from .plotter import save_figure, create_figure, BitsPlot

class Scrambler:
    def __init__(self):
        r"""
        Inicializa o embaralhador no padrão ARGOS-3.

        Exemplo: 
            ![pageplot](assets/example_scrambler_time.svg)

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.1.4.5)
        </div>
        """
        pass

    def scramble(self, vt0, vt1):
        r"""
        Embaralha os vetores $v_t^{(0)}$ e $v_t^{(1)}$, retornando os vetores $X_n$ e $Y_n$ embaralhados. O processo de embaralhamento é dado pela expressão abaixo.

        \begin{equation}
            X_n = \begin{cases}
            A, & \text{se } n = 0 \pmod{3} \\
            B, & \text{se } n = 1 \pmod{3} \\
            C, & \text{se } n = 2 \pmod{3}
            \end{cases} \quad
            Y_n = \begin{cases}
            A, & \text{se } n = 0 \pmod{3} \\
            B, & \text{se } n = 1 \pmod{3} \\
            C, & \text{se } n = 2 \pmod{3}
            \end{cases}
        \end{equation}

        Sendo: 
            - $X_n$ e $Y_n$: Vetores de saída embaralhados.
            - $A$, $B$ e $C$: Combinação de bits dos vetores de entrada $v_t^{(0)}$ e $v_t^{(1)}$.
            - $n$: Indice do bit a ser embaralhado.

        O processo de embaralhamento é ilustrado pelo diagrama de blocos abaixo. 

        ![pageplot](../assets/embaralhador.svg)
        
        Args:
            vt0 (np.ndarray): Vetor de entrada $v_t^{(0)}$.
            vt1 (np.ndarray): Vetor de entrada $v_t^{(1)}$.

        Returns:
            X_scrambled (np.ndarray): Vetor $X_n$ embaralhado.
            Y_scrambled (np.ndarray): Vetor $Y_n$ embaralhado.

        Raises:
            AssertionError: Se os vetores X e Y não tiverem o mesmo comprimento.
        """
        assert len(vt0) == len(vt1), "Vetores X e Y devem ter o mesmo comprimento"
        X_scrambled = []
        Y_scrambled = []

        for i in range(0, len(vt0), 3):
            x_blk = vt0[i:i+3]
            y_blk = vt1[i:i+3]
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

    def descramble(self, X_prime, Y_prime):
        r"""
        Desembaralha os vetores $X'_n$ e $Y'_n$ embaralhados, retornando os vetores $v_t^{(0)'}$ e $v_t^{(1)'}$ restaurados. O processo de desembaralhamento
        é dado pela expressão abaixo.

        \begin{equation}
            v_t^{(0)'} = \begin{cases}
            A, & \text{se } n = 0 \pmod{3} \\
            B, & \text{se } n = 1 \pmod{3} \\
            C, & \text{se } n = 2 \pmod{3}
            \end{cases}, \quad
            v_t^{(1)'} = \begin{cases}
            A, & \text{se } n = 0 \pmod{3} \\
            B, & \text{se } n = 1 \pmod{3} \\
            C, & \text{se } n = 2 \pmod{3}
            \end{cases} \text{ .}
            \label{eq:desembaralhador_Y}
        \end{equation}

        Sendo: 
            - $v_t^{(0)'}$ e $v_t^{(1)'}$: Vetores de saida desembaralhados.
            - $A$, $B$ e $C$: Combinação de bits dos vetores de entrada $X'_n$ e $Y'_n$ embaralhados.
            - $n$: Indice do bit a ser embaralhado.

        O processo de desembaralhamento é ilustrado pelo diagrama de blocos abaixo.

        ![pageplot](../assets/desembaralhador.svg)

        Args:
            X_prime (np.ndarray): Vetor $X'_{n}$ embaralhado.
            Y_prime (np.ndarray): Vetor $Y'_{n}$ embaralhado.

        Returns:
            vt0_prime (np.ndarray): Vetor $v_t^{(0)}$ restaurado.
            vt1_prime (np.ndarray): Vetor $v_t^{(1)}$ restaurado.
        
        Raises:
            AssertionError: Se os vetores X e Y não tiverem o mesmo comprimento.
        """
        assert len(X_prime) == len(Y_prime), "Vetores X e Y devem ter o mesmo comprimento"
        vt0_prime = []
        vt1_prime = []

        for i in range(0, len(X_prime), 3):
            x_blk = X_prime[i:i+3]
            y_blk = Y_prime[i:i+3]
            n = len(x_blk)

            if n == 3:
                # Desembaralhamento do bloco [y1, x2, y2], [x1, x3, y3]
                x1, x2, x3 = y_blk[0], x_blk[1], y_blk[1]
                y1, y2, y3 = x_blk[0], x_blk[2], y_blk[2]
                vt0_prime.extend([x1, x2, x3])
                vt1_prime.extend([y1, y2, y3])
            elif n == 2:
                # Desembaralhamento do bloco [y1, x2], [x1, y2]
                x1, x2 = y_blk[0], x_blk[1]
                y1, y2 = x_blk[0], y_blk[1]
                vt0_prime.extend([x1, x2])
                vt1_prime.extend([y1, y2])
            elif n == 1:
                # Desembaralhamento do bloco [y1], [x1]
                x1 = y_blk[0]
                y1 = x_blk[0]
                vt0_prime.append(x1)
                vt1_prime.append(y1)

        return vt0_prime, vt1_prime



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

    fig_scrambler, grid_scrambler = create_figure(3, 2, figsize=(16, 9))

    BitsPlot(
        fig_scrambler, grid_scrambler, (0, 0),
        bits_list=[vt0],
        sections=[("$v_t^{0}$", len(vt0))],
        colors=["darkgreen"]
    ).plot(ylabel="Original")

    BitsPlot(
        fig_scrambler, grid_scrambler, (0, 1),
        bits_list=[vt1],
        sections=[("$v_t^{1}$", len(vt1))],
        colors=["navy"]
    ).plot()

    BitsPlot(
        fig_scrambler, grid_scrambler, (1, 0),
        bits_list=[Xn],
        sections=[("$X_n$", len(Xn))],
        colors=["darkgreen"]
    ).plot(ylabel="Embaralhado")

    BitsPlot(
        fig_scrambler, grid_scrambler, (1, 1),
        bits_list=[Yn],
        sections=[("$Y_n$", len(Yn))],
        colors=["navy"]
    ).plot()

    BitsPlot(
        fig_scrambler, grid_scrambler, (2, 0),
        bits_list=[vt0_prime],
        sections=[("$v_t^{0}$", len(vt0_prime))],
        colors=["darkgreen"]
    ).plot(ylabel="Restaurado", xlabel="Index de Bit")

    BitsPlot(
        fig_scrambler, grid_scrambler, (2, 1),
        bits_list=[vt1_prime],
        sections=[("$v_t^{1}$", len(vt1_prime))],
        colors=["navy"]
    ).plot(xlabel="Index de Bit")

    fig_scrambler.tight_layout()
    save_figure(fig_scrambler, "example_scrambler_time.pdf")