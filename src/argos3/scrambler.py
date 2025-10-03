# """
# Implementação do embaralhador e desembaralhador compatível com o padrão PPT-A3.

# Referência:
#     AS3-SP-516-274-CNES (3.1.4.5)

# Autor: Arthur Cadore
# Data: 28-07-2025
# """

import numpy as np
from .plotter import save_figure, create_figure, BitsPlot
from .env_vars import *

class Scrambler:
    def __init__(self):
        r"""
        Scramble the vectors $v_t^{(0)}$ and $v_t^{(1)}$, returning the vectors $X[n]$ and $Y[n]$ scrambled. The scrambling process is given by the expression below.

        \begin{equation}
            X[n] = \begin{cases}
            A, & \text{if } n = 0 \pmod{3} \\
            B, & \text{if } n = 1 \pmod{3} \\
            C, & \text{if } n = 2 \pmod{3}
            \end{cases} \quad
            Y[n] = \begin{cases}
            A, & \text{if } n = 0 \pmod{3} \\
            B, & \text{if } n = 1 \pmod{3} \\
            C, & \text{if } n = 2 \pmod{3}
            \end{cases}
        \end{equation}

        Where: 
            - $X[n]$ and $Y[n]$: Scrambled output vectors.
            - $A$, $B$ and $C$: Combination of bits of the input vectors $v_t^{(0)}$ and $v_t^{(1)}$.
            - $n$: Index of the bit to be scrambled.

        The scrambling process is illustrated by the block diagram below. 

        ![pageplot](../assets/scrambler.svg)
        

        Examples: 
            >>> import argos3
            >>> import numpy as np
            >>> 
            >>> vt0 = np.random.randint(0, 2, 15)
            >>> vt1 = np.random.randint(0, 2, 15)
            >>> 
            >>> idx_vt0 = [f"X{i+1}" for i in range(len(vt0))]
            >>> idx_vt1 = [f"Y{i+1}" for i in range(len(vt1))]
            >>> 
            >>> scrambler = argos3.Scrambler()
            >>> Xn, Yn = scrambler.scramble(vt0, vt1)
            >>> idx_Xn, idx_Yn = scrambler.scramble(idx_vt0, idx_vt1)
            >>> 
            >>> print(vt0)
            [0 1 1 1 1 0 0 0 1 0 0 0 0 0 1]
            >>> print(vt1)
            [1 0 0 1 1 1 0 1 0 1 0 0 0 1 1]
            >>> print(idx_vt0)
            ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15']
            >>> print(idx_vt1)
            ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15']
            >>> 
            >>> print(idx_Xn)
            ['Y1', 'X2', 'Y2', 'Y4', 'X5', 'Y5', 'Y7', 'X8', 'Y8', 'Y10', 'X11', 'Y11', 'Y13', 'X14', 'Y14']
            >>> print(idx_Yn)
            ['X1', 'X3', 'Y3', 'X4', 'X6', 'Y6', 'X7', 'X9', 'Y9', 'X10', 'X12', 'Y12', 'X13', 'X15', 'Y15']
            >>>
            >>> vt0_prime, vt1_prime = scrambler.descramble(Xn,Yn)
            >>> idx_vt0_prime, idx_vt1_prime = scrambler.descramble(idx_Xn, idx_Yn)
            >>> 
            >>> print(idx_vt0_prime)
            ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15']
            >>> print(idx_vt1_prime)
            ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15']
             
            
            - Bitstream Plot Example: ![pageplot](assets/example_scrambler_time.svg)

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.1.4.5)
        </div>
        """
        pass

    def scramble(self, vt0, vt1):
        r"""
        Receive the vectors $v_t^{(0)}$ and $v_t^{(1)}$ as input and return the vectors $X[n]$ and $Y[n]$ scrambled.

        Args:
            vt0 (np.ndarray): Input vector $v_t^{(0)}$.
            vt1 (np.ndarray): Input vector $v_t^{(1)}$.

        Returns:
            X_scrambled (np.ndarray): Scrambled vector $X[n]$.
            Y_scrambled (np.ndarray): Scrambled vector $Y[n]$.

        Raises:
            AssertionError: If the vectors X and Y do not have the same length.
        """
        assert len(vt0) == len(vt1), "Vectors X and Y must have the same length"
        X_scrambled = []
        Y_scrambled = []

        for i in range(0, len(vt0), 3):
            x_blk = vt0[i:i+3]
            y_blk = vt1[i:i+3]
            n = len(x_blk)

            if n == 3:
                # Scrambling of the block [x1, x2, x3], [y1, y2, y3]
                x1, x2, x3 = x_blk
                y1, y2, y3 = y_blk
                X_scrambled += [y1, x2, y2]
                Y_scrambled += [x1, x3, y3]
            elif n == 2:
                # Scrambling of the block [x1, x2], [y1, y2]
                x1, x2 = x_blk
                y1, y2 = y_blk
                X_scrambled += [y1, x2]
                Y_scrambled += [x1, y2]
            elif n == 1:
                # Scrambling of the block [x1], [y1]
                x1 = x_blk[0]
                y1 = y_blk[0]
                X_scrambled += [y1]
                Y_scrambled += [x1]

        return X_scrambled, Y_scrambled

    def unscramble(self, X_prime, Y_prime):
        r"""
        Receive the vectors $X'[n]$ and $Y'[n]$ scrambled and return the vectors $v_t^{(0)'}$ and $v_t^{(1)'}$ restored.

        \begin{equation}
            v_t^{(0)'} = \begin{cases}
            A, & \text{if } n = 0 \pmod{3} \\
            B, & \text{if } n = 1 \pmod{3} \\
            C, & \text{if } n = 2 \pmod{3}
            \end{cases}, \quad
            v_t^{(1)'} = \begin{cases}
            A, & \text{if } n = 0 \pmod{3} \\
            B, & \text{if } n = 1 \pmod{3} \\
            C, & \text{if } n = 2 \pmod{3}
            \end{cases} \text{ .}
            \label{eq:desembaralhador_Y}
        \end{equation}

        Where: 
            - $v_t^{(0)'}$ and $v_t^{(1)'}$: Output vectors restored.
            - $A$, $B$ and $C$: Combination of bits of the input vectors $X'[n]$ and $Y'[n]$ scrambled.
            - $n$: Index of the bit to be scrambled.

        The descrambling process is illustrated by the block diagram below.

        ![pageplot](../assets/unscrewer.svg)

        Args:
            X_prime (np.ndarray): Vector $X'_{n}$ scrambled.
            Y_prime (np.ndarray): Vector $Y'_{n}$ scrambled.

        Returns:
            vt0_prime (np.ndarray): Vector $v_t^{(0)}$ restored.
            vt1_prime (np.ndarray): Vector $v_t^{(1)}$ restored.
        
        Raises:
            AssertionError: If the vectors X and Y do not have the same length.
        """
        assert len(X_prime) == len(Y_prime), "Vectors X and Y must have the same length"
        vt0_prime = []
        vt1_prime = []

        for i in range(0, len(X_prime), 3):
            x_blk = X_prime[i:i+3]
            y_blk = Y_prime[i:i+3]
            n = len(x_blk)

            if n == 3:
                # Descrambling of the block [y1, x2, y2], [x1, x3, y3]
                x1, x2, x3 = y_blk[0], x_blk[1], y_blk[1]
                y1, y2, y3 = x_blk[0], x_blk[2], y_blk[2]
                vt0_prime.extend([x1, x2, x3])
                vt1_prime.extend([y1, y2, y3])
            elif n == 2:
                # Descrambling of the block [y1, x2], [x1, y2]
                x1, x2 = y_blk[0], x_blk[1]
                y1, y2 = x_blk[0], y_blk[1]
                vt0_prime.extend([x1, x2])
                vt1_prime.extend([y1, y2])
            elif n == 1:
                # Descrambling of the block [y1], [x1]
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

    # Scramble the content of the vectors and the indices
    scrambler = Scrambler()
    Xn, Yn = scrambler.scramble(vt0, vt1)
    idx_Xn, idx_Yn = scrambler.scramble(idx_vt0, idx_vt1)

    print("\nOriginal sequence:")
    print("vt0: ", ''.join(str(b) for b in vt0))
    print("vt1: ", ''.join(str(b) for b in vt1))
    print("idx_vt0:", idx_vt0[:12])
    print("idx_vt1:", idx_vt1[:12])

    print("\nScrambled sequence:")
    print("Xn  :", ''.join(str(int(b)) for b in Xn))
    print("Yn  :", ''.join(str(int(b)) for b in Yn))
    print("idx_Xn: ", idx_Xn[:12])
    print("idx_Yn: ", idx_Yn[:12])

    # Descramble the content of the vectors and the indices
    vt0_prime, vt1_prime = scrambler.unscramble(Xn, Yn)
    idx_vt0_prime, idx_vt1_prime = scrambler.unscramble(idx_Xn, idx_Yn)

    print("\nVerification:")
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
        sections=[(r"$v_t^{0}$", len(vt0))],
        colors=[COLOR_I],
        ylabel=BITSTREAM_Y,
        title=I_CHANNEL_TITLE
    ).plot()

    BitsPlot(
        fig_scrambler, grid_scrambler, (0, 1),
        bits_list=[vt1],
        sections=[(r"$v_t^{1}$", len(vt1))],
        colors=[COLOR_Q],
        title=Q_CHANNEL_TITLE
    ).plot()

    BitsPlot(
        fig_scrambler, grid_scrambler, (1, 0),
        bits_list=[Xn],
        sections=[(r"$X[n]$", len(Xn))],
        colors=[COLOR_I],
        ylabel=BITSTREAM_Y
    ).plot()

    BitsPlot(
        fig_scrambler, grid_scrambler, (1, 1),
        bits_list=[Yn],
        sections=[(r"$Y[n]$", len(Yn))],
        colors=[COLOR_Q]
    ).plot()

    BitsPlot(
        fig_scrambler, grid_scrambler, (2, 0),
        bits_list=[vt0_prime],
        sections=[(r"$v_t^{0} \prime$", len(vt0_prime))],
        colors=[COLOR_I],
        ylabel=BITSTREAM_Y, xlabel=BITSTREAM_X
    ).plot()

    BitsPlot(
        fig_scrambler, grid_scrambler, (2, 1),
        bits_list=[vt1_prime],
        sections=[(r"$v_t^{1} \prime$", len(vt1_prime))],
        colors=[COLOR_Q],
        xlabel=BITSTREAM_X
    ).plot()

    fig_scrambler.tight_layout()
    save_figure(fig_scrambler, "example_scrambler_time.pdf")