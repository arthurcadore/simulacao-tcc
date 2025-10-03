# """
# Implements the multiplexer in the ARGOS-3 standard.
#
# Autor: Arthur Cadore
# Data: 28-07-2025
# """

import numpy as np
from .plotter import create_figure, save_figure, BitsPlot
from .env_vars import *

class Multiplexer:
    def __init__(self):
        r"""
        Initializes the multiplexer used on transmission. Used to concatenate the input data vectors $X[n]$ and $Y[n]$ with the preamble vectors $S_I[n]$ and $S_Q[n]$, returning the concatenated vectors $X[n]$ and $Y[n]$. The multiplexing process is given by the expression below.

        $$
        \begin{align}
        X[n] = S_I[n] \oplus X[n] \text{ , } \quad Y[n] = S_Q[n] \oplus Y[n]
        \end{align}
        $$
        
        Examples: 
            >>> import argos3
            >>> import numpy as np
            >>> 
            >>> Si = np.random.randint(0,2,15)
            >>> Sq = np.random.randint(0,2,15)
            >>> 
            >>> X = np.random.randint(0,2,30)
            >>> Y = np.random.randint(0,2,30)
            >>> 
            >>> mux = argos3.Multiplexer()
            >>> 
            >>> Xn, Yn = mux.concatenate(Si, Sq, X, Y)
            >>> 
            >>> print(Xn)
            [0 0 0 1 0 0 1 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 1 0 1 0 1 1 1
             0 1 0 1 0 0 0 1]
            >>> 
            >>> print(Yn)
            [0 1 1 0 0 0 1 1 0 0 0 1 1 0 0 1 1 0 0 0 1 1 0 1 1 0 1 1 0 1 1 0 0 1 0 1 1
             0 1 0 0 1 1 1 0]

            - Bitstream Plot Example: ![pageplot](assets/example_mux.svg)
        """
        pass

    def concatenate(self, SI, SQ, Xn, Yn):
        r"""
        Concatenates the input data vectors $X[n]$ and $Y[n]$ with the preamble vectors $S_I[n]$ and $S_Q[n]$, returning the concatenated vectors $X[n]$ and $Y[n]$. 

        Args:
            SI (np.ndarray): Input vector $S_I[n]$.
            SQ (np.ndarray): Input vector $S_Q[n]$.
            Xn (np.ndarray): Input vector $X[n]$.
            Yn (np.ndarray): Input vector $Y[n]$.

        Returns:
            Xn (np.ndarray): Concatenated vector $X[n]$.
            Yn (np.ndarray): Concatenated vector $Y[n]$.
        
        Raises:
            AssertionError: If the vectors I and Q do not have the same length in both channels.
        """
        assert len(SI) == len(SQ) and len(Xn) == len(Yn), "The vectors I and Q must have the same length in both channels."

        Xn = np.concatenate((SI, Xn))
        Yn = np.concatenate((SQ, Yn))

        return Xn, Yn

if __name__ == "__main__":

    SI = np.random.randint(0, 2, 15)
    SQ = np.random.randint(0, 2, 15)
    X = np.random.randint(0, 2, 60)
    Y = np.random.randint(0, 2, 60)

    print("SI:", ''.join(str(int(b)) for b in SI))
    print("SQ:", ''.join(str(int(b)) for b in SQ))
    print("X: ", ''.join(str(int(b)) for b in X))
    print("Y: ", ''.join(str(int(b)) for b in Y))

    mux = Multiplexer()
    Xn, Yn = mux.concatenate(SI, SQ, X, Y)

    print("Xn:", ''.join(str(int(b)) for b in Xn))
    print("Yn:", ''.join(str(int(b)) for b in Yn))

    fig_mux, grid_mux = create_figure(2, 1, figsize=(16, 9))

    BitsPlot(
        fig_mux, grid_mux, (0,0),
        bits_list=[SI, X],
        sections=[(r"$S_I[n]$", len(SI)),(r"$X[n]$", len(X))],
        colors=[COLOR_AUX1, COLOR_I],
        ylabel=BITSTREAM_Y,
        title=I_CHANNEL_TITLE
    ).plot()
    
    BitsPlot(
        fig_mux, grid_mux, (1,0),
        bits_list=[SQ, Y],
        sections=[(r"$S_Q[n]$", len(SQ)),(r"$Y[n]$", len(Y))],
        colors=[COLOR_AUX1, COLOR_Q],
        xlabel=BITSTREAM_X, 
        ylabel=BITSTREAM_Y,
        title=Q_CHANNEL_TITLE
    ).plot()

    fig_mux.tight_layout()
    save_figure(fig_mux, "example_mux.pdf")
