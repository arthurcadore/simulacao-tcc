# """
# Channel encoder for I and Q channels using NRZ and Manchester encoding according to the PPT-A3 standard.
#
# Author: Arthur Cadore
# Date: 28-07-2025
# """

import numpy as np
from .plotter import BitsPlot, SymbolsPlot, create_figure, save_figure

class Encoder:
    def __init__(self, method="NRZ"):
        r"""
        Initializes the channel encoder with the specified encoding method, used to encode the bitstream into symbols.

        Args:
            method (str): Encoding method, 'NRZ' or 'Manchester'.

        Raises:
            ValueError: If the encoding method is not supported.

        Examples: 
            >>> import argos3
            >>> import numpy as np
            >>> 
            >>> Xn = np.random.randint(0, 2, 20)
            >>> Yn = np.random.randint(0, 2, 20)
            >>> 
            >>> print(Xn)
            [1 0 1 1 0 0 1 0 0 0 0 0 0 0 1 0 1 1 0 0]
            >>> print(Yn)
            [1 0 0 0 0 0 1 0 1 0 0 0 0 1 1 1 0 1 1 1]
            >>> 
            >>> encoder_nrz = argos3.Encoder(method="NRZ")
            >>> encoder_man = argos3.Encoder(method="Manchester")
            >>> 
            >>> Xnrz = encoder_nrz.encode(Xn)
            >>> Yman = encoder_man.encode(Yn)
            >>> 
            >>> print(Xnrz)
            [ 1 -1  1  1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1  1 -1  1  1 -1 -1]
            >>> 
            >>> print(Yman)
            [ 1 -1 -1  1 -1  1 -1  1 -1  1 -1  1  1 -1 -1  1  1 -1 -1  1 -1  1 -1  1
             -1  1  1 -1  1 -1  1 -1 -1  1  1 -1  1 -1  1 -1]
            >>> 
            >>> Xn_prime = encoder_nrz.decode(Xnrz)
            >>> Yn_prime = encoder_man.decode(Yman)
            >>> 
            >>> print(Xn_prime)
            [1 0 1 1 0 0 1 0 0 0 0 0 0 0 1 0 1 1 0 0]
            >>> print(Yn_prime)
            [1 0 0 0 0 0 1 0 1 0 0 0 0 1 1 1 0 1 1 1]
            
            - Bitstream Plot Example: ![pageplot](assets/example_encoder_time.svg)

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (section 3.2.4)
        </div>
        """
        method_map = {
            "nrz": 0,
            "manchester": 1
        }

        method = method.lower()
        if method not in method_map:
            raise ValueError("Invalid encoding method. Use 'NRZ' or 'Manchester'.")
                
        self.method = method_map[method]

    def encode(self, bitstream):
        r"""
        Encodes the bitstream using the specified encoding method. The encoding process is given by the expressions below corresponding to each method.

        $$
        \begin{equation}
        \begin{aligned}
        X_{\text{NRZ}}[n] &= 
        \begin{cases}
        +1, & \text{if } X_n = 1 \\
        -1, & \text{if } X_n = 0 ,
        \end{cases}
        &\quad\quad
        X_{\text{MAN}}[n] &=
        \begin{cases}
        +1,-1, & \text{if } X_n = 1 \\
        -1, +1, & \text{if } X_n = 0 .
        \end{cases}
        \end{aligned}
        \end{equation}
        $$

        Where:
            - $X_n$: Input bitstream.
            - $X_{\text{NRZ}}[n]$ or $X_{\text{MAN}}[n]$: Output symbol stream.

        Args:
            bitstream (np.ndarray): Input bitstream.

        Returns:
            out (np.ndarray): Encoded symbol stream.
        """

        if self.method == 0:  # NRZ
            out = np.empty(bitstream.size, dtype=int)
            for i, bit in enumerate(bitstream):
                if bit == 0:
                    out[i] = -1
                elif bit == 1:
                    out[i] = +1

        elif self.method == 1:  # Manchester
            out = np.empty(bitstream.size * 2, dtype=int)
            for i, bit in enumerate(bitstream):
                if bit == 0:
                    out[2*i] = -1
                    out[2*i + 1] = +1
                elif bit == 1:
                    out[2*i] = +1
                    out[2*i + 1] = -1

        else:
            raise ValueError(f"Encoding method not implemented: {self.method}")

        return out


    def decode(self, encodedstream):
        r"""
        Decodes the symbol stream using the specified encoding method. The decoding process is given by the expressions below corresponding to each method.

        $$
        \begin{equation}
        \begin{aligned}
        X_n &= 
        \begin{cases}
        1, & \text{if } X_{\text{NRZ}}[n] = +1 \\
        0, & \text{if } X_{\text{NRZ}}[n] = -1
        \end{cases}
        &\quad\quad
        X_n &=
        \begin{cases}
        1, & \text{if } X_{\text{MAN}}[n] = +1, -1 \\
        0, & \text{if } X_{\text{MAN}}[n] = -1, +1
        \end{cases}
        \end{aligned}
        \end{equation}
        $$
        
        Where: 
            - $X_{\text{NRZ}}[n]$ or $X_{\text{MAN}}[n]$: Input symbol stream
            - $X_n$: Output bitstream.

        Args:
            encoded_stream (np.ndarray): Input symbol stream.

        Returns:
            out (np.ndarray): Decoded bitstream.

        """

        if self.method == 0:  # NRZ
            n = encodedstream.size 
            decoded = np.empty(n, dtype=int)
            for i in range(n):
                if encodedstream[i] <= 0:
                    decoded[i] = 0
                else:
                    decoded[i] = 1


        elif self.method == 1:  # Manchester
            n = encodedstream.size // 2
            decoded = np.empty(n, dtype=int)
            for i in range(n):
                pair = encodedstream[2*i:2*i + 2]
                if np.array_equal(pair, [-1, 1]):
                    decoded[i] = 0
                else:
                    decoded[i] = 1

        else:
            raise ValueError(f"Decoding method not implemented: {self.method}")

        return decoded

if __name__ == "__main__":

    Xn = np.random.randint(0, 2, 20)
    Yn = np.random.randint(0, 2, 20)
    print("Channel Xn: ", ''.join(str(int(b)) for b in Xn))
    print("Channel Yn: ", ''.join(str(int(b)) for b in Yn))

    encoder_nrz = Encoder(method="NRZ")
    encoder_man = Encoder(method="Manchester")

    Xnrz = encoder_nrz.encode(Xn)
    Yman = encoder_man.encode(Yn)

    print("Channel X(NRZ)[n]:", ' '.join(f"{x:+d}" for x in Xnrz[:10]))
    print("Channel Y(MAN)[n]:", ' '.join(f"{y:+d}" for y in Yman[:10]))

    fig_encoder, grid = create_figure(4, 1, figsize=(16, 9))

    BitsPlot(
        fig_encoder, grid, (0, 0),
        bits_list=[Xn],
        sections=[("$X_n$", len(Xn))],
        colors=["darkgreen"],
        xlabel="Bit Index", 
        ylabel="$X_n$"
    ).plot()

    SymbolsPlot(
        fig_encoder, grid, (1, 0),
        symbols_list=[Xnrz],
        samples_per_symbol=1,
        colors=["darkgreen"],
        xlabel="Symbol Index",
        ylabel="$X_{NRZ}[n]$", 
        label="$X_{NRZ}[n]$"
    ).plot()

    BitsPlot(
        fig_encoder, grid, (2, 0),
        bits_list=[Yn],
        sections=[("$Y_n$", len(Yn))],
        colors=["navy"],
        xlabel="Bit Index", 
        ylabel="$Y_n$"
    ).plot()

    SymbolsPlot(
        fig_encoder, grid, (3, 0),
        symbols_list=[Yman],
        samples_per_symbol=2,
        colors=["navy"],
        xlabel="Symbol Index",
        ylabel="$Y_{MAN}[n]$", 
        label="$Y_{MAN}[n]$"
    ).plot()


    fig_encoder.tight_layout()
    save_figure(fig_encoder, "example_encoder_time.pdf")

    Xn_prime = encoder_nrz.decode(Xnrz)
    print("Channel X'n:", ''.join(str(int(b)) for b in Xn_prime))
    Yn_prime = encoder_man.decode(Yman)
    print("Channel Y'n:", ''.join(str(int(b)) for b in Yn_prime))

    print("Xn = X'n: ", np.array_equal(Xn, Xn_prime))
    print("Yn = Y'n: ", np.array_equal(Yn, Yn_prime))