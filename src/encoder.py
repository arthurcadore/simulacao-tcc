"""
Codificação de canais I e Q usando NRZ e Manchester conforme o padrão PPT-A3.

Referência:
    AS3-SP-516-274-CNES (seção 3.2.4)

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from plotter import BitsPlot, EncodedBitsPlot, create_figure, save_figure

class Encoder:
    def __init__(self, method):
        r"""
        Inicializa uma instância do codificador com o método especificado.

        Referência:
            AS3-SP-516-274-CNES (seção 3.2.4)

        Args:
            method (str): Método de codificação ('NRZ' ou 'Manchester').

        Raises:
            ValueError: Se o método de codificação não for suportado.
        """
        method_map = {
            "nrz": 0,
            "manchester": 1
        }

        method = method.lower()
        if method not in method_map:
            raise ValueError("Método de codificação inválido. Use 'NRZ' ou 'Manchester'.")
                
        self.method = method_map[method]

    def encode(self, bitstream):
        r"""
        Codifica o vetor de bits usando o método especificado na inicialização.

        Args:
            bitstream (np.ndarray): Vetor de bits a ser codificado.

        Returns:
            out (np.ndarray): Vetor de bits codificado.
        """
        out = np.empty(bitstream.size * 2, dtype=int)

        if self.method == 0:  # NRZ
            for i, bit in enumerate(bitstream):
                if bit == 0:
                    out[2*i] = 0
                    out[2*i + 1] = 0
                elif bit == 1:
                    out[2*i] = 1
                    out[2*i + 1] = 1

        elif self.method == 1:  # Manchester
            for i, bit in enumerate(bitstream):
                if bit == 0:
                    out[2*i] = 0
                    out[2*i + 1] = 1
                elif bit == 1:
                    out[2*i] = 1
                    out[2*i + 1] = 0

        else:
            raise ValueError(f"Método de codificação não implementado: {self.method}")

        return out


    def decode(self, encoded_stream):
        r"""
        Decodifica o vetor codificado de volta para o vetor original.

        Args:
            encoded_stream (np.ndarray): Vetor codificado a ser decodificado.

        Returns:
            out (np.ndarray): Vetor de bits decodificado.

        """
        if encoded_stream.size % 2 != 0:
            raise ValueError("Tamanho do vetor codificado inválido. Deve ser múltiplo de 2.")

        n = encoded_stream.size // 2
        decoded = np.empty(n, dtype=int)

        if self.method == 0:  # NRZ
            for i in range(n):
                pair = encoded_stream[2*i:2*i + 2]
                if np.array_equal(pair, [0, 0]):
                    decoded[i] = 0
                else:
                    decoded[i] = 1


        elif self.method == 1:  # Manchester
            for i in range(n):
                pair = encoded_stream[2*i:2*i + 2]
                if np.array_equal(pair, [0, 1]):
                    decoded[i] = 0
                else:
                    decoded[i] = 1

        else:
            raise ValueError(f"Método de decodificação não implementado: {self.method}")

        return decoded

if __name__ == "__main__":

    Xn = np.random.randint(0, 2, 30)
    Yn = np.random.randint(0, 2, 30)
    print("Channel Xn: ", ''.join(str(int(b)) for b in Xn))
    print("Channel Yn: ", ''.join(str(int(b)) for b in Yn))

    # Inicializando o Encoder com o nome do método desejado ('NRZ' ou 'Manchester')
    encoder_nrz = Encoder(method="NRZ")
    encoder_man = Encoder(method="Manchester")

    Xnrz = encoder_nrz.encode(Xn)
    print("Channel X(NRZ)[n]:", ''.join(str(int(b)) for b in Xnrz))
    Yman = encoder_man.encode(Yn)
    print("Channel Y(MAN)[n]:", ''.join(str(int(b)) for b in Yman))

    fig_encoder, grid = create_figure(4, 1, figsize=(16, 9))

    BitsPlot(
        fig_encoder, grid, (0, 0),
        bits_list=[Xn],
        sections=[("$X_n$", len(Xn))],
        colors=["darkgreen"]
    ).plot(ylabel="$X_n$")

    EncodedBitsPlot(
        fig_encoder, grid, (1, 0),
        bits=Xnrz,
        color='darkgreen',
    ).plot(ylabel="$X_{NRZ}[n]$", label="$X_{NRZ}[n]$")

    BitsPlot(
        fig_encoder, grid, (2, 0),
        bits_list=[Yn],
        sections=[("$Y_n$", len(Yn))],
        colors=["navy"]
    ).plot(ylabel="$Y_n$")

    EncodedBitsPlot(
        fig_encoder, grid, (3, 0),
        bits=Yman,
        color="navy",
    ).plot(xlabel="Bit", ylabel="$Y_{MAN}[n]$", label="$Y_{MAN}[n]$")

    fig_encoder.tight_layout()
    save_figure(fig_encoder, "example_encoder.pdf")

    Xn_prime = encoder_nrz.decode(Xnrz)
    print("Channel X'n:", ''.join(str(int(b)) for b in Xn_prime))
    Yn_prime = encoder_man.decode(Yman)
    print("Channel Y'n:", ''.join(str(int(b)) for b in Yn_prime))

    print("Xn = Y'n: ", np.array_equal(Xn, Xn_prime))
    print("Yn = X'n: ", np.array_equal(Yn, Yn_prime))