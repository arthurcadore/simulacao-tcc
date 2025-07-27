import numpy as np
from plots import Plotter

class Encoder:
    def __init__(self, bitstream, method):
        self.bitstream = np.array(bitstream)
        self.method = method.lower()

    def encode(self):
        out = np.empty(self.bitstream.size * 2, dtype=int)

        if self.method == "nrz":
            for i, bit in enumerate(self.bitstream):
                if bit == 0:
                    out[2*i] = 0
                    out[2*i + 1] = 0
                elif bit == 1:
                    out[2*i] = 1
                    out[2*i + 1] = 1

        elif self.method == "manchester":
            for i, bit in enumerate(self.bitstream):
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
        if encoded_stream.size % 2 != 0:
            raise ValueError("Tamanho do vetor codificado inválido. Deve ser múltiplo de 2.")

        n = encoded_stream.size // 2
        decoded = np.empty(n, dtype=int)

        if self.method == "nrz":
            for i in range(n):
                pair = encoded_stream[2*i:2*i + 2]
                if np.array_equal(pair, [0, 0]):
                    decoded[i] = 0
                elif np.array_equal(pair, [1, 1]):
                    decoded[i] = 1
                else:
                    raise ValueError(f"Padrão NRZ inválido no índice {i}: {pair}")

        elif self.method == "manchester":
            for i in range(n):
                pair = encoded_stream[2*i:2*i + 2]
                if np.array_equal(pair, [0, 1]):
                    decoded[i] = 0
                elif np.array_equal(pair, [1, 0]):
                    decoded[i] = 1
                else:
                    raise ValueError(f"Padrão Manchester inválido no índice {i}: {pair}")

        else:
            raise ValueError(f"Método de decodificação não implementado: {self.method}")

        return decoded

if __name__ == "__main__":
    Xn = np.random.randint(0, 2, 30)
    Yn = np.random.randint(0, 2, 30)
    print("Channel Xn: ", ''.join(str(int(b)) for b in Xn))
    print("Channel Yn: ", ''.join(str(int(b)) for b in Yn))

    Xnrz = Encoder(Xn, "NRZ").encode()
    print("Channel X(NRZ)[n]:", ''.join(str(int(b)) for b in Xnrz))
    Yman = Encoder(Yn, "Manchester").encode()
    print("Channel Y(MAN)[n]:", ''.join(str(int(b)) for b in Yman))

    plot = Plotter()
    plot.plot_encode(s1=Xn, 
                     s2=Yn, 
                     s3=Xnrz, 
                     s4=Yman, 
                     label1="Canal I $(X_n)$", 
                     label2="Canal Q $(Y_n)$", 
                     label3="Canal I $(X_{NRZ}[n])$", 
                     label4="Canal Q $(Y_{MAN}[n])$", 
                     title1="$X_n$", 
                     title2="$Y_n$", 
                     title3="$X_{NRZ}[n]$", 
                     title4="$Y_{MAN}[n]$", 
                     save_path="../out/example_encoder.pdf")

    Xn_prime = Encoder(Xnrz, "NRZ").decode(Xnrz)
    print("Channel X'n:", ''.join(str(int(b)) for b in Xn_prime))
    Yn_prime = Encoder(Yman, "Manchester").decode(Yman)
    print("Channel Y'n:", ''.join(str(int(b)) for b in Yn_prime))

    print("Xn = Y'n: ", np.array_equal(Xn, Xn_prime))
    print("Yn = X'n: ", np.array_equal(Yn, Yn_prime))
    