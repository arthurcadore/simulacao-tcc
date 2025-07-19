import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

# Estilo science
plt.style.use('science')
plt.rcParams["figure.figsize"] = (16, 9)
plt.rc('font', size=16)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=22)
plt.rc('legend', frameon=True, edgecolor='black', facecolor='white', fancybox=True, fontsize=12)


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

    @staticmethod
    def plot_encode(Ie, Qe, Xnrz, Ym, save_path=None):
        Ie_up = np.repeat(Ie, 2)
        Qe_up = np.repeat(Qe, 2)
        x = np.arange(len(Ie_up))
        bit_edges = np.arange(0, len(Ie_up) + 1, 2)

        fig, axs = plt.subplots(4, 1, sharex=True)

        def setup_grid(ax):
            ax.set_xlim(0, len(Ie_up))
            ax.set_ylim(-0.2, 1.4)
            ax.grid(False)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        # Channel I original
        axs[0].step(x, Ie_up, where='post', label=r"Channel I $(X_n)$", color='navy', linewidth=3)
        for i, bit in enumerate(Ie):
            axs[0].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[0].set_ylabel(r"$X_n$")
        leg0 = axs[0].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        setup_grid(axs[0])

        # NRZ
        axs[1].step(x, Xnrz, where='post', label=r"Channel I ($X_{NRZ}[n]$)", color='navy', linewidth=3)
        for i in range(len(Ie)):
            pair = ''.join(str(b) for b in Xnrz[2 * i:2 * i + 2])
            axs[1].text(i * 2 + 1, 1.15, pair, ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[1].set_ylabel(r"$X_{NRZ}[n]$")
        leg1 = axs[1].legend( 
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)
        setup_grid(axs[1])

        # Channel Q original
        axs[2].step(x, Qe_up, where='post', label=r"Channel Q $(Y_n)$", color='darkred', linewidth=3)
        for i, bit in enumerate(Qe):
            axs[2].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[2].set_ylabel(r"$Y_n$")
        leg2 = axs[2].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg2.get_frame().set_facecolor('white')
        leg2.get_frame().set_edgecolor('black')
        leg2.get_frame().set_alpha(1.0)
        setup_grid(axs[2])

        # Manchester
        axs[3].step(x, Ym, where='post', label=r"Channel Q ($Y_{MAN}[n]$)", color='darkred', linewidth=3)
        for i in range(len(Qe)):
            pair = ''.join(str(b) for b in Ym[2 * i:2 * i + 2])
            axs[3].text(i * 2 + 1, 1.15, pair, ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[3].set_ylabel(r"$Y_{MAN}[n]$")
        leg3 = axs[3].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg3.get_frame().set_facecolor('white')
        leg3.get_frame().set_edgecolor('black')
        leg3.get_frame().set_alpha(1.0)
        setup_grid(axs[3])

        plt.xlabel('Bits')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

class Decoder:
    def __init__(self, bitstream, method):
        self.bitstream = np.array(bitstream)
        self.method = method.lower()

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
    print("Channel I (Xn):", ''.join(str(int(b)) for b in Xn))
    print("Channel Q (Yn):", ''.join(str(int(b)) for b in Yn))

    Xnrz = Encoder(Xn, "NRZ").encode()
    print("Channel I X(NRZ)[n]:", ''.join(str(int(b)) for b in Xnrz))
    Yman = Encoder(Yn, "Manchester").encode()
    print("Channel Q Y(MAN)[n]:", ''.join(str(int(b)) for b in Yman))

    output_path = os.path.join("out", "example_nrz_man.pdf")
    Encoder.plot_encode(Xn, Yn, Xnrz, Yman, save_path=output_path)

    Xn_prime = Decoder(Xn, "NRZ").decode(Xnrz)
    print("Channel I (X'n):", ''.join(str(int(b)) for b in Xn_prime))
    Yn_prime = Decoder(Yn, "Manchester").decode(Yman)
    print("Channel Q (Y'n):", ''.join(str(int(b)) for b in Yn_prime))

    print("Xn = Y'n: ", np.array_equal(Xn, Xn_prime))
    print("Yn = X'n: ", np.array_equal(Yn, Yn_prime))