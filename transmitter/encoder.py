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
        if self.method == "nrz":
            return np.repeat(self.bitstream, 2)
        elif self.method == "manchester":
            out = np.empty(self.bitstream.size * 2, dtype=int)
            out[::2] = 1 - self.bitstream
            out[1::2] = self.bitstream
            return out
        else:
            raise ValueError(f"Metodo de codificação não implementado: {self.method}")

    @staticmethod
    def plot_signals(Ie, Qe, Xnrz, Ym, save_path=None):
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

        # Canal I original
        axs[0].step(x, Ie_up, where='post', label=r"Canal I $(X_n)$", color='navy', linewidth=3)
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
        axs[1].step(x, Xnrz, where='post', label=r"Canal I ($X_{NRZ}[n]$)", color='navy', linewidth=3)
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

        # Canal Q original
        axs[2].step(x, Qe_up, where='post', label=r"Canal Q $(Y_n)$", color='darkred', linewidth=3)
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
        axs[3].step(x, Ym, where='post', label=r"Canal Q ($Y_{MAN}[n]$)", color='darkred', linewidth=3)
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


if __name__ == "__main__":
    Xn = np.random.randint(0, 2, 30)
    Yn = np.random.randint(0, 2, 30)
    print("Canal I (Xn):", Xn)
    print("Canal Q (Yn):", Yn)

    encoded_I = Encoder(Xn, "NRZ").encode()
    print("Canal I X(NRZ)[n]:", encoded_I)
    encoded_Q = Encoder(Yn, "Manchester").encode()
    print("Canal Q Y(MAN)[n]:", encoded_Q)

    output_path = os.path.join("out", "example_nrz_man.pdf")
    Encoder.plot_signals(Xn, Yn, encoded_I, encoded_Q, save_path=output_path)
