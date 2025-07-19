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

class Preamble:
    def __init__(self, preamble_hex):
        self.preamble_hex = preamble_hex
        self.preamble_bits = self.hex_to_bits(self.preamble_hex)

    def hex_to_bits(self, hex_string):

        # Retorna a string de bits com 32 bits - 2 (30bits)
        return format(int(hex_string, 16), '032b')[2:] 

    
    def i_channel(self):
        # Get the even bits of the preamble
        return np.array([int(bit) for bit in self.preamble_bits[::2]])
    
    def q_channel(self):
        # Get the odd bits of the preamble
        return np.array([int(bit) for bit in self.preamble_bits[1::2]])

    def plot_preamble(self, save_path=None):

        # Superamostragem do sinal para o plot
        SI_up = np.repeat(self.i_channel(), 2)
        SQ_up = np.repeat(self.q_channel(), 2)
        x = np.arange(len(SI_up))
        bit_edges = np.arange(0, len(SI_up) + 1, 2)

        # Configuração do gráfico
        fig, axs = plt.subplots(2, 1, sharex=True)
        def setup_grid(ax):
            ax.set_xlim(0, len(SI_up))
            ax.set_ylim(-0.2, 1.4)
            ax.grid(False)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
        
        # Canal I
        axs[0].step(x, SI_up, where='post', label=r"Canal I $(S_I)$", color='navy', linewidth=3)
        for i, bit in enumerate(self.i_channel()):
            axs[0].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[0].set_ylabel(r"$S_I$")
        leg0 = axs[0].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        axs[0].set_yticks([0, 1])
        setup_grid(axs[0])

        # Canal Q
        axs[1].step(x, SQ_up, where='post', label=r"Canal Q $(S_Q)$", color='darkred', linewidth=3)
        for i, bit in enumerate(self.q_channel()):
            axs[1].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[1].set_ylabel(r"$S_Q$")
        leg1 = axs[1].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)
        axs[1].set_yticks([0, 1])
        setup_grid(axs[1])

        # Configuração do layout
        plt.xlabel('Bits')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Salvar ou mostrar o gráfico
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()


if __name__ == "__main__":

    preamble_hex = "2BEEEEBF"

    preamble = Preamble(preamble_hex=preamble_hex)
    S_i = preamble.i_channel()
    S_q = preamble.q_channel()

    print("Preamble (S_i):", ''.join(str(int(b)) for b in S_i))
    print("Preamble (S_q):", ''.join(str(int(b)) for b in S_q))

    output_path = os.path.join("out", "example_preamble.pdf")
    preamble.plot_preamble(save_path=output_path)