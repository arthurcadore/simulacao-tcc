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

class Multiplexer:
    def __init__(self):
        pass

    def concatenate(self, I1, Q1, I2, Q2):
        I = np.concatenate((I1, I2))
        Q = np.concatenate((Q1, Q2))
        return I, Q

    @staticmethod
    def plot_concatenation(I1, Q1, I2, Q2, save_path=None):
        # Concatenação
        SI = np.concatenate((I1, I2))
        SQ = np.concatenate((Q1, Q2))

        # Superamostragem
        SI_up = np.repeat(SI, 2)
        SQ_up = np.repeat(SQ, 2)
        x = np.arange(len(SI_up))
        bit_edges = np.arange(0, len(SI_up) + 1, 2)

        fig, axs = plt.subplots(2, 1, sharex=True)

        def setup_grid(ax):
            ax.set_xlim(0, len(SI_up))
            ax.set_ylim(-0.2, 1.4)
            ax.grid(False)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        # --- Canal I ---
        sep_I = len(I1) * 2
        x_I1 = x[:sep_I]
        x_I2 = x[sep_I - 1:]  # Inclui o último ponto anterior

        y_I1 = SI_up[:sep_I]
        y_I2 = SI_up[sep_I - 1:]

        axs[0].step(x_I1, y_I1, where='post', color='navy', linewidth=3, label=r'Preambulo $S_I$')
        axs[0].step(x_I2, y_I2, where='post', color='darkred', linewidth=3, label=r'Canal $X_n$')

        for i, bit in enumerate(SI):
            axs[0].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')

        axs[0].set_ylabel(r"$X_n$")
        leg0 = axs[0].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        axs[0].set_yticks([0, 1])
        axs[0].set_yticks([0, 1])
        setup_grid(axs[0])

        # --- Canal Q ---
        sep_Q = len(Q1) * 2
        x_Q1 = x[:sep_Q]
        x_Q2 = x[sep_Q - 1:]

        y_Q1 = SQ_up[:sep_Q]
        y_Q2 = SQ_up[sep_Q - 1:]

        axs[1].step(x_Q1, y_Q1, where='post', color='navy', linewidth=3, label=r'Preambulo $S_Q$')
        axs[1].step(x_Q2, y_Q2, where='post', color='darkred', linewidth=3, label=r'Canal $Y_n$')

        for i, bit in enumerate(SQ):
            axs[1].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')

        axs[1].set_ylabel(r"$Y_n$")
        leg1 = axs[1].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)
        axs[1].set_yticks([0, 1])
        setup_grid(axs[1])

        # Layout final
        plt.xlabel('Bits')

        num_bits = len(SI)
        step = 5
        axs[1].set_xticks(np.arange(0, num_bits * 2, step * 2))      
        axs[1].set_xticklabels(np.arange(0, num_bits, step)) 

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()




# Exemplo de uso
if __name__ == "__main__":

    mux = Multiplexer()

    SI = np.random.randint(0, 2, 10)
    print("Preamble SI:", SI)
    SQ = np.random.randint(0, 2, 10)
    print("Preamble SQ:", SQ)
    Xn = np.random.randint(0, 2, 40)
    print("Canal I (Xn):", Xn)
    Yn = np.random.randint(0, 2, 40)
    print("Canal Q (Yn):", Yn)

    Xn, Yn = mux.concatenate(SI, SQ, Xn, Yn)

    output_path = os.path.join("out", "example_multiplexing.pdf")
    mux.plot_concatenation(SI, SQ, Xn, Yn, save_path=output_path)

    print("Canal I (Xn):", Xn)
    print("Canal Q (Yn):", Yn)
