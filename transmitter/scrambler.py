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

class Scrambler:
    def __init__(self):
        self.polynomial = [1, 0, 0, 0, 1, 0, 0, 1]

    def scramble(self, X, Y):
        assert len(X) == len(Y), "Vetores X e Y devem ter o mesmo comprimento"
        X_scrambled = []
        Y_scrambled = []

        for i in range(0, len(X), 3):
            # bloco de até 3 bits
            x_blk = X[i:i+3]
            y_blk = Y[i:i+3]
            n = len(x_blk)

            if n == 3:
                x1, x2, x3 = x_blk
                y1, y2, y3 = y_blk
                X_scrambled += [y1, x2, y2]
                Y_scrambled += [x1, x3, y3]
            elif n == 2:
                x1, x2 = x_blk
                y1, y2 = y_blk
                X_scrambled += [y1, x2]
                Y_scrambled += [x1, y2]
            elif n == 1:
                x1 = x_blk[0]
                y1 = y_blk[0]
                X_scrambled += [y1]
                Y_scrambled += [x1]

        return X_scrambled, Y_scrambled

    @staticmethod
    def plot_scrambler(I_before, Q_before, I_after, Q_after, save_path=None):
        '''Plota os sinais I e Q antes e depois do embaralhamento'''

        def superplot(ax, signal, label, color):
            sig_up = np.repeat(signal, 2)
            x = np.arange(len(sig_up))
            bit_edges = np.arange(0, len(sig_up) + 1, 2)

            ax.step(x, sig_up, where='post', label=label, color=color, linewidth=2)
            ax.set_xlim(0, len(sig_up))
            ax.set_ylim(-0.2, 1.4)
            ax.set_yticks([0, 1])
            ax.grid(False)
            for i, bit in enumerate(signal):
                ax.text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=12)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
            leg = ax.legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_alpha(1.0)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

        # Top: Antes
        superplot(axs[0, 0], I_before, r"$vt^{(0)}$", "navy")
        superplot(axs[0, 1], Q_before, r"$vt^{(1)}$", "darkred")

        # Bottom: Depois
        superplot(axs[1, 0], I_after, r"$X_n$", "navy")
        superplot(axs[1, 1], Q_after, r"$Y_n$", "darkred")

        axs[1, 0].set_xlabel("Bits")
        axs[1, 1].set_xlabel("Bits")
        axs[0, 0].set_ylabel("Original")
        axs[1, 0].set_ylabel("Embaralhado")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

class Descrambler:
    def __init__(self):
        self.polynomial = [1, 0, 0, 0, 1, 0, 0, 1]

    def descramble(self, X_scrambled, Y_scrambled):
        assert len(X_scrambled) == len(Y_scrambled), "Vetores devem ter o mesmo comprimento"
        X_original = []
        Y_original = []

        for i in range(0, len(X_scrambled), 3):
            x_ = X_scrambled[i:i+3]
            y_ = Y_scrambled[i:i+3]
            n = len(x_)

            if n == 3:
                # [y1, x2, y2], [x1, x3, y3]
                x1 = y_[0]
                x2 = x_[1]
                x3 = y_[1]
                y1 = x_[0]
                y2 = x_[2]
                y3 = y_[2]
                X_original += [x1, x2, x3]
                Y_original += [y1, y2, y3]
            elif n == 2:
                # [y1, x2], [x1, y2]
                x1 = y_[0]
                x2 = x_[1]
                y1 = x_[0]
                y2 = y_[1]
                X_original += [x1, x2]
                Y_original += [y1, y2]
            elif n == 1:
                x1 = y_[0]
                y1 = x_[0]
                X_original += [x1]
                Y_original += [y1]

        return X_original, Y_original
    
    @staticmethod
    def plot_descrambler(I_before, Q_before, I_after, Q_after, save_path=None):
        '''Plota os sinais I e Q antes e depois do embaralhamento'''

        def superplot(ax, signal, label, color):
            sig_up = np.repeat(signal, 2)
            x = np.arange(len(sig_up))
            bit_edges = np.arange(0, len(sig_up) + 1, 2)

            ax.step(x, sig_up, where='post', label=label, color=color, linewidth=2)
            ax.set_xlim(0, len(sig_up))
            ax.set_ylim(-0.2, 1.4)
            ax.set_yticks([0, 1])
            ax.grid(False)
            for i, bit in enumerate(signal):
                ax.text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=12)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
            leg = ax.legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_alpha(1.0)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

        # Top: Antes
        superplot(axs[0, 0], I_before, r"$X_n$", "navy")
        superplot(axs[0, 1], Q_before, r"$Y'_n$", "darkred")

        # Bottom: Depois
        superplot(axs[1, 0], I_after, r"$vt^{(0)}$", "navy")
        superplot(axs[1, 1], Q_after, r"$vt^{(1)}$", "darkred")

        axs[1, 0].set_xlabel("Bits")
        axs[1, 1].set_xlabel("Bits")
        axs[0, 0].set_ylabel("Original")
        axs[1, 0].set_ylabel("Desembaralhado")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()
    

if __name__ == "__main__":
    scrambler = Scrambler()
    descrambler = Descrambler()
    
    vt0 = np.random.randint(0, 2, 30)
    vt1 = np.random.randint(0, 2, 30)
    print("vt0: ", ''.join(str(b) for b in vt0))
    print("vt1: ", ''.join(str(b) for b in vt1))

    Xn, Yn = scrambler.scramble(vt0, vt1)
    print("Xn  :", ''.join(str(int(b)) for b in Xn))
    print("Yn  :", ''.join(str(int(b)) for b in Yn))
    
    vt0_prime, vt1_prime = descrambler.descramble(Xn, Yn)
    print("vt0':", ''.join(str(int(b)) for b in vt0_prime))
    print("vt1':", ''.join(str(int(b)) for b in vt1_prime))

    print("\nVerificação:")
    print("vt0 = vt0': ", np.array_equal(vt0, vt0_prime))
    print("vt1 = vt1': ", np.array_equal(vt1, vt1_prime))

    output_scrambling = os.path.join("out", "example_scrambling.pdf")
    scrambler.plot_scrambler(vt0, vt1, Xn, Yn, save_path=output_scrambling)

    output_descrambling = os.path.join("out", "example_descrambling.pdf")
    descrambler.plot_descrambler(Xn, Yn, vt0, vt1, save_path=output_descrambling)
