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
        assert len(X) % 3 == 0, "O comprimento dos vetores deve ser múltiplo de 3"

        X_scrambled = []
        Y_scrambled = []

        for i in range(0, len(X), 3):
            # bloco de 3 bits
            x1, x2, x3 = X[i:i+3]
            y1, y2, y3 = Y[i:i+3]

            # saída embaralhada
            X_scrambled += [y1, x2, y2]
            Y_scrambled += [x1, x3, y3]

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
            ax.legend(loc='upper right', fontsize=10)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

        # Top: Antes
        superplot(axs[0, 0], I_before, "I antes", "navy")
        superplot(axs[0, 1], Q_before, "Q antes", "darkred")

        # Bottom: Depois
        superplot(axs[1, 0], I_after, "I após", "navy")
        superplot(axs[1, 1], Q_after, "Q após", "darkred")

        axs[1, 0].set_xlabel("Bits")
        axs[1, 1].set_xlabel("Bits")
        axs[0, 0].set_ylabel("Original")
        axs[1, 0].set_ylabel("Embaralhado")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle("Sinais I e Q antes e após embaralhamento", fontsize=16)

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
        assert len(X_scrambled) % 3 == 0, "O comprimento dos vetores deve ser múltiplo de 3"

        X_original = []
        Y_original = []

        for i in range(0, len(X_scrambled), 3):
            # Pega blocos de 3
            x_ = X_scrambled[i:i+3]  # [y1, x2, y2]
            y_ = Y_scrambled[i:i+3]  # [x1, x3, y3]

            # Reconstrói os valores originais
            x1 = y_[0]
            x2 = x_[1]
            x3 = y_[1]

            y1 = x_[0]
            y2 = x_[2]
            y3 = y_[2]

            X_original += [x1, x2, x3]
            Y_original += [y1, y2, y3]

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
            ax.legend(loc='upper right', fontsize=10)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

        # Top: Antes
        superplot(axs[0, 0], I_before, "I antes", "navy")
        superplot(axs[0, 1], Q_before, "Q antes", "darkred")

        # Bottom: Depois
        superplot(axs[1, 0], I_after, "I após", "navy")
        superplot(axs[1, 1], Q_after, "Q após", "darkred")

        axs[1, 0].set_xlabel("Bits")
        axs[1, 1].set_xlabel("Bits")
        axs[0, 0].set_ylabel("Original")
        axs[1, 0].set_ylabel("Desembaralhado")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle("Sinais I e Q antes e após desembaralhamento", fontsize=16)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()
    

if __name__ == "__main__":
    scrambler = Scrambler()
    descrambler = Descrambler()
    
    X = np.random.randint(0, 2, 21)
    Y = np.random.randint(0, 2, 21)
    print("X original     :", ''.join(str(b) for b in X))
    print("Y original     :", ''.join(str(b) for b in Y))

    Ip, Qp = scrambler.scramble(X, Y)
    print("X embaralhado  :", ''.join(str(int(b)) for b in Ip))
    print("Y embaralhado  :", ''.join(str(int(b)) for b in Qp))
    
    Xr, Yr = descrambler.descramble(Ip, Qp)
    print("X recuperado   :", ''.join(str(int(b)) for b in Xr))
    print("Y recuperado   :", ''.join(str(int(b)) for b in Yr))

    print("\nVerificação:")
    print("X ok?", np.array_equal(X, Xr))
    print("Y ok?", np.array_equal(Y, Yr))

    output_scrambling = os.path.join("out", "example_scrambling.pdf")
    scrambler.plot_scrambler(Y, X, Ip, Qp, save_path=output_scrambling)

    output_descrambling = os.path.join("out", "example_descrambling.pdf")
    descrambler.plot_descrambler(Ip, Qp, X, Y, save_path=output_descrambling)
