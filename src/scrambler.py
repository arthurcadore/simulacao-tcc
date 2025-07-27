import numpy as np
from plots import Plotter

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



if __name__ == "__main__":
    scrambler = Scrambler()

    vt0 = np.random.randint(0, 2, 30)
    vt1 = np.random.randint(0, 2, 30)
    Xn, Yn = scrambler.scramble(vt0, vt1)

    print("vt0: ", ''.join(str(b) for b in vt0))
    print("vt1: ", ''.join(str(b) for b in vt1))
    print("Xn  :", ''.join(str(int(b)) for b in Xn))
    print("Yn  :", ''.join(str(int(b)) for b in Yn))
    
    vt0_prime, vt1_prime = scrambler.descramble(Xn, Yn)

    print("\nVerificação:")
    print("vt0':", ''.join(str(int(b)) for b in vt0_prime))
    print("vt1':", ''.join(str(int(b)) for b in vt1_prime))
    print("vt0 = vt0': ", np.array_equal(vt0, vt0_prime))
    print("vt1 = vt1': ", np.array_equal(vt1, vt1_prime))

    
    plotter = Plotter()
    plotter.plot_scrambler(vt0, 
                           vt1, 
                           Xn, 
                           Yn, 
                           vt0_prime, 
                           vt1_prime,
                           "Canal I $v_t^{(0)}$", 
                           "Canal Q $v_t^{(1)}$", 
                           "Canal I $(X_n)$", 
                           "Canal Q $(Y_n)$", 
                           "Canal I $v_t^{(0)'} $", 
                           "Canal Q $v_t^{(1)'} $",
                           save_path="../out/example_scrambling.pdf"
    )
    