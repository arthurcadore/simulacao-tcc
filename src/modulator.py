import numpy as np
from formatter import Formatter
from plots import Plotter

class Modulator:
    def __init__(self, fc, fs):
        self.fc = fc
        self.fs = fs

    def modulate(self, i_signal, q_signal):
        n = len(i_signal)
        if len(q_signal) != n:
            raise ValueError("i_signal e q_signal devem ter o mesmo tamanho.")
        
        t = np.arange(n) / self.fs
        carrier_cos = np.cos(2 * np.pi * self.fc * t)
        carrier_sin = np.sin(2 * np.pi * self.fc * t)
        
        modulated_signal = i_signal * carrier_cos - q_signal * carrier_sin
        return t, modulated_signal

if __name__ == "__main__":

    fs = 128_000
    fc = 4000
    Rb = 400
    alpha = 0.8
    span = 8

    Xnrz = np.random.randint(0, 2, 900)
    Yman = np.random.randint(0, 2, 900)
    print("Xnrz:", ''.join(str(b) for b in Xnrz[:20]))
    print("Yman:", ''.join(str(b) for b in Yman[:20]))

    formatter = Formatter(alpha=alpha, fs=fs, Rb=Rb, span=span)
    dI = formatter.apply_format(Xnrz)
    dQ = formatter.apply_format(Yman)
    print("dI:", ''.join(str(b) for b in dI[:5]))
    print("dQ:", ''.join(str(b) for b in dQ[:5]))
    
    modulator = Modulator(fc=fc, fs=fs)
    t, s = modulator.modulate(dI, dQ)
    print("s:", ''.join(str(b) for b in s[:5]))
    
    plot = Plotter()
    plot.plot_modulation_time(dI, 
                              dQ, 
                              s, 
                              "dI(t)", 
                              "dQ(t)",
                              "s(t)",
                              "Sinal $IQ$ - Formatados RRC",
                              "Sinal Modulado $IQ$",
                              fs=fs, 
                              t_xlim=0.10, 
                              save_path="../out/example_modulator.pdf"
    )

    plot.plot_modulation_freq(dI, 
                              dQ, 
                              s,
                              "$D_I'(f)$",
                              "$D_Q'(f)$",
                              "$S(f)$",
                              "Sinal Banda Base - Componente $I$",
                              "Sinal Banda Base - Componente $Q$",
                              "Sinal Modulado $IQ$",
                              fs=fs, 
                              fc=fc, 
                              save_path="../out/example_frequency.pdf"
    )
    
    plot.plot_modulation_eye(dI, 
                             dQ, 
                             "dI(t)", 
                             "dQ(t)", 
                             "Sinal Modulado $IQ$", 
                             "Sinal Modulado $IQ$", 
                             fs=fs, 
                             Rb=Rb, 
                             save_path="../out/example_eye.pdf"
    )

    plot.plot_modulation_iq(dI,
                            dQ,
                            fr'Amostras $IQ$',
                            fr'Simbolos $QPSK$',
                            fr'Plano $IQ$ (Scatter)',
                            fr'Plano $IQ$ (Constelação)',
                            save_path="../out/example_constellation.pdf"
    )
    

    
    