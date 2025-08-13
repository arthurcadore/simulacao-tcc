"""
Implementação de um modulador IQ para transmissão de sinais digitais.

Autor: Arthur Cadore
Data: 28-07-2025
"""
import numpy as np
from formatter import Formatter
from plots import Plotter

class Modulator:
    r"""
    Inicializa uma instância do modulador IQ.
    O modulador IQ é responsável por modular os sinais I e Q em uma portadora de frequência específica.

    Args:
        fc (float): Frequência da portadora.
        fs (int): Frequência de amostragem.

    Raises:
        ValueError: Se a frequência de amostragem não for maior que o dobro da frequência da portadora. (Teorema de Nyquist)
    """
    def __init__(self, fc, fs):
        if fc <= 0:
            raise ValueError("A frequência da portadora deve ser maior que zero.")
        
        if fs <= fc*2:
            raise ValueError("A frequência de amostragem deve ser maior que o dobro da frequência da portadora.")
        
        self.fc = fc
        self.fs = fs

    def modulate(self, i_signal, q_signal):
        r"""
        Modula os sinais I e Q em uma portadora de frequência específica. O processo de modulação é dado pela expressão:

        $$
            s(t) = I(t) \cdot \cos(2\pi f_c t) - Q(t) \cdot \sin(2\pi f_c t)
        $$

        Args:
            i_signal (np.ndarray): Sinal I a ser modulado.
            q_signal (np.ndarray): Sinal Q a ser modulado.

        Returns:
            t (np.ndarray): Vetor de tempo $t$ correspondente ao sinal modulado.
            modulated_signal (np.ndarray): Sinal modulado $s(t)$ resultante.

        Raises:
            ValueError: Se os sinais I e Q não tiverem o mesmo tamanho.
        """
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
    

    
    