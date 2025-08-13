"""
Implementa um formatador de pulso para transmissão de sinais digitais. 

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from plots import Plotter

class Formatter:
    def __init__(self, alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC"):
        r"""
        Inicializa uma instância de formatador de pulso. O pulso formatador é usado para preparar os símbolos nos canais I e Q para transmissão.

        Args:
            alpha (float): Fator de roll-off do pulso RRC.
            fs (int): Frequência de amostragem.
            Rb (int): Taxa de bits.
            span (int): Duração do pulso em termos de períodos de bit.
            type (str): Tipo de pulso, atualmente apenas "RRC" é suportado.

        Raises:
            ValueError: Se o tipo de pulso não for suportado.
        """
        self.alpha = alpha
        self.fs = fs
        self.Rb = Rb
        self.Tb = 1 / Rb
        self.sps = int(fs / Rb)
        self.span = span
        self.t_rc = np.linspace(-span * self.Tb, span * self.Tb, span * self.sps * 2)

        type_map = {
            "rrc": 0
        }

        type = type.lower()
        if type not in type_map:
            raise ValueError("Tipo de pulso inválido. Use 'RRC'.")
        
        self.type = type_map[type]

        if self.type == 0:  # RRC
            self.g = self.rrc_pulse()

    def rrc_pulse(self):
        r"""
        Gera o pulso Root Raised Cosine (RRC) para a transmissão de sinais digitais. O pulso RRC é definido como:
        $$
        \begin{equation}
            g(t) = \frac{\sin(\pi \frac{t}{T_b})}{\pi \frac{t}{T_b}} \cdot \frac{\cos(\pi \alpha \frac{t}{T_b})}{1 - (2\alpha \frac{t}{T_b})^2}
        \end{equation}
        $$

        Nota:
            - $g(t)$ é o pulso formatador,
            - $\alpha$ é o fator de roll-off, 
            - $T_b$ é o período de bit, 
            - $t$ é o tempo.

        Args:
            None

        Returns:
           rc (np.ndarray): Pulso RRC.
        """
        self.t_rc = np.array(self.t_rc, dtype=float) 
        rc = np.zeros_like(self.t_rc)
        for i, ti in enumerate(self.t_rc):
            if np.isclose(ti, 0.0):
                rc[i] = 1.0 + self.alpha * (4/np.pi - 1)
            elif self.alpha != 0 and np.isclose(np.abs(ti), self.Tb/(4*self.alpha)):
                rc[i] = (self.alpha/np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi/(4*self.alpha)) +
                    (1 - 2/np.pi) * np.cos(np.pi/(4*self.alpha))
                )
            else:
                num = np.sin(np.pi * ti * (1 - self.alpha) / self.Tb) + \
                      4 * self.alpha * (ti / self.Tb) * np.cos(np.pi * ti * (1 + self.alpha) / self.Tb)
                den = np.pi * ti * (1 - (4 * self.alpha * ti / self.Tb) ** 2) / self.Tb
                rc[i] = num / den
        # Normaliza energia para 1
        rc = rc / np.sqrt(np.sum(rc**2))
        return rc

    def apply_format(self, symbols):
        r"""
        Formata os símbolos de entrada usando o pulso inicializado. O processo de formatação é dado por: 

        $$
           d(t) = \sum_{n} x[n] \cdot g(t - nT_b)
        $$

        Nota: 
            - $d(t)$ é o sinal formatado de saída,
            - $x$ é o vetor de símbolos de entrada,
            - $g(t)$ é o pulso formatador,
            - $n$ é o índice de tempo,
            - $T_b$ é o período de bit.

        Args:
            symbols (np.ndarray): Vetor de símbolos a serem formatados.
        
        Returns:
            out_symbols (np.ndarray): Vetor formatado com o pulso aplicado.
        """
        pulse = self.g
        sps = self.sps
        upsampled = np.zeros(len(symbols) * sps)
        upsampled[::sps] = symbols
        return np.convolve(upsampled, pulse, mode='same')

if __name__ == "__main__":

    Xnrz = np.random.randint(0, 2, 50)
    Yman = np.random.randint(0, 2, 50)

    formatter = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC")
    dI = formatter.apply_format(Xnrz)
    dQ = formatter.apply_format(Yman)
    
    print("Xnrz:", ''.join(str(b) for b in Xnrz))
    print("Yman:", ''.join(str(b) for b in Yman))
    print("dI:", ''.join(str(b) for b in dI[:5]))
    print("dQ:", ''.join(str(b) for b in dQ[:5]))
    
    plot = Plotter()
    plot.plot_filter(formatter.g, 
                     formatter.t_rc, 
                     formatter.Tb, 
                     formatter.span, 
                     formatter.fs, 
                     dI, 
                     dQ,
                     fr'Pulso RRC ($\alpha={formatter.alpha}$)', 
                     fr'$d_I(t)$', 
                     fr'$d_Q(t)$', 
                     'Pulso Root Raised Cosine (RRC)', 
                     fr'Sinal $d_I(t)$', 
                     fr'Sinal $d_Q(t)$', 
                     0.05,
                     save_path="../out/example_formatter.pdf"
    )
    