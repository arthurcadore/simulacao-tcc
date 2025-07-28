import numpy as np
from plots import Plotter

class Formatter:
    def __init__(self, alpha=0.8, fs=128_000, Rb=400, span=6):
        self.alpha = alpha
        self.fs = fs
        self.Rb = Rb
        self.Tb = 1 / Rb
        self.sps = int(fs / Rb)
        self.span = span
        self.t_rc = np.linspace(-span * self.Tb, span * self.Tb, span * self.sps * 2)
        self.g = self.rrc_pulse(self.t_rc, self.Tb, self.alpha)

    def rrc_pulse(self, t, Tb, alpha):
        t = np.array(t, dtype=float) 
        rc = np.zeros_like(t)
        for i, ti in enumerate(t):
            if np.isclose(ti, 0.0):
                rc[i] = 1.0 + alpha * (4/np.pi - 1)
            elif alpha != 0 and np.isclose(np.abs(ti), Tb/(4*alpha)):
                rc[i] = (alpha/np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                    (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
                )
            else:
                num = np.sin(np.pi * ti * (1 - alpha) / Tb) + \
                      4 * alpha * (ti / Tb) * np.cos(np.pi * ti * (1 + alpha) / Tb)
                den = np.pi * ti * (1 - (4 * alpha * ti / Tb) ** 2) / Tb
                rc[i] = num / den
        # Normaliza energia para 1
        rc = rc / np.sqrt(np.sum(rc**2))
        return rc

    def format(self, symbols, pulse=None, sps=None):
        if pulse is None:
            pulse = self.g
        if sps is None:
            sps = self.sps
        upsampled = np.zeros(len(symbols) * sps)
        upsampled[::sps] = symbols
        return np.convolve(upsampled, pulse, mode='same')

if __name__ == "__main__":

    Xnrz = np.random.randint(0, 2, 50)
    Yman = np.random.randint(0, 2, 50)

    formatter = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6)
    dI = formatter.format(Xnrz)
    dQ = formatter.format(Yman)
    
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
    