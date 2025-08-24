"""
Implementa um formatador de pulso para transmissão de sinais digitais. 

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from plotter import ImpulseResponsePlot, TimePlot, create_figure, save_figure

class Formatter:
    def __init__(self, alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC"):
        r"""
        Inicializa uma instância de formatador, utilizado preparar os símbolos para modulação.

        Args:
            alpha (float): Fator de roll-off do pulso RRC.
            fs (int): Frequência de amostragem.
            Rb (int): Taxa de bits.
            span (int): Duração do pulso em termos de períodos de bit.
            type (str): Tipo de pulso, atualmente apenas $RRC$ é suportado.

        Raises:
            ValueError: Se o tipo de pulso não for suportado.

        Exemplo: 
            ![pageplot](assets/example_formatter_time.svg)

        <div class="referencia">
        <b>Referência:</b><br>
        EEL7062 – Princípios de Sistemas de Comunicação, Richard Demo Souza (Pg. 55)
        </div>
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
        Gera o pulso Root Raised Cosine ($RRC$). O pulso $RRC$ no dominio do tempo é definido pela expressão abaixo.

        $$
            g(t) = \frac{(1 - \alpha) sinc((1- \alpha) t / T_b) + \alpha (4/\pi) \cos(\pi (1 + \alpha) t / T_b)}{1 - (4 \alpha t / T_b)^2}
        $$

        Sendo: 
            - $g(t)$: Pulso formatador $RRC$ no dominio do tempo.
            - $\alpha$: Fator de roll-off do pulso.
            - $T_b$: Período de bit.
            - $t$: Vetor de tempo.

        Returns:
           rc (np.ndarray): Pulso RRC.

        Exemplo: 
            - ![pageplot](assets/example_formatter_impulse.svg)
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

        Sendo: 
            - $d(t)$: Sinal formatado de saída.
            - $x$: Vetor de símbolos de entrada.
            - $g(t)$: Pulso formatador.
            - $n$: Indice de bit.
            - $T_b$: Período de bit.

        Args:
            symbols (np.ndarray): Vetor de símbolos a serem formatados.
        
        Returns:
            out_symbols (np.ndarray): Vetor formatado com o pulso aplicado.
        """

        # TODO: Alterar para o encoder.
        symbols = np.asarray(symbols, dtype=float)
        if symbols.min() >= 0.0 and symbols.max() <= 1.0:
            # mapa 0/1 -> -1/+1
            symbols = 2.0*symbols - 1.0
            
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

    # Plotando a resposta ao impulso
    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0, 0),
        formatter.t_rc, formatter.g,
        t_unit="ms",
        colors="darkorange",
    ).plot(label="$g(t)$", xlabel="Tempo (ms)", ylabel="Amplitude", xlim=(-15, 15))

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_formatter_impulse.pdf")

    # Plotando os sinais formatados
    
    fig_format, grid_format = create_figure(2, 2, figsize=(16, 9))

    ImpulseResponsePlot(
        fig_format, grid_format, (0, slice(0, 2)),
        formatter.t_rc, formatter.g,
        t_unit="ms",
        colors="darkorange",
    ).plot(label="$g(t)$", xlabel="Tempo (ms)", ylabel="Amplitude", xlim=(-15, 15))
    
    TimePlot(
        fig_format, grid_format, (1,0),
        t= np.arange(len(dI)) / formatter.fs,
        signals=[dI],
        labels=["$d_I(t)$"],
        title="Canal $I$",
        xlim=(0, 0.1),
        ylim=(-0.1, 0.1),
        colors="darkgreen",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    TimePlot(
        fig_format, grid_format, (1,1),
        t= np.arange(len(dQ)) / formatter.fs,
        signals=[dQ],
        labels=["$d_Q(t)$"],
        title="Canal $Q$",
        xlim=(0, 0.1),
        ylim=(-0.1, 0.1),
        colors="darkblue",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    fig_format.tight_layout()
    save_figure(fig_format, "example_formatter_time.pdf")