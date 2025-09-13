"""
Implementa uma classe de filtro casado para maximizar a SNR do sinal recebido.

Autor: Arthur Cadore
Data: 15-08-2025
"""

import numpy as np
from .plotter import create_figure, save_figure, ImpulseResponsePlot
from .formatter import Formatter

class MatchedFilter:
    def __init__(self, alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC-Inverted", channel=None, bits_per_symbol=1):
        r"""
        Inicializa um filtro casado. O filtro casado é usado para maximizar a SNR do sinal recebido.

        Args:
            alpha (float): Fator de roll-off do filtro casado.
            fs (int): Frequência de amostragem.
            Rb (int): Taxa de bits.
            span (int): Duração do pulso em termos de períodos de bit.
            type (str): Tipo de filtro, atualmente apenas "RRC-Inverted" e "Manchester-Inverted" são suportados.

        Raises:     
            ValueError: Se o tipo de filtro não for suportado.

        Exemplo: 
            ![pageplot](assets/receiver_mf_time.svg) 
        """
        self.alpha = alpha
        self.fs = fs
        self.Rb = Rb
        self.Tb = 1 / Rb
        self.sps = int(fs / Rb)
        self.span = span
        self.channel = channel
        self.bits_per_symbol=bits_per_symbol
        type_map = {
            "rrc-inverted": 0,
            "manchester-inverted": 1
        }

        type = type.lower()
        if type not in type_map:
            raise ValueError("Tipo de filtro inválido. Use 'RRC-inverted' ou 'Manchester-inverted'.")
        
        self.type = type_map[type]

        if self.type == 0:  # RRC
            self.formatter = Formatter(alpha=self.alpha, fs=self.fs, Rb=self.Rb, span=self.span, type="RRC", channel=self.channel, bits_per_symbol=self.bits_per_symbol)
        elif self.type == 1:  # Manchester
            self.formatter = Formatter(alpha=self.alpha, fs=self.fs, Rb=self.Rb, span=self.span, type="Manchester", channel=self.channel, bits_per_symbol=self.bits_per_symbol)
        
        self.g = self.formatter.g
        self.t_rc = self.formatter.t_rc

        # Inverte o pulso
        self.g_inverted = self.inverted_pulse(self.g)

    def inverted_pulse(self, pulse):
        return pulse[::-1]


    def apply_filter(self, signal):
        r"""
        Aplica o filtro casado com resposta ao impulso $g(-t)$ ao sinal de entrada $s(t)$. O processo de filtragem é dado pela expressão abaixo. 

        $$
            x(t) = s(t) \ast g(-t)
        $$

        Sendo: 
            - $x(t)$: Sinal filtrado.
            - $s(t)$: Sinal de entrada.
            - $g(-t)$: Pulso formatador $RRC$ invertido.

        Args:
            signal (np.ndarray): Sinal de entrada $s(t)$.

        Returns:
            signal_filtered (np.ndarray): Sinal filtrado $x(t)$.
        """
        signal_filtered = np.convolve(signal, self.g_inverted, mode='same')

        # normalização
        signal_filtered = signal_filtered / np.max(np.abs(signal_filtered))

        return signal_filtered


if __name__ == "__main__":
    filtro = MatchedFilter(alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC-Inverted", channel="I")

    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0,0),
        filtro.t_rc, [filtro.g, filtro.g_inverted],
        t_unit="ms",
        colors=["darkorange", "steelblue"],
    ).plot(
        label=[r"$g(t)$", r"$g(-t)$"],
        xlabel=r"Tempo ($ms$)",
        ylabel="Amplitude",
        xlim=(-15, 15)
    )

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_mf_impulse.pdf")
    
    filtro = MatchedFilter(alpha=0.8, fs=128_000, Rb=400, span=6, type="Manchester-Inverted", channel="Q")

    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0,0),
        filtro.t_rc, [filtro.g, filtro.g_inverted],
        t_unit="ms",
        colors=["darkorange", "steelblue"],
    ).plot(
        label=[r"$g(t)$", r"$g(-t)$"],
        xlabel=r"Tempo ($ms$)",
        ylabel="Amplitude",
        xlim=(-15, 15)
    )

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_mf_impulse_man.pdf")