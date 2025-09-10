"""
Implementa uma classe de filtro casado para maximizar a SNR do sinal recebido.

Autor: Arthur Cadore
Data: 15-08-2025
"""

import numpy as np
from .plotter import create_figure, save_figure, ImpulseResponsePlot

class MatchedFilter:
    def __init__(self, alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC-Inverted"):
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
        self.t_rc = np.linspace(-span * self.Tb, span * self.Tb, span * self.sps * 2)

        type_map = {
            "rrc-inverted": 0,
            "manchester-inverted": 1
        }

        type = type.lower()
        if type not in type_map:
            raise ValueError("Tipo de filtro inválido. Use 'RRC-inverted' ou 'Manchester-inverted'.")
        
        self.type = type_map[type]

        if self.type == 0:  # RRC
            self.g = self.rrc_inverted_pulse()
        elif self.type == 1:  # Manchester
            self.g = self.manchester_inverted_pulse()
        
        # Calculate impulse response
        self.impulse_response, self.t_impulse = self.calc_impulse_response()

    def rrc_inverted_pulse(self):
        r"""
        Gera o pulso Root Raised Cosine ($RRC$) invertido $g(-t)$ para filtragem casada do sinal de entrada.
        
        $$
        \begin{equation}
            g(-t) = \frac{(1 - \alpha) sinc((1- \alpha) (-t) / T_b) + \alpha (4/\pi) \cos(\pi (1 + \alpha) (-t) / T_b)}{1 - (4 \alpha (-t) / T_b)^2}
        \end{equation}
        $$

        Sendo: 
            - $g(-t)$: Pulso formatador $RRC$ invertido no dominio do tempo.
            - $\alpha$: Fator de roll-off do pulso.
            - $T_b$: Período de bit.
            - $(-t)$: Vetor de tempo invertido.

        Returns:
           rc (np.ndarray): Pulso RRC invertido $g(-t)$.

        Exemplo: 
            ![pageplot](assets/example_mf_impulse.svg)
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
        
        # Inverte o pulso no tempo para criar o filtro casado
        rc = rc[::-1]
        return rc

    def manchester_inverted_pulse(self):
        """
        Gera o pulso Manchester invertido como um filtro casado.

        g(-t) = g_RRC((-t) + T_b/4) - g_RRC((-t) - T_b/4)

        Exemplo: 
            ![pageplot](assets/example_mf_impulse_manchester.svg)
        """
        # Base temporal e parâmetros
        t = np.asarray(self.t_rc, dtype=float)
        Tb = self.Tb
        sps = int(self.sps)
        if sps < 2:
            sps = 2
            self.sps = sps

        # Gera o RRC centrado em 0
        g_rrc = self.rrc_inverted_pulse()  # Pode usar o mesmo pulso RRC invertido

        # Função utilitária: desloca em amostras com zeros (sem "wrap")
        def shift_with_zeros(x, k):
            y = np.zeros_like(x)
            N = len(x)
            if k > 0:           # desloca para a direita
                y[k:] = x[:N-k]
            elif k < 0:         # desloca para a esquerda
                k = -k
                y[:N-k] = x[k:]
            else:
                y[:] = x
            return y

        # Deslocamento de +/- T_b/4 em amostras (≈ sps/4)
        q = max(1, int(round(sps/4)))   # garante ao menos 1 amostra de deslocamento

        g_left  = shift_with_zeros(g_rrc, +q)   # centro em +Tb/4
        g_right = shift_with_zeros(g_rrc, -q)  # centro em -Tb/4
        g = g_left - g_right


        return g

    def calc_impulse_response(self, impulse_len=512):
        r"""
        Calcula a resposta ao impulso do filtro casado.

        Args:
            impulse_len (int): Comprimento do vetor de impulso.

        Returns:
            impulse_response (tuple[np.ndarray, np.ndarray]): Resposta ao impulso e vetor de tempo.
        """
        # Usa o comprimento do pulso RRC como base para garantir que capturamos toda a resposta
        pulse_len = len(self.g)
        
        # Cria um impulso unitário no meio do vetor
        impulse = np.zeros(pulse_len)
        impulse[pulse_len // 2] = 1
        
        # Aplica o filtro casado ao impulso
        impulse_response = np.convolve(impulse, self.g, mode='full')
        
        # Cria o vetor de tempo centrado em zero
        t_impulse = (np.arange(len(impulse_response)) - len(impulse_response) // 2) / self.fs
        
        return impulse_response, t_impulse
        
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
        signal_filtered = np.convolve(signal, self.impulse_response, mode='same')

        # normalização
        signal_filtered = signal_filtered / np.max(np.abs(signal_filtered))

        return signal_filtered


if __name__ == "__main__":
    filtro = MatchedFilter(alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC-Inverted")

    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0, 0),
        filtro.t_impulse, filtro.impulse_response,
        t_unit="ms",
        colors="darkorange",
    ).plot(label=r"$g(-t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_mf_impulse.pdf")
    
    filtro = MatchedFilter(alpha=0.8, fs=128_000, Rb=400, span=6, type="Manchester-Inverted")

    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0, 0),
        filtro.t_impulse, filtro.impulse_response,
        t_unit="ms",
        colors="darkorange",
    ).plot(label=r"$g(-t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_mf_impulse_manchester.pdf")