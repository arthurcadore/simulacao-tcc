"""
Implementa um formatador de pulso para transmissão de sinais digitais. 

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from .plotter import ImpulseResponsePlot, TimePlot, EncodedBitsPlot, create_figure, save_figure
from .encoder import Encoder

class Formatter:
    def __init__(self, alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC", prefix_duration=0.082, channel=None, bits_per_symbol=1):
        r"""
        Inicializa um formatador, utilizado preparar os símbolos para modulação.

        Args:
            alpha (float): Fator de roll-off do pulso RRC.
            fs (int): Frequência de amostragem.
            Rb (int): Taxa de bits.
            span (int): Duração do pulso em termos de períodos de bit.
            type (str): Tipo de pulso, atualmente apenas $RRC$ é suportado.
            prefix_duration (int): Duração da portadora pura no inicio do vetor
            channel (str): Canal a ser formatado, apenas $I$ e $Q$ são suportados.
            bits_per_symbol (int): Número de bits por símbolo.

        Raises:
            ValueError: Se o tipo de pulso não for suportado.
            ValueError: Se o canal não for suportado.

        Exemplo: 
            ![pageplot](assets/example_formatter_time.svg)

        <div class="referencia">
        <b>Referência:</b><br>
        EEL7062 – Princípios de Sistemas de Comunicação, Richard Demo Souza (Pg. 55)
        </div>
        """

        if channel not in ["I", "Q"]:
            raise ValueError("Canal inválido. Use 'I' ou 'Q'.")
        
        self.channel = channel
        self.prefix_duration = prefix_duration  
        self.alpha = alpha
        self.fs = fs
        self.Rb = Rb
        self.Tb = 1 / Rb
        self.sps = int(fs / Rb)
        self.span = span
        self.bits_per_symbol = bits_per_symbol
        self.t_rc = np.linspace(-span * self.Tb, span * self.Tb, span * self.sps * 2)

        type_map = {
            "rrc": 0,
            "manchester": 1
        }


        type = type.lower()
        if type not in type_map:
            raise ValueError("Tipo de pulso inválido. Use 'RRC' ou 'Manchester'.")
        
        self.type = type_map[type]

        if self.type == 0:  # RRC
            self.g = self.rrc_pulse()
        elif self.type == 1:  # Manchester
            self.g, self.g_left, self.g_right = self.manchester_pulse()

    def rrc_pulse(self, shift=0.0):
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

        Args: 
            - shift: Deslocamento no tempo.

        Returns:
           rc (np.ndarray): Pulso RRC normalizado.
        """
        self.t_rc = np.array(self.t_rc, dtype=float)
        # aplica deslocamento no tempo
        t_shifted = self.t_rc - shift

        rc = np.zeros_like(t_shifted)
        for i, ti in enumerate(t_shifted):
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
    
    def manchester_pulse(self):
        r"""
        Pulso Manchester como diferença de dois RRC simetricamente deslocados. 

        $$
            g_{MAN}(t) = g_{RRC}(t + T_b/2) - g_{RRC}(t - T_b/2)
        $$

        Sendo: 
            - $g_{MAN}(t)$: Pulso formatador Manchester no dominio do tempo.
            - $g_{RRC}(t)$: Pulso formatador $RRC$ no dominio do tempo.
            - $T_b$: Período de bit.
            - $t$: Vetor de tempo.

        Exemplo: 
            ![pageplot](assets/example_formatter_impulse_man.svg)
        """
        g_left = -self.rrc_pulse(shift=self.Tb/2)
        g_right = +self.rrc_pulse(shift=-self.Tb/2)
        g = g_left + g_right

        return g, g_left, g_right

    def apply_format(self, symbols, add_prefix=True):
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

        # adiciona prefixo
        if add_prefix:
            symbols = self.add_prefix(symbols)

        pulse = self.g
        # samples per symbol agora é por bit dividido pelo número de bits por símbolo
        sps = int(self.fs / (self.Rb / self.bits_per_symbol))

        upsampled = np.zeros(len(symbols) * sps)
        upsampled[::sps] = symbols
        out_sys = np.convolve(upsampled, pulse, mode='same')

        out_sys = out_sys / np.max(np.abs(out_sys))
        return out_sys

    def add_prefix(self, symbols):
        """
        Adiciona um prefixo de portadora pura no inicio do sinal. Para o canal $I$, adiciona um vetor de simbolos $+1$, para o canal $Q$, adiciona um vetor de simbolos $0$, com duração de `prefix_duration`, pois ao aplicar o modulador `IQ` temos uma portadora pura no inicio do sinal, conforme a expressão abaixo: 

        $$
            s(t) = 1(t) \cdot \cos(2\pi f_c t) - 0(t) \cdot \sin(2\pi f_c t) \mapsto s(t) = \cos(2\pi f_c t)
        $$

        Sendo: 
            - $s(t)$: Sinal modulado.
            - $1(t)$ e $0(t)$: Prefixo de portadora pura.
            - $f_c$: Frequência da portadora.
            - $t$: Vetor de tempo.
        
        Args:
            symbols (np.ndarray): Vetor de símbolos a serem formatados.
        
        Returns:
            symbols (np.ndarray): Vetor de símbolos com prefixo adicionado.
        """
        if self.channel == "I":
            carrier = np.ones(int(self.prefix_duration * self.Rb / self.bits_per_symbol))
        elif self.channel == "Q":
            carrier = np.zeros(int(self.prefix_duration * self.Rb / self.bits_per_symbol))

        symbols = np.concatenate([carrier, symbols])
        return symbols


if __name__ == "__main__":

    bitN = np.random.randint(0, 2, 10)
    bitM = np.ones(10)

    encoder_nrz = Encoder(method="NRZ")
    encoder_man = Encoder(method="Manchester")

    Xnrz1 = encoder_nrz.encode(bitN)
    Yman1 = encoder_man.encode(bitM)
    
    formatterI = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC", channel="I", bits_per_symbol=1)
    formatterQ = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC", channel="Q", bits_per_symbol=1)
    
    dI1 = formatterI.apply_format(Xnrz1)
    dQ1 = formatterQ.apply_format(Yman1)
    
    print("Xnrz:",  ' '.join(f"{x:+d}" for x in Xnrz1[:10]))
    print("Yman:",  ' '.join(f"{y:+d}" for y in Yman1[:10]))
    print("dI:", ''.join(str(b) for b in dI1[:5]))
    print("dQ:", ''.join(str(b) for b in dQ1[:5]))

    # Plotando a resposta ao impulso
    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0, 0),
        formatterI.t_rc, formatterI.g,
        t_unit="ms",
        colors="darkorange",
    ).plot(label=r"$g(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_formatter_impulse.pdf")

    # Plotando os sinais formatados
    
    fig_format, grid_format = create_figure(2, 2, figsize=(16, 9))

    ImpulseResponsePlot(
        fig_format, grid_format, (0, slice(0, 2)),
        formatterI.t_rc, formatterI.g,
        t_unit="ms",
        colors="darkorange",
    ).plot(label=r"$g(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))
    
    TimePlot(
        fig_format, grid_format, (1,0),
        t= np.arange(len(dI1)) / formatterI.fs,
        signals=[dI1],
        labels=[r"$d_I(t)$"],
        title=r"Canal $I$",
        xlim=(40, 140),
        colors="darkgreen",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    TimePlot(
        fig_format, grid_format, (1,1),
        t= np.arange(len(dQ1)) / formatterQ.fs,
        signals=[dQ1],
        labels=[r"$d_Q(t)$"],
        title=r"Canal $Q$",
        xlim=(40, 140),
        colors="darkblue",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    fig_format.tight_layout()
    save_figure(fig_format, "example_formatter_time.pdf")


    ##### TESTE V1.0.5
    encoder_nrz = Encoder(method="NRZ")
    encoder_man = Encoder(method="NRZ2")

    Xnrz2 = encoder_nrz.encode(bitN)
    Yman2 = encoder_man.encode(bitM)

    formatterI = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC", channel="I", bits_per_symbol=1)
    formatterQ = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6, type="Manchester", channel="Q", bits_per_symbol=2)

    dI2 = formatterI.apply_format(Xnrz2)
    dQ2 = formatterQ.apply_format(Yman2)

    fig_impulse, grid_impulse = create_figure(2, 1, figsize=(16, 9))
    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0, 0),
        formatterQ.t_rc, formatterQ.g,
        t_unit="ms",
        colors="darkorange",
    ).plot(label=r"$g_{MAN}(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (1, 0),
        formatterQ.t_rc, [formatterQ.g_left, formatterQ.g_right],
        t_unit="ms",
        colors=["darkorange", "steelblue"],
    ).plot(
        label=[r"$g_{L}(t)$", r"$g_{R}(t)$"],
        xlabel=r"Tempo ($ms$)",
        ylabel="Amplitude",
        xlim=(-15, 15)
    )

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_formatter_impulse_man.pdf")   

    # Plotando os sinais formatados
    fig_format, grid_format = create_figure(2, 2, figsize=(16, 9))
    
    EncodedBitsPlot(
        fig_format, grid_format, (0, 0),
        bits=Yman1,
        color='darkgreen',
    ).plot(xlabel="Index de Simbolo", ylabel="$Y_{MAN}[n]$", label="$Y_{MAN}[n]$")

    EncodedBitsPlot(
        fig_format, grid_format, (0, 1),
        bits=Yman2,
        color='darkgreen',
    ).plot(xlabel="Index de Simbolo", ylabel="$Y_{MAN}[n]$", label="$Y_{MAN}[n]$")


    TimePlot(
        fig_format, grid_format, (1,0),
        t= np.arange(len(dQ1)) / formatterQ.fs,
        signals=[dQ1],
        labels=[r"$d_Q(t)$"],
        title=r"Canal $Q$",
        # xlim=(40, 140),
        colors="darkblue",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()


    TimePlot(
        fig_format, grid_format, (1,1),
        t= np.arange(len(dQ2)) / formatterQ.fs,
        signals=[dQ2],
        labels=[r"$d_Q(t)$"],
        title=r"Canal $Q$",
        # xlim=(40, 140),
        colors="darkblue",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    fig_format.tight_layout()
    save_figure(fig_format, "example_formatter_time_comparison.pdf")
