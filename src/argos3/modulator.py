"""
Implementação de um modulador IQ para transmissão de sinais digitais.

Autor: Arthur Cadore
Data: 28-07-2025
"""
import numpy as np
from .formatter import Formatter
from .encoder import Encoder
from .plotter import PhasePlot, create_figure, save_figure, TimePlot, FrequencyPlot, ConstellationPlot 

class Modulator:
    def __init__(self, fc, fs=128_000):
        r"""
        Inicializa um modulador QPSK no padrão ARGOS-3. O modulador pode ser representado pelo diagrama de blocos apresentado abaixo.

        ![pageplot](../assets/modulador.svg)

        Args:
            fc (float): Frequência da portadora.
            fs (int): Frequência de amostragem.

        Raises:
            ValueError: Se a frequência de amostragem não for maior que o dobro da frequência da portadora. (Teorema de Nyquist)
       
        Exemplo: 
            ![pageplot](assets/transmitter_modulator_time.svg)

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.2.5.3)
        </div>
        """
        if fc <= 0:
            raise ValueError("A frequência da portadora deve ser maior que zero.")
        
        if fs <= fc*2:
            raise ValueError("A frequência de amostragem deve ser maior que o dobro da frequência da portadora.")
        
        self.fc = fc
        self.fs = fs

    def modulate(self, i_signal, q_signal):
        r"""
        Modula em QPSK os sinais $d_I$ e $d_Q$ com uma portadora $f_c$, resultando no sinal modulado $s(t)$. O processo de modulação é dado pela expressão abaixo.

        $$
            s(t) = d_I(t) \cdot \cos(2\pi f_c t) - d_Q(t) \cdot \sin(2\pi f_c t)
        $$

        Sendo: 
            - $s(t)$: Sinal modulado.
            - $d_I(t)$ e $d_Q(t)$: Sinais formatados correspondentes aos canais $I$ e $Q$.
            - $f_c$: Frequência da portadora.
            - $t$: Vetor de tempo.

        Args:
            i_signal (np.ndarray): Sinal $d_I$ correspondente ao canal $I$ a ser modulado.
            q_signal (np.ndarray): Sinal $d_Q$ correspondente ao canal $Q$ a ser modulado.

        Returns:
            modulated_signal (np.ndarray): Sinal modulado $s(t)$ resultante.
            t (np.ndarray): Vetor de tempo $t$ correspondente ao sinal modulado.

        Raises:
            ValueError: Se os sinais I e Q não tiverem o mesmo tamanho.
        """
        n = len(i_signal)
        if len(q_signal) != n:
            raise ValueError("i_signal e q_signal devem ter o mesmo tamanho.")
        
        t = np.arange(n) / self.fs
        carrier_cos = np.cos(2 * np.pi * self.fc * t)
        carrier_sin = np.sin(2 * np.pi * self.fc * t)
        
        modulated_signal = (i_signal * carrier_cos - q_signal * carrier_sin)

        # normalização
        modulated_signal = modulated_signal / np.max(np.abs(modulated_signal))

        return t, modulated_signal
    
    def demodulate(self, modulated_signal):
        r"""
        Demodula o sinal modulado em QPSK. Para o processo de demodulação, utiliza-se duas componentes auxiliares $x_I(t)$ e $y_Q(t)$ definidas pelas expressões abaixo.

        $$
        \begin{aligned}
        x_I(t) &= 2 \cos(2\pi f_c t), &\quad
        y_Q(t) &= 2 \sin(2\pi f_c t)
        \end{aligned}
        $$

        Sendo: 
            - $x_I(t)$ e $y_Q(t)$: Componentes auxiliares utilizadas para a demodulação.
            - $f_c$: Frequência da portadora.
            - $t$: Vetor de tempo.

        A partir das componentes criadas é realizado o processo de demodulação (translação em frequência), que resulta em duas componentes, uma em banda base e outra em $2f_c$, conforme as expressões abaixo.

        $$
        x_I'(t) = s(t) \cdot x_I(t) = \left[d_I(t) \cos(2\pi f_c t ) - d_Q(t) \sin(2\pi f_c t )\right] \cdot 2\cos(2\pi f_c t )
        $$

        $$
        y_Q'(t) = -s(t) \cdot y_Q(t) = \left[d_I(t) \cos(2\pi f_c t ) - d_Q(t) \sin(2\pi f_c t )\right] \cdot 2\sin(2\pi f_c t )
        $$

        Sendo: 
            - $x_I'(t)$ e $y_Q'(t)$: Componentes resultantes da demodulação.
            - $x_I(t)$ e $y_Q(t)$: Componentes auxiliares utilizadas para a demodulação.
            - $s(t)$: Sinal modulado.
            - $d_I(t)$ e $d_Q(t)$: Sinais formatados correspondentes aos canais $I$ e $Q$.
            - $f_c$: Frequência da portadora.
            - $t$: Vetor de tempo.

        Args:
            modulated_signal (np.ndarray): Sinal modulado $s(t)$ a ser demodulado.

        Returns:
            i_signal (np.ndarray): Sinal $x_I'(t)$ recuperado.
            q_signal (np.ndarray): Sinal $y_Q'(t)$ recuperado.
        
        Raises:
            ValueError: Se o sinal modulado estiver vazio.
        """
        n = len(modulated_signal)
        if n == 0:
            raise ValueError("O sinal modulado não pode estar vazio.")
        
        t = np.arange(n) / self.fs
        carrier_cos = 2 * np.cos(2 * np.pi * self.fc * t)
        carrier_sin = 2 * np.sin(2 * np.pi * self.fc * t)
        
        i_signal = modulated_signal * carrier_cos
        q_signal = -modulated_signal * carrier_sin
        
        return i_signal, q_signal

if __name__ == "__main__":

    fs = 128_000
    fc = 4000
    Rb = 400
    alpha = 0.8
    span = 8

    Xnrz = np.random.randint(0, 2, 900)
    Yman = np.random.randint(0, 2, 900)

    encoder_nrz = Encoder(method="NRZ")
    encoder_man = Encoder(method="Manchester")

    Xnrz = encoder_nrz.encode(Xnrz)
    Yman = encoder_man.encode(Yman)

    print("Xnrz:", ''.join(str(b) for b in Xnrz[:20]))
    print("Yman:", ''.join(str(b) for b in Yman[:20]))

    formatterI = Formatter(alpha=alpha, fs=fs, Rb=Rb, span=span, channel="I")
    formatterQ = Formatter(alpha=alpha, fs=fs, Rb=Rb, span=span, channel="Q")
    
    dI = formatterI.apply_format(Xnrz)
    dQ = formatterQ.apply_format(Yman)

    print("dI:", ''.join(str(b) for b in dI[:5]))
    print("dQ:", ''.join(str(b) for b in dQ[:5]))
    
    modulator = Modulator(fc=fc, fs=fs)
    t, s = modulator.modulate(dI, dQ)
    print("s:", ''.join(str(b) for b in s[:5]))

    # PLOT 1 - Tempo
    fig_time, grid = create_figure(2, 1, figsize=(16, 9))
    TimePlot(
        fig_time, grid, (0, 0),
        t=t,
        signals=[dI, dQ],
        labels=["$dI(t)$", "$dQ(t)$"],
        title="Sinal $IQ$ - Formatados RRC",
        xlim=(40, 140),
        colors=["darkgreen", "navy"],
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    TimePlot(
        fig_time, grid, (1, 0),
        t=t,
        signals=[s],
        labels=["$s(t)$"],
        title="Sinal Modulado $IQ$",
        xlim=(40, 140),
        colors="darkred",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    fig_time.tight_layout()
    save_figure(fig_time, "example_modulator_time.pdf")

    # PLOT 2 - Frequência
    fig_freq, grid = create_figure(2, 2, figsize=(16, 9))
    FrequencyPlot(
        fig_freq, grid, (0, 0),
        fs=fs,
        signal=dI,
        fc=fc,
        labels=["$D_I(f)$"],
        title="Componente I",
        xlim=(-1.5, 1.5),
        colors="navy",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()

    FrequencyPlot(
        fig_freq, grid, (0, 1),
        fs=fs,
        signal=dQ,
        fc=fc,
        labels=["$D_Q(f)$"],
        title="Componente Q",
        xlim=(-1.5, 1.5),
        colors="darkgreen",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()

    FrequencyPlot(
        fig_freq, grid, (1, slice(0, 2)),
        fs=fs,
        signal=s,
        fc=fc,
        labels=["$S(f)$"],
        title="Sinal Modulado $IQ$",
        xlim=(-10, 10),
        colors="darkred",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()

    fig_freq.tight_layout()
    save_figure(fig_freq, "example_modulator_freq.pdf")

    # PLOT 3 - Constelação
    fig_const, grid = create_figure(1, 2, figsize=(16, 8))
    PhasePlot(
        fig_const, grid, (0, 0),
        t=t,
        signals=[dI, dQ],
        labels=["Fase $I + jQ$"],
        title="Fase $I + jQ$",
        xlim=(40, 140),
        ylim=(-np.pi, np.pi),
        colors=["darkred"],
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()

    ConstellationPlot(
        fig_const, grid, (0, 1),
        dI=dI[:20000:5],
        dQ=dQ[:20000:5],
        title="Constelação $IQ$",
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        colors=["darkred"],
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()

    fig_const.tight_layout()
    save_figure(fig_const, "example_modulator_constellation.pdf")

    # Plot 4 - Portadora pura e sinal modulado
    fig_portadora, grid = create_figure(1, 2, figsize=(16, 8))
    FrequencyPlot(
        fig_portadora, grid, (0, 0),
        fs=fs,
        signal=s[0:(int(round(0.082 * fs)))],
        fc=fc,
        labels=["$S(f)$"],
        title="Portadora Pura - $0$ a $80$ms",
        xlim=(-10, 10),
        colors="darkred",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()

    FrequencyPlot(
        fig_portadora, grid, (0, 1),
        fs=fs,
        signal=s[(int(round(0.082 * fs))):],
        fc=fc,
        labels=["$S(f)$"],
        title="Sinal Modulado - $80$ms em diante",
        xlim=(-10, 10),
        colors="darkred",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()

    fig_portadora.tight_layout()
    save_figure(fig_portadora, "example_modulator_portadora.pdf")
    
    # Demodulação
    i_signal, q_signal = modulator.demodulate(s)
    print("i_signal:", ''.join(str(b) for b in i_signal[:5]))
    print("q_signal:", ''.join(str(b) for b in q_signal[:5]))

    # PLOT 1 - Tempo
    fig_time, grid = create_figure(2, 1, figsize=(16, 9))
    TimePlot(
        fig_time, grid, (0, 0),
        t=t,
        signals=[i_signal, q_signal],
        labels=["$xI'(t)$", "$yQ'(t)$"],
        title="Componentes $IQ$ - Demoduladas",
        xlim=(40, 140),
        colors=["darkgreen", "navy"],
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    TimePlot(
        fig_time, grid, (1, 0),
        t=t,
        signals=[s],
        labels=["$s(t)$"],
        title="Sinal Modulado $IQ$",
        xlim=(40, 140),
        colors="darkred",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    fig_time.tight_layout()
    save_figure(fig_time, "example_demodulator_time.pdf")
    

    # PLOT 2 - Frequência
    fig_freq, grid = create_figure(3, 1, figsize=(16, 9))
    FrequencyPlot(
        fig_freq, grid, (0, 0),
        fs=fs,
        signal=s,
        fc=fc,
        labels=["$S(f)$"],
        title="Sinal Modulado $IQ$",
        xlim=(-10, 10),
        colors="darkred",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    FrequencyPlot(
        fig_freq, grid, (1, 0),
        fs=fs,
        signal=i_signal,
        fc=fc,
        labels=["$X_I'(f)$"],
        title="Componente $I$ - Demodulado",
        xlim=(-10, 10),
        colors="darkgreen",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()

    FrequencyPlot(
        fig_freq, grid, (2, 0),
        fs=fs,
        signal=q_signal,
        fc=fc,
        labels=["$Y_Q'(f)$"],
        title="Componente $Q$ - Demodulado",
        xlim=(-10, 10),
        colors="navy",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    

    fig_freq.tight_layout()
    save_figure(fig_freq, "example_demodulator_freq.pdf")