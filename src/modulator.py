"""
Implementação de um modulador IQ para transmissão de sinais digitais.

Autor: Arthur Cadore
Data: 28-07-2025
"""
import numpy as np
from formatter import Formatter
from encoder import Encoder
from plotter import create_figure, save_figure, TimePlot, FrequencyPlot, ConstellationPlot 

class Modulator:
    def __init__(self, fc, fs):
        r"""
        Inicializa uma instância do modulador IQ.
        O modulador IQ é responsável por modular os sinais I e Q em uma portadora de frequência específica.

        Args:
            fc (float): Frequência da portadora.
            fs (int): Frequência de amostragem.

        Raises:
            ValueError: Se a frequência de amostragem não for maior que o dobro da frequência da portadora. (Teorema de Nyquist)
        """
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
        
        modulated_signal = (i_signal * carrier_cos - q_signal * carrier_sin)
        return t, modulated_signal
    
    def demodulate(self, modulated_signal):
        r"""
        Demodula o sinal modulado para recuperar os sinais I e Q originais. 

        Para o processo de demodulação, utilizamos os sinais de portadora $x_I(t)$ e $y_Q(t)$ definidos como:

        $$
            x_I(t) = 2 \cos(2\pi f_c t)
        $$

        $$
            y_Q(t) = 2 \sin(2\pi f_c t)
        $$

        Nota: 
            - A constante 2 é utilizada para manter a amplitude do sinal original, devido a translação do sinal modulado.
        
        O processo resulta em dois sinais, contendo uma componente em banda base e outra em banda $2f_c$: 

        $$
        x_I'(t) = s(t) \cdot x_I(t) = \left[Ad_I(t) \cos(2\pi f_c t ) - Ad_Q(t) \sin(2\pi f_c t )\right] \cdot 2\cos(2\pi f_c t )
        $$

        $$
        y_Q'(t) = -s(t) \cdot y_Q(t) = \left[Ad_I(t) \cos(2\pi f_c t ) - Ad_Q(t) \sin(2\pi f_c t )\right] \cdot 2\sin(2\pi f_c t )
        $$

        Args:
            modulated_signal (np.ndarray): Sinal modulado $s(t)$ a ser demodulado.

        Returns:
            i_signal (np.ndarray): Sinal I recuperado.
            q_signal (np.ndarray): Sinal Q recuperado.
        
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
    fc = 2000
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

    # PLOT 1 - Tempo
    fig_time, grid = create_figure(2, 1, figsize=(16, 9))
    TimePlot(
        fig_time, grid, (0, 0),
        t=t,
        signals=[dI, dQ],
        labels=["$dI(t)$", "$dQ(t)$"],
        title="Sinal $IQ$ - Formatados RRC",
        xlim=(0, 0.1),
        ylim=(-0.1, 0.1),
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
        xlim=(0, 0.1),
        ylim=(-0.15, 0.15),
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
        xlim=(-4, 4),
        colors="darkred",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()

    fig_freq.tight_layout()
    save_figure(fig_freq, "example_modulator_freq.pdf")

    # PLOT 3 - Constelação
    fig_const, grid = create_figure(1, 2, figsize=(16, 8))
    TimePlot(
        fig_const, grid, (0, 0),
        t=t,
        signals=[dI, dQ],
        labels=["$dI(t)$", "$dQ(t)$"],
        title="Sinal $IQ$ - Formatados RRC",
        xlim=(0, 0.025),
        ylim=(-0.1, 0.1),
        colors=["darkgreen", "navy"],
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()

    ConstellationPlot(
        fig_const, grid, (0, 1),
        dI=dI[:20000],
        dQ=dQ[:20000],
        title="Constelação $IQ$",
        xlim=(-0.1, 0.1),
        ylim=(-0.1, 0.1),
        colors=["darkgreen", "navy"],
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()

    fig_const.tight_layout()
    save_figure(fig_const, "example_modulator_constellation.pdf")
    
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
        xlim=(0, 0.1),
        ylim=(-0.2, 0.2),
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
        xlim=(0, 0.1),
        ylim=(-0.15, 0.15),
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
        xlim=(-5.5, 5.5),
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
        xlim=(-5.5, 5.5),
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
        xlim=(-5.5, 5.5),
        colors="navy",
        style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    

    fig_freq.tight_layout()
    save_figure(fig_freq, "example_demodulator_freq.pdf")