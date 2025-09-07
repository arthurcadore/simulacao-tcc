"""
Implementação de um decisor (amostrador e quantizador) para recepção.

Autor: Arthur Cadore
Data: 15-08-2025
"""

import numpy as np
from .plotter import save_figure, create_figure, SampledSignalPlot

class Sampler:
    def __init__(self, fs=128_000, Rb=400, t=None, delay=0.08):
        r"""
        Inicializa o decisor, utilizado para amostragem e quantização no receptor.

        Args: 
            fs (int): Frequência de amostragem.
            delay (float): Delay de amostragem, em segundos.
            Rb (int): Taxa de bits.
            t (numpy.ndarray): Vetor de tempo.

        Exemplo: 
            ![pageplot](assets/example_sampler_time.svg) 
        """
        self.fs = fs
        self.Rb = Rb
        self.sps = int(self.fs / self.Rb)
        self.delay = int(round(delay * self.fs))
        self.indexes = self.calc_indexes(t)
    
    def calc_indexes(self, t):
        r"""
        Calcula os índices de amostragem $I[n]$ com base no vetor de tempo $t$. O vetor de índices de amostragem $I[n]$ é dado pela expressão abaixo. 

        $$
        \begin{align}
        I[n] = \tau + n \cdot \left( \frac{f_s}{R_b}\right) \text{ , onde: } \quad I[n] < \text{len}(t)
        \end{align}
        $$

        Sendo:
            - $\tau$: Delay inicial de amostragem.
            - $f_s$: Frequência de amostragem.
            - $R_b$: Taxa de bits.
            - $n$: Índice da amostra.
            - $\text{len}(t)$: Comprimento do vetor de tempo.

        Args:
            t (numpy.ndarray): Vetor de tempo.

        Returns:
            indexes (numpy.ndarray): Vetor de índices de amostragem $I[n]$.
        """
        indexes = np.arange(self.delay, len(t), self.sps)
        indexes = indexes[indexes < len(t)]
        return indexes
    
    def sample(self, signal):
        r"""
        Amostra o sinal $s(t)$ com base nos índices de amostragem $I[n]$.

        $$
            s(t) \rightarrow  s([I[n]) \rightarrow s[n]
        $$

        Sendo:
            - $s(t)$: Sinal de entrada $s(t)$.
            - $s[n]$ Sinal amostrado $s[n]$.
            - $I[n]$ Índices de amostragem $I[n]$.

        Args:
            signal (numpy.ndarray): Sinal de entrada $s(t)$ a ser amostrado.

        Returns:
            sampled_signal (numpy.ndarray): Sinal amostrado $s[n]$.
        """
        sampled_signal = signal[self.indexes]
        return sampled_signal

    def quantize(self, signal):
        r"""
        Quantiza o sinal $s[n]$ em valores discretos. O processo de quantização é dado pela expressão abaixo.

        $$
        \begin{align}
        s'[n] = \begin{cases}
            +1 & \text{se } s[n] \geq 0 \\
            -1 & \text{se } s[n] < 0
        \end{cases}
        \end{align}
        $$

        Sendo:
            - $s[n]$ Simbolos amostrados $s[n]$.
            - $s'[n]$ Símbolos quantizados $s'[n]$.

        Args:
            signal (numpy.ndarray): Sinal de entrada $s[n]$.

        Returns:
            symbols (numpy.ndarray): Símbolos quantizados $s'[n]$.
        """
        symbols = []
        for i in range(len(signal)):
            if signal[i] >= 0:
                symbols.append(+1)
            else:
                symbols.append(-1)
        return symbols

if __name__ == "__main__":

    fs = 128_000
    Rb = 2000
    t = np.arange(100000) / fs
    signal = np.cos(2 * np.pi * 400 * t) + np.cos(2 * np.pi * 100 * t)
    signal2 = np.sin(2 * np.pi * 400 * t) + np.sin(2 * np.pi * 1000 * t)

    sampler = Sampler(fs=fs, Rb=Rb, t=t)
    sampled_signal = sampler.sample(signal)
    sampled_time = sampler.sample(t)

    sampler2 = Sampler(fs=fs, Rb=Rb, t=t)
    sampled_signal2 = sampler2.sample(signal2)
    sampled_time2 = sampler2.sample(t)

    symbols = sampler.quantize(sampled_signal)
    symbols2 = sampler2.quantize(sampled_signal2)
    print(symbols[:20], "...")
    print(symbols2[:20], "...")

    fig_sampler, grid_sampler = create_figure(2, 1, figsize=(16, 9))

    SampledSignalPlot(
        fig_sampler, grid_sampler, (0, 0),
        t,
        signal,
        sampled_time,
        sampled_signal,
        colors='red'
    ).plot(label_signal="Sinal original", label_samples="Amostras", xlim=(40, 100), title="Sinal $Cos(t)$ amostrado")

    SampledSignalPlot(
        fig_sampler, grid_sampler, (1, 0),
        t,
        signal2,
        sampled_time2,
        sampled_signal2,
        colors='navy'
    ).plot(label_signal="Sinal original", label_samples="Amostras", xlim=(40, 100), title="Sinal $Sin(t)$ amostrado")

    fig_sampler.tight_layout()
    save_figure(fig_sampler, "example_sampler_time.pdf")