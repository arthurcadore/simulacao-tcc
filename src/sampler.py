"""
Implementação de um decisor (amostrador e quantizador) para recepção.

Autor: Arthur Cadore
Data: 15-08-2025
"""

import numpy as np
from plotter import save_figure, create_figure, SampledSignalPlot

class Sampler:
    def __init__(self, fs=128_000, Rb=400, t=None, output_print=True, output_plot=True):
        r"""
        Implementação de decisor (amostragem e quantização) para o receptor.

        Args: 
            fs (int): Frequência de amostragem.
            Rb (int): Taxa de bits.
            t (numpy.ndarray): Vetor de tempo.
            output_print (bool): Se True, imprime a saída.
            output_plot (bool): Se True, plota a saída.
        """
        self.fs = fs
        self.Rb = Rb
        self.sps = int(self.fs / self.Rb)
        self.output_print = output_print
        self.output_plot = output_plot
        self.delay = 0
        self.indexes = self.calc_indexes(t)
    
    def calc_indexes(self, t):
        r"""
        Calcula os índices de amostragem com base no vetor de tempo.

        Args:
            t (numpy.ndarray): Vetor de tempo.

        Returns:
            numpy.ndarray: Índices de amostragem.
        """
        indexes = np.arange(self.delay, len(t), self.sps)
        indexes = indexes[indexes < len(t)]
        return indexes
    
    def sample(self, signal):
        r"""
        Amostra o sinal com base nos índices de amostragem.

        Args:
            signal (numpy.ndarray): Sinal a ser amostrado.

        Returns:
            numpy.ndarray: Sinal amostrado.
        """
        sampled_signal = signal[self.indexes]
        return sampled_signal

    def quantize(self, signal):
        r"""
        Quantiza o sinal em uma representação binária.

        Args:
            signal (numpy.ndarray): Sinal a ser quantizado.

        Returns:
            list: Representação binária do sinal quantizado.
        """
        bits = []
        for i in range(len(signal)):
            if signal[i] > 0:
                bits.append(1)
            else:
                bits.append(0)
        return bits

if __name__ == "__main__":

    fs = 128_000
    Rb = 2000
    t = np.arange(10000) / fs
    signal = np.cos(2 * np.pi * 1000 * t) + np.cos(2 * np.pi * 4000 * t)
    signal2 = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 4000 * t)

    sampler = Sampler(fs=fs, Rb=Rb, t=t)
    sampled_signal = sampler.sample(signal)
    sampled_time = sampler.sample(t)

    sampler2 = Sampler(fs=fs, Rb=Rb, t=t)
    sampled_signal2 = sampler2.sample(signal2)
    sampled_time2 = sampler2.sample(t)

    bits = sampler.quantize(sampled_signal)
    bits2 = sampler2.quantize(sampled_signal2)
    print(bits)
    print(bits2)

    fig_sampler, grid_sampler = create_figure(2, 1, figsize=(16, 9))

    SampledSignalPlot(
        fig_sampler, grid_sampler, (0, 0),
        t,
        signal,
        sampled_time,
        sampled_signal,
        colors='red'
    ).plot(label_signal="Sinal original", label_samples="Amostras", x_lim=0.01, title="Sinal $Cos(t)$ amostrado")

    SampledSignalPlot(
        fig_sampler, grid_sampler, (1, 0),
        t,
        signal2,
        sampled_time2,
        sampled_signal2,
        colors='navy'
    ).plot(label_signal="Sinal original", label_samples="Amostras", x_lim=0.01, title="Sinal $Sin(t)$ amostrado")

    fig_sampler.tight_layout()
    save_figure(fig_sampler, "example_sampler_time.pdf")