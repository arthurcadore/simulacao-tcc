"""
Implementa uma classe de filtro passa baixa para remover a componente de frequência alta do sinal recebido.

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from plots import Plotter
from scipy.signal import butter, filtfilt, lfilter

class LPF:
    def __init__(self, cut_off, order, fs=128_000, type="butter"):
        self.cut_off = cut_off
        self.order = order
        self.fs = fs

        type = type.lower()
        if type != "butter":
            raise ValueError("Tipo de filtro inválido. Use 'butter'.")

        # Coeficientes b (numerador) e a (denominador) do filtro
        self.b, self.a = self.butterworth_filter()
        self.impulse_response, self.t_impulse = self.calc_impulse_response()

    def butterworth_filter(self, fNyquist=0.5):
        b, a = butter(self.order, self.cut_off / (fNyquist * self.fs), btype='low')
        return b, a

    def calc_impulse_response(self, impulse_len=512):
        # Impulso unitário
        impulse_input = np.zeros(impulse_len)
        impulse_input[0] = 1

        # Resposta ao impulso
        impulse_response = lfilter(self.b, self.a, impulse_input)
        t_impulse = np.arange(impulse_len) / self.fs
        return impulse_response, t_impulse

    def apply_filter(self, signal):
        signal_filtered = filtfilt(self.b, self.a, signal)

        # Remover offset DC
        signal_filtered -= np.mean(signal_filtered)

        # Normalizar amplitude
        signal_filtered *= 2

        return signal_filtered


if __name__ == "__main__":

    fs = 128_000

    # create two cossine signals with different frequencies
    f1 = 1000
    f2 = 4000
    t = np.arange(10000) / fs
    signal = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)

    filtro = LPF(cut_off=1500, order=6, fs=fs, type="butter")

    signal_filtered = filtro.apply_filter(signal)

    plotter = Plotter()
    plotter.plot_impulse_response(filtro.t_impulse,
                                  filtro.impulse_response,
                                  "Resposta ao Impulso - FPB",
                                  save_path="../out/example_lpf_impulse.pdf")

    plotter.plot_filtered_signals(filtro.t_impulse, 
                                  filtro.impulse_response, 
                                  t, 
                                  signal,
                                  signal_filtered,
                                  "Resposta ao Impulso - FPB",
                                  "Sinal original",
                                  "Sinal filtrado",
                                  "Resposta ao Impulso - FPB", 
                                  "Sinal original", 
                                  "Sinal filtrado", 
                                  0.01, 
                                  save_path="../out/example_lpf_signals.pdf")
