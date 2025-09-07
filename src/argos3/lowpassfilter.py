"""
Implementa uma classe de filtro passa baixa para remover a componente de frequência alta do sinal recebido.

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from .plotter import create_figure, save_figure, ImpulseResponsePlot, TimePlot, PoleZeroPlot, FrequencyResponsePlot

class LPF:
    def __init__(self, cut_off, order, fs=128_000, type="butter"):
        r"""
        Inicializa um filtro passa-baixa com base em uma frequência de corte $f_{cut}$ e uma ordem $N$.

        Args:
            cut_off (float): Frequência de corte $f_{cut}$ do filtro.
            order (int): Ordem $N$ do filtro.
            fs (int, opcional): Frequência de amostragem $f_s$. 
            type (str, opcional): Tipo de filtro. Padrão é "butter".
        
        Raises:
            ValueError: Se o tipo de filtro for inválido.

        Exemplo: 
            ![pageplot](assets/example_lpf_signals.svg) 
        """

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
        r"""
        Calcula os coeficientes do filtro Butterworth utilizando a biblioteca `scipy.signal`. A função de transferência contínua $H(s)$ de um filtro Butterworth é dada pela expressão abaixo.

        $$
        H(s) = \frac{1}{1 + \left(\frac{s}{2 \pi f_{cut}}\right)^{2n}}
        $$

        Sendo:
            - $s$: Variável complexa no domínio de Laplace.
            - $2 \pi f_{cut}$: Frequência angular de corte do filtro.
            - $n$: Ordem do filtro.

        Args: 
            fNyquist (float): Fator de Nyquist. Padrão é 0.5 * fs.

        Returns:
            tuple: Coeficientes $b$ e $a$ correspondentes à função de transferência do filtro Butterworth.

        Exemplos:
            ![pageplot](assets/example_lpf_pz.svg)
        """
        b, a = butter(self.order, self.cut_off / (fNyquist * self.fs), btype='low')
        return b, a

    def calc_impulse_response(self, impulse_len=1024):
        r"""
        Para obter a resposta ao impulso no dominio do tempo, um impulso unitário é aplicado como entrada. Para um filtro Butterworth, o calculo é dado pela expressão abaixo. 

        $$
        h(t) = \mathcal{L}^{-1}\left\{H(f)\right\}
        $$

        Sendo:
            - $h(t)$: Resposta ao impulso do filtro.
            - $H(f)$: Função de transferência do filtro.
            - $\mathcal{L}^{-1}$: Transformada de Laplace inversa.

        Args: 
            impulse_len (int): Comprimento do vetor de impulso.

        Returns:
            impulse_response (tuple[np.ndarray, np.ndarray]): Resposta ao impulso e vetor de tempo.
        
        Exemplo: 
            ![pageplot](assets/example_lpf_impulse.svg)
        """
        # Impulso unitário
        impulse_input = np.zeros(impulse_len)
        impulse_input[0] = 1

        # Resposta ao impulso
        impulse_response = lfilter(self.b, self.a, impulse_input)
        t_impulse = np.arange(impulse_len) / self.fs
        return impulse_response, t_impulse

    def apply_filter(self, signal):
        r"""
        Aplica o filtro passa-baixa com resposta ao impulso $h(t)$ ao sinal de entrada $s(t)$, utilizando a função `scipy.signal.filtfilt`. O processo de filtragem é dado pela expressão abaixo. 

        $$
            x(t) = s(t) \ast h(t)
        $$

        Sendo: 
            - $x(t)$: Sinal filtrado.
            - $s(t)$: Sinal de entrada.
            - $h(t)$: Resposta ao impulso do filtro.

        Args:
            signal (np.ndarray): Sinal de entrada $s(t)$.

        Returns:
            signal_filtered (np.ndarray): Sinal filtrado $x(t)$.
        """
        signal_filtered = filtfilt(self.b, self.a, signal)

        # normalização
        signal_filtered *= np.sqrt(2)

        return signal_filtered


if __name__ == "__main__":
    
    fs = 128_000
    t = np.arange(10000) / fs

    # create two cossine signals with different frequencies
    f1 = 1000
    f2 = 4000
    signal = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)

    filtro = LPF(cut_off=1500, order=6, fs=fs, type="butter")
    signal_filtered = filtro.apply_filter(signal)

    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0, 0),
        filtro.t_impulse, filtro.impulse_response,
        t_unit="ms",
        colors="darkorange",
    ).plot(label=r"$h(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(0, 5))

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_lpf_impulse.pdf")

    fig_signal, grid_signal = create_figure(2, 2, figsize=(16, 9))

    ImpulseResponsePlot(
        fig_signal, grid_signal, (0, slice(0, 2)),
        filtro.t_impulse, filtro.impulse_response,
        t_unit="ms",
        colors="darkorange",
    ).plot(label=r"$h(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(0, 5))
    
    TimePlot(
        fig_signal, grid_signal, (1, 0),
        t, 
        signal,
        labels=[r"$x(t)$"],
        title="Sinal original",
        xlim=(0, 8),
        ylim=(-4, 4),
        colors="navy"
    ).plot()

    TimePlot(
        fig_signal, grid_signal, (1, 1),
        t, 
        signal_filtered,
        labels=[r"$x'(t)$"],
        title="Sinal filtrado",
        xlim=(0, 8),
        ylim=(-4, 4),
        colors="darkred"
    ).plot()

    fig_signal.tight_layout()
    save_figure(fig_signal, "example_lpf_signals.pdf")

    fig_pz, grid_pz = create_figure(1, 1, figsize=(16,9))
    PoleZeroPlot(fig_pz, grid_pz, (0,0), filtro.b, filtro.a).plot()
    save_figure(fig_pz, "example_lpf_pz.pdf")

    freq_response, grid_freq_response = create_figure(1, 1, figsize=(16,6))
    FrequencyResponsePlot(freq_response, grid_freq_response, (0,0), filtro.b, filtro.a, fs=filtro.fs, f_cut=filtro.cut_off, xlim=(0, 3*filtro.cut_off)).plot()
    save_figure(freq_response, "example_lpf_freq_response.pdf")
    