"""
Implementação de um canal para aplicação de ruido AWGN.

Autor: Arthur Cadore
Data: 16-08-2025
"""

import numpy as np
from datagram import Datagram
from transmitter import Transmitter
from plotter import save_figure, create_figure, TimePlot, FrequencyPlot

class Noise:
    def __init__(self, snr=10):
        r"""
        Implementação de canal para aplicação de ruido AWGN no sinal transmitido.

        Args:
            snr (float): Relação sinal-ruído em decibéis (dB). Padrão é 10 dB.
        """
        self.snr = snr
    
    def add_noise(self, signal):
        r"""
        Adiciona ruído AWGN ao sinal fornecido.

        Args:
            signal (np.ndarray): Sinal ao qual o ruído será adicionado.
        
        Returns:
            np.ndarray: Sinal com ruído adicionado.
        """
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (self.snr / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise
    
class NoiseEBN0:
    def __init__(self, ebn0_db, bits_per_symbol=2, rng=None, fs=128_000, Rb=400):
        r"""
        Implementação de canal AWGN, controlado por $Eb$/$N_{0}$.

        Args:
            ebn0_db (float): Valor alvo de$Eb$/$N_{0}$ em decibéis ($dB$).
            bits_per_symbol (int): Número de bits por símbolo ($k$). Para QPSK, $k=2$.
            rng (np.random.Generator, opcional): Gerador de números aleatórios (NumPy).
            fs (int): Taxa de amostragem do sinal em $Hz$.
            Rb (int): Taxa de bits em bits/s.
        """
        self.ebn0_db = ebn0_db
        self.ebn0_lin = 10 ** (ebn0_db / 10)
        self.k = bits_per_symbol
        self.rng = rng if rng is not None else np.random.default_rng()
        self.fs = fs
        self.Rb = Rb
        self.Rs = Rb / bits_per_symbol  # taxa de símbolos

    def add_noise(self, s):
        r"""
        Adiciona ruído AWGN ao sinal de entrada 

        Etapas do cálculo:
            1. Calcula a potência média do sinal amostrado:
               $P = \mathbb{E}[|s|^2]$.
            2. Calcula a energia por bit: $E_b = P / R_b$.
            3. Calcula a densidade espectral de ruído:
               $N_0 = E_b / (E_b/N_0)$.
            4. Converte $N_0$ em variância de amostras discretas:
               $\sigma^2 = \dfrac{N_0 \cdot f_s}{2}$.
            5. Gera vetor gaussiano $n(t)$ com variância $\sigma^2$.
            6. Adiciona ruído ao sinal: $r(t) = s(t) + n(t)$.

        Args:
            s (np.ndarray): Sinal transmitido $s(t)$.

        Returns:
            s_noisy (np.ndarray): Sinal recebido $r(t) = s(t) + n(t)$, com ruído AWGN.
        """
        # Potência média do sinal (por amostra)
        P = np.mean(np.abs(s)**2)

        # Energia por bit
        Eb = P / self.Rb

        # Densidade espectral de potência do ruído
        N0 = Eb / self.ebn0_lin

        # Variância do ruído discreto por amostra (real)
        sigma2 = (N0 * self.fs) / 2.0
        sigma = np.sqrt(sigma2)

        # Ruído gaussiano branco
        n = self.rng.normal(0.0, sigma, size=s.shape)
        return s + n

def check_ebn0(s, s_noisy, add_noise:NoiseEBN0):
    n_est = s_noisy - s
    P = np.mean(s**2)
    Eb = P / add_noise.Rb
    # de sigma^2 -> N0 estimado:
    sigma2_meas = np.var(n_est)
    N0_meas = 2 * sigma2_meas / add_noise.fs
    ebn0_meas_db = 10*np.log10(Eb / N0_meas)
    print("Eb/N0 alvo:", add_noise.ebn0_db, "dB | medido:", ebn0_meas_db, "dB")
    

if __name__ == "__main__":
    datagram = Datagram(pcdnum=1234, numblocks=1)
    transmitter = Transmitter(datagram, output_print=False)
    t, s = transmitter.run()

    # snr_db = 15
    # add_noise = Noise(snr=snr_db)
    # s_noisy = add_noise.add_noise(s)

    eb_n0 = 12
    add_noise = NoiseEBN0(eb_n0)
    s_noisy = add_noise.add_noise(s)
    check_ebn0(s, s_noisy, add_noise)


    fig_time, grid_time = create_figure(2, 1, figsize=(16, 9))

    TimePlot(
        fig_time, grid_time, (0,0),
        t=t,
        signals=[s],
        labels=["$s(t)$"],
        title="Domínio do Tempo - Sem Ruído",
        xlim=(0, 0.1),
        ylim=(-0.1, 0.1),
        colors="darkblue",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    TimePlot(
        fig_time, grid_time, (1,0),
        t=t,
        signals=[s_noisy],
        labels=["$s(t) + AWGN$"],
        title="Domínio do Tempo - Com Ruído",
        xlim=(0, 0.1),
        ylim=(-0.1, 0.1),
        colors="darkred",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    fig_time.tight_layout()
    save_figure(fig_time, "example_noise_time.pdf")

    fig_freq, grid_freq = create_figure(2, 1, figsize=(16, 9))

    FrequencyPlot(
        fig_freq, grid_freq, (0,0),
        fs=transmitter.fs,
        signal=s,
        fc=transmitter.fc,
        labels=["$S(f)$"],
        title="Domínio da Frequência - Sem Ruído",
        xlim=(-8, 8),
        colors="darkblue",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    FrequencyPlot(
        fig_freq, grid_freq, (1,0),
        fs=transmitter.fs,
        signal=s_noisy,
        fc=transmitter.fc,
        labels=["$S(f) + AWGN$"],
        title="Domínio da Frequência - Com Ruído",
        xlim=(-8, 8),
        colors="darkred",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    fig_freq.tight_layout()
    save_figure(fig_freq, "example_noise_freq.pdf")