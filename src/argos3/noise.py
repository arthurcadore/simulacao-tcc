"""
Implementação de um canal para aplicação de ruido AWGN.

Autor: Arthur Cadore
Data: 16-08-2025
"""

import numpy as np
from .datagram import Datagram
from .transmitter import Transmitter
from .plotter import save_figure, create_figure, TimePlot, FrequencyPlot, GaussianNoisePlot

class Noise:
    def __init__(self, snr=15, seed=None):
        r"""
        Implementação de canal para aplicação de ruido $AWGN$, com base em $SNR$.

        Args:
            snr (float): Relação sinal-ruído em decibéis (dB).
            seed (int): Seed do gerador de números aleatórios.

        Exemplo: 
            ![pageplot](assets/example_noise_time.svg) 
        """
        self.snr = snr
        self.rng = np.random.default_rng(seed)
    
    def add_noise(self, signal):
        r"""
        Adiciona ruído AWGN $n(t)$ ao sinal de entrada $s(t)$, com base na $\mathrm{SNR}_{dB}$ definida na inicialização. 

        $$
        r(t) = s(t) + n(t), \qquad n(t) \sim \mathcal{N}(0, \sigma^2)
        $$

        Sendo: 
            - $r(t)$: Sinal retornado com ruído AWGN adicionado.
            - $s(t)$: Sinal de entrada sem ruído. 
            - $n(t)$: Ruído adicionado, com distribuição normal $\mathcal{N}(0, \sigma^2)$.

        A variância do ruído $\sigma^2$ é dada por:

        $$
        \sigma^2 = \frac{\mathbb{E}\!\left[ |s(t)|^2 \right]}{10^{\frac{\mathrm{SNR}_{dB}}{10}}}
        $$

        Sendo: 
            - $\sigma^2$: A variância do ruído.
            - $\mathbb{E}\!\left[ |s(t)|^2 \right]$: Potência média do sinal de entrada.
            - $\mathrm{SNR}_{dB}$: Relação sinal-ruído em decibéis (dB).

        Args:
            signal (np.ndarray): Sinal transmitido $s(t)$.

        Returns:
            signal (np.ndarray): Sinal $r(t)$, com ruído AWGN adicionado.

        Exemplo:
            ![pageplot](assets/example_noise_gaussian_snr.svg)
        """

        self.signal_power = np.mean(np.abs(signal) ** 2)
        self.snr_linear = 10 ** (self.snr / 10)
        self.variance = self.signal_power / self.snr_linear
        
        # gera ruído com base na seed
        noise = self.rng.normal(0, np.sqrt(self.variance), len(signal))
        
        # adiciona ruído
        signal = signal + noise

        # normalização
        signal = signal / np.max(np.abs(signal))
        
        return signal 

class NoiseEBN0:
    def __init__(self, ebn0_db=10, fs=128_000, Rb=400, seed=None):
        r"""
        Implementação de canal para aplicação de ruido $AWGN$, com base em $Eb/N_{0}$.

        Args:
            ebn0_db (float): Valor alvo de $Eb/N_{0}$ em $dB$
            fs (int): Taxa de amostragem do sinal em $Hz$.
            Rb (int): Taxa de bits em bits/s.
            seed (int): Seed do gerador de números aleatórios.
        
        Exemplo: 
            ![pageplot](assets/example_noise_time.svg)
        """
        self.ebn0_db = ebn0_db
        self.ebn0_lin = 10 ** (ebn0_db / 10)
        self.fs = fs
        self.Rb = Rb

        # gera ruído com base na seed
        self.rng = np.random.default_rng(seed)

    def add_noise(self, signal):
        r"""
        Adiciona ruído AWGN $n(t) ao sinal de entrada $s(t), com base na $Eb/N0_{dB}$ definida na inicialização. 

        $$
        r(t) = s(t) + n(t), \qquad n(t) \sim \mathcal{N}(0, \sigma^2)
        $$

        Sendo: 
            - $r(t)$: Sinal retornado com ruído AWGN adicionado.
            - $s(t)$: Sinal de entrada sem ruído. 
            - $n(t)$: Ruído adicionado, com distribuição normal $\mathcal{N}(0, \sigma^2)$.

        
        A variância do ruído $\sigma^2$ é dada por:

        $$
        \sigma^2 = \frac{N_0 \cdot f_s}{2}
        $$

        Sendo: 
            - $\sigma^2$: A variância do ruído.
            - $N_0$: Densidade espectral de ruído.
            - $f_s$: Taxa de amostragem do sinal em $Hz$.

        
        A densidade espectral de ruído $N_0$ é dada por:

        $$
        N_0 = \frac{\mathbb{E}\!\left[ |s(t)|^2 \right]}{R_b \cdot 10^{\frac{Eb/N_0}{10}}}
        $$

        Sendo: 
            - $N_0$: Densidade espectral de ruído.
            - $\mathbb{E}\!\left[ |s(t)|^2 \right]$: Potência média do sinal amostrado.
            - $R_b$: Taxa de bits em bits/s.
            - $Eb/N_0$: Relação $dB$ da energia por bit $E_b$ por densidade espectral de ruído $N_0$ dada na inicialização.

        Args:
            signal (np.ndarray): Sinal transmitido $s(t)$.

        Returns:
            signal (np.ndarray): Sinal recebido $r(t)$, com ruído AWGN adicionado.

        Exemplo:
            ![pageplot](assets/example_noise_gaussian_ebn0.svg)

        <div class="referencia">
          <b>Referência:</b>
          <p>Digital communications / John G. Proakis, Masoud Salehi.—5th ed. (pg. 283)</p>
          <p>https://rwnobrega.page/posts/snr/</p>
        </div>
        """

        self.signal_power = np.mean(np.abs(signal)**2)
        self.bit_energy = self.signal_power / self.Rb
        self.noise_density = self.bit_energy / self.ebn0_lin
        self.variance = (self.noise_density * self.fs) / 2.0
        noise = self.rng.normal(0, np.sqrt(self.variance), len(signal))

        # adiciona ruído
        signal = signal + noise

        # normalização
        signal = signal / np.max(np.abs(signal))

        return signal 

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
    transmitter = Transmitter(datagram, output_print=False, output_plot=False)
    t, s = transmitter.run()

    # ADIÇÃO DE RUIDO USANDO SNR
    snr_db = 15
    add_noise = Noise(snr=snr_db, seed=0)
    s_noisy = add_noise.add_noise(s)

    fig_gauss, grid_gauss = create_figure(1, 1, figsize=(16, 9))
    GaussianNoisePlot(
        fig_gauss, grid_gauss, (0,0),
        variance=add_noise.variance,
        colors="darkorange",
        legend=f"Ruído AWGN SNR - {snr_db} dB",
    ).plot(xlim=(-0.5, 0.5))
    save_figure(fig_gauss, "example_noise_gaussian_snr.pdf")

    # ADIÇÃO DE RUIDO USANDO EBN0
    eb_n0 = 10
    add_noise = NoiseEBN0(ebn0_db=eb_n0, seed=0)
    s_noisy = add_noise.add_noise(s)
    check_ebn0(s, s_noisy, add_noise)

    fig_gauss, grid_gauss = create_figure(1, 1, figsize=(16, 9))
    GaussianNoisePlot(
        fig_gauss, grid_gauss, (0,0),
        variance=add_noise.variance,
        colors="darkorange",
        legend=f"Ruído AWGN Eb/N0 - {eb_n0} dB",
    ).plot(xlim=(-10, 10))
    save_figure(fig_gauss, "example_noise_gaussian_ebn0.pdf")


    fig_time, grid_time = create_figure(2, 1, figsize=(16, 9))

    TimePlot(
        fig_time, grid_time, (0,0),
        t=t,
        signals=[s],
        labels=["$s(t)$"],
        title="Domínio do Tempo - Sem Ruído",
        xlim=(40, 200),
        colors="darkblue",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    TimePlot(
        fig_time, grid_time, (1,0),
        t=t,
        signals=[s_noisy],
        labels=["$s(t) + AWGN$"],
        title="Domínio do Tempo - Com Ruído",
        xlim=(40, 200),
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
        labels=["$S(f)$ + AWGN"],
        title="Domínio da Frequência - Com Ruído",
        xlim=(-8, 8),
        colors="darkred",
        style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
    ).plot()
    
    fig_freq.tight_layout()
    save_figure(fig_freq, "example_noise_freq.pdf")