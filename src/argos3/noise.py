# """
# Implementation of AWGN noise classes. 
#
# Author: Arthur Cadore
# Date: 16-08-2025
# """

import numpy as np
from .datagram import Datagram
from .transmitter import Transmitter
from .plotter import save_figure, create_figure, TimePlot, FrequencyPlot, GaussianNoisePlot

class Noise:
    def __init__(self, snr=15, seed=None, length_multiplier=1, position_factor=0.5):
        r"""
        Implementation of AWGN noise, based on $SNR$.

        Args:
            snr (float): Signal-to-noise ratio in decibels (dB).
            seed (int): Seed of the random number generator.
            length_multiplier (float): Multiplier of the signal length.
            position_factor (float): Position factor of the noise.

        Examples: 
            >>> import argos3
            >>> import numpy as np 
            >>> 
            >>> transmitter = argos3.Transmitter(fc=2400, output_print=False, output_plot=False)
            >>> t, s = transmitter.transmit(argos3.Datagram(pcdnum=1234, numblocks=1))
            >>> 
            >>> noise = argos3.Noise(snr=15, seed=11)
            >>> s_prime = noise.add_noise(s)
            >>>                   
            >>> receiver = argos3.Receiver(fc=2400, output_print=False, output_plot=False)
            >>> datagramRX, success = receiver.receive(s_prime)
            >>> 
            >>> print(success)
            True
            >>> print(datagramRX.parse_datagram())
            {
              "msglength": 1,
              "pcdid": 1234,
              "data": {
                "bloco_1": {
                  "sensor_1": 37,
                  "sensor_2": 198,
                  "sensor_3": 9
                }
              },
              "tail": 7
            }

            - Time Domain Plot Example: ![pageplot](assets/example_noise_time.svg) 
        """
        self.snr = snr
        self.rng = np.random.default_rng(seed)
        self.length_multiplier = length_multiplier
        self.position_factor = np.clip(position_factor, 0, 1)

    
    def add_noise(self, signal):
        r"""
        Adds AWGN noise $n(t)$ to the input signal $s(t)$, based on the $\mathrm{SNR}_{dB}$ defined in initialization. 

        $$
        r(t) = s(t) + n(t), \qquad n(t) \sim \mathcal{N}(0, \sigma^2)
        $$

        Where: 
            - $r(t)$: Signal returned with AWGN noise added.
            - $s(t)$: Input signal without noise. 
            - $n(t)$: Noise added, with normal distribution $\mathcal{N}(0, \sigma^2)$.

        The noise variance $\sigma^2$ is given by:

        $$
        \sigma^2 = \frac{\mathbb{E}\!\left[ |s(t)|^2 \right]}{10^{\frac{\mathrm{SNR}_{dB}}{10}}}
        $$

        Where: 
            - $\sigma^2$: Noise variance.
            - $\mathbb{E}\!\left[ |s(t)|^2 \right]$: Input signal power.
            - $\mathrm{SNR}_{dB}$: Signal-to-noise ratio in decibels (dB).

        Args:
            signal (np.ndarray): Input signal $s(t)$.

        Returns:
            signal (np.ndarray): Signal $r(t)$, with AWGN noise added.

        Examples:
            - Noise Density Plot Example: ![pageplot](assets/example_noise_gaussian_snr.svg)
        """

        self.signal_power = np.mean(np.abs(signal) ** 2)
        self.snr_linear = 10 ** (self.snr / 10)
        self.variance = self.signal_power / self.snr_linear

        sig_len = len(signal)
        noise_len = int(sig_len * self.length_multiplier)

        # generate a noise vector
        noise = self.rng.normal(0, np.sqrt(self.variance), noise_len)

        # calculate the position of the signal
        start_idx = int((noise_len - sig_len) * self.position_factor)
        end_idx = start_idx + sig_len

        # insert the signal into the noise
        noisy_signal = noise.copy()
        noisy_signal[start_idx:end_idx] += signal

        return noisy_signal

class NoiseEBN0:
    def __init__(self, ebn0_db=10, fs=128_000, Rb=400, seed=None, length_multiplier=1, position_factor=0.5):
        r"""
        Implementation of AWGN noise, based on $Eb/N_{0}$.

        Args:
            ebn0_db (float): Target value of $Eb/N_{0}$ in $dB$
            fs (int): Signal sampling rate in $Hz$.
            Rb (int): Bit rate in bits/s.
            seed (int): Seed of the random number generator.
            length_multiplier (float): Multiplier of the signal length.
            position_factor (float): Position factor of the noise.
        
        Examples: 
            >>> import argos3
            >>> import numpy as np 
            >>>
            >>> transmitter = argos3.Transmitter(fc=2400, output_print=False, output_plot=False)
            >>> t, s = transmitter.transmit(argos3.Datagram(pcdnum=1234, numblocks=1))
            >>> 
            >>> noise = argos3.NoiseEBN0(ebn0_db=15, fs=128000, Rb=400, seed=11)
            >>> s_prime = noise.add_noise(s)
            >>>                   
            >>> receiver = argos3.Receiver(fc=2400, output_print=False, output_plot=False)
            >>> datagramRX, success = receiver.receive(s_prime)
            >>> 
            >>> print(success)
            True
            >>> print(datagramRX.parse_datagram())
            {
              "msglength": 1,
              "pcdid": 1234,
              "data": {
                "bloco_1": {
                  "sensor_1": 37,
                  "sensor_2": 198,
                  "sensor_3": 9
                }
              },
              "tail": 7
            }

            - Time Domain Plot Example: ![pageplot](assets/example_noise_time.svg)
        """
        self.ebn0_db = ebn0_db
        self.ebn0_lin = 10 ** (ebn0_db / 10)
        self.fs = fs
        self.Rb = Rb
        self.rng = np.random.default_rng(seed)
        self.length_multiplier = length_multiplier
        self.position_factor = np.clip(position_factor, 0, 1)

    def add_noise(self, signal):
        r"""
        Adds AWGN noise $n(t) to the input signal $s(t), based on the $Eb/N0_{dB}$ defined in initialization. 

        $$
        r(t) = s(t) + n(t), \qquad n(t) \sim \mathcal{N}(0, \sigma^2)
        $$

        Where: 
            - $r(t)$: Signal returned with AWGN noise added.
            - $s(t)$: Input signal without noise. 
            - $n(t)$: Noise added, with normal distribution $\mathcal{N}(0, \sigma^2)$.

        
        The noise variance $\sigma^2$ is given by:

        $$
        \sigma^2 = \frac{N_0 \cdot f_s}{2}
        $$

        Where: 
            - $\sigma^2$: Noise variance.
            - $N_0$: Noise density.
            - $f_s$: Signal sampling rate in $Hz$.

        
        The noise density $N_0$ is given by:

        $$
        N_0 = \frac{\mathbb{E}\!\left[ |s(t)|^2 \right]}{R_b \cdot 10^{\frac{Eb/N_0}{10}}}
        $$

        Where: 
            - $N_0$: Noise density.
            - $\mathbb{E}\!\left[ |s(t)|^2 \right]$: Signal power.
            - $R_b$: Bit rate in bits/s.
            - $Eb/N_0$: Target value of $Eb/N_{0}$ in $dB$.

        Args:
            signal (np.ndarray): Sinal transmitido $s(t)$.

        Returns:
            signal (np.ndarray): Sinal recebido $r(t)$, com ruído AWGN adicionado.

        Examples:
            - Noise Density Plot Example: ![pageplot](assets/example_noise_gaussian_ebn0.svg)

        <div class="referencia">
          <b>Reference:</b>
          <p>Digital communications / John G. Proakis, Masoud Salehi.—5th ed. (pg. 283)</p>
          <p>https://rwnobrega.page/posts/snr/</p>
        </div>
        """

        self.signal_power = np.mean(np.abs(signal)**2)
        self.bit_energy = self.signal_power / self.Rb

        # noise density
        self.noise_density = self.bit_energy / self.ebn0_lin
        self.variance = (self.noise_density * self.fs) / 2.0

        sig_len = len(signal)
        noise_len = int(sig_len * self.length_multiplier)

        # generate a noise vector
        noise = self.rng.normal(0, np.sqrt(self.variance), noise_len)

        # calculate the position of the signal
        start_idx = int((noise_len - sig_len) * self.position_factor)
        end_idx = start_idx + sig_len

        # insert the signal into the noise
        noisy_signal = noise.copy()
        noisy_signal[start_idx:end_idx] += signal

        return noisy_signal

def check_ebn0(s, s_noisy, add_noise:NoiseEBN0):
    n_est = s_noisy - s
    P = np.mean(s**2)
    Eb = P / add_noise.Rb
    # from sigma^2 -> N0 estimated:
    sigma2_meas = np.var(n_est)
    N0_meas = 2 * sigma2_meas / add_noise.fs
    ebn0_meas_db = 10*np.log10(Eb / N0_meas)
    print("Eb/N0 alvo:", add_noise.ebn0_db, "dB | medido:", ebn0_meas_db, "dB")
    

if __name__ == "__main__":
    datagram = Datagram(pcdnum=1234, numblocks=1)
    transmitter = Transmitter(output_print=False, output_plot=False)
    t, s = transmitter.transmit(datagram)

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
        xlim=(-1, 1),
        span=200
    ).plot()
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
        xlim=(-1, 1)
    ).plot()
    save_figure(fig_gauss, "example_noise_gaussian_ebn0.pdf")


    fig_time, grid_time = create_figure(2, 1, figsize=(16, 9))

    TimePlot(
        fig_time, grid_time, (0,0),
        t=t,
        signals=[s],
        labels=["$s(t)$"],
        title="Domínio do Tempo - Sem Ruído",
        xlim=(40, 200),
        amp_norm=True,
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
        amp_norm=True,
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