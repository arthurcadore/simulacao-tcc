import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from collections import defaultdict
from transmitter import Transmitter
import matplotlib.gridspec as gridspec

# Estilo science
plt.style.use('science')
plt.rcParams["figure.figsize"] = (16, 9)
plt.rc('font', size=16)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=22)
plt.rc('legend', frameon=True, edgecolor='black', facecolor='white', fancybox=True, fontsize=12)

def mag2db(signal):
    mag = np.abs(signal)
    mag /= np.max(mag)
    return 20 * np.log10(mag + 1e-12)

class AddNoise:
    def __init__(self, snr=10):
        self.snr = snr
    
    def add_noise(self, signal):
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (self.snr / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise

    @staticmethod
    def plot_time_domain(signal, signal_noise, t, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

        # Tempo - Sem ruído
        ax1.plot(t, signal, label='Sinal Sem Ruído', color='blue')
        ax1.set_title('Domínio do Tempo - Sem Ruído')
        ax1.set_xlim(0, 0.05)
        ax1.set_xlabel('Tempo (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax1.legend()

        # Tempo - Com ruído
        ax2.plot(t, signal_noise, label='Sinal com AWGN', color='red')
        ax2.set_title('Domínio do Tempo - Com Ruído (SNR = 10 dB)')
        ax2.set_xlim(0, 0.05)
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def plot_frequency_domain(signal, signal_noise, fs, fc, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

        # Eixo de frequência
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1/fs))

        # FFT sem ruído
        fft_clean = np.fft.fftshift(np.fft.fft(signal))
        fft_clean_db = mag2db(fft_clean)
        ax1.plot(freqs, fft_clean_db, color='blue')
        ax1.set_title("Domínio da Frequência - Sem Ruído")
        ax1.set_ylim(-80, 5)
        ax1.set_xlim(-2.5 * fc, 2.5 * fc)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True)

        # FFT com ruído
        fft_noisy = np.fft.fftshift(np.fft.fft(signal_noise))
        fft_noisy_db = mag2db(fft_noisy)
        ax2.plot(freqs, fft_noisy_db, color='red')
        ax2.set_title("Domínio da Frequência - Com Ruído (SNR = 10 dB)")
        ax2.set_ylim(-80, 5)
        ax2.set_xlim(-2.5 * fc, 2.5 * fc)
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_xlabel("Frequência (Hz)")
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()
        
if __name__ == "__main__":
    
    add_noise = AddNoise(snr=10)
    transmitter = Transmitter(pcdid=1234, numblocks=2, output_print=False)
    t, s = transmitter.transmit()
    
    s_noisy = add_noise.add_noise(s)

    # output_path = os.path.join("out", "receiver_add_noise.pdf")
    # add_noise.plot_add_noise(s, s_noisy, t, transmitter.fs, transmitter.fc, save_path=output_path)
    
    output_path = os.path.join("out", "receiver_add_noise_time.pdf")
    add_noise.plot_time_domain(s, s_noisy, t, save_path=output_path)
    
    output_path = os.path.join("out", "receiver_add_noise_frequency.pdf")
    add_noise.plot_frequency_domain(s, s_noisy, transmitter.fs, transmitter.fc, save_path=output_path)
    
