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
    def plot_add_noise(signal, signal_noise, t, fs, fc, save_path=None):
        # 20. Plot 2x2 - Tempo e Frequência
        fig_tf = plt.figure(figsize=(16, 10))
        gs_tf = gridspec.GridSpec(2, 2)

        # Tempo - Sem ruído
        ax1 = fig_tf.add_subplot(gs_tf[0, 0])
        ax1.plot(t, signal, label='Sinal Sem Ruído', color='blue')
        ax1.set_title('Tempo - Sem Ruído')
        ax1.set_xlim(0, 0.01)
        ax1.set_xlabel('Tempo (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax1.legend()
        leg1 = ax1.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)

        # Tempo - Com ruído
        ax2 = fig_tf.add_subplot(gs_tf[0, 1])
        ax2.plot(t, signal_noise, label='Sinal com AWGN (10 dB)', color='red')
        ax2.set_title('Tempo - Com Ruído (SNR = 10 dB)')
        ax2.set_xlim(0, 0.01)
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        ax2.legend()
        leg2 = ax2.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg2.get_frame().set_facecolor('white')
        leg2.get_frame().set_edgecolor('black')
        leg2.get_frame().set_alpha(1.0)

        # Frequência - Sem ruído
        ax3 = fig_tf.add_subplot(gs_tf[1, 0])
        fft_clean = np.fft.fftshift(np.fft.fft(signal))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1/fs))
        fft_clean_db = mag2db(fft_clean)
        ax3.plot(freqs, fft_clean_db, color='blue')
        ax3.set_ylim(-60, 5)
        ax3.set_xlim(-2.5 * fc, 2.5 * fc)
        ax3.set_ylabel("Magnitude (dB)")
        ax3.grid(True)

        # Frequência - Com ruído
        ax4 = fig_tf.add_subplot(gs_tf[1, 1])
        fft_noisy = np.abs(np.fft.fftshift(np.fft.fft(signal_noise)))
        fft_noisy = np.fft.fftshift(np.fft.fft(signal_noise))
        fft_noisy_db = mag2db(fft_noisy)
        ax4.plot(freqs, fft_noisy_db, color='red')
        ax4.set_ylim(-60, 5)
        ax4.set_xlim(-2.5 * fc, 2.5 * fc)
        ax4.set_ylabel("Magnitude (dB)")
        ax4.grid(True)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)

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

    output_path = os.path.join("out", "receiver_add_noise.pdf")
    add_noise.plot_add_noise(s, s_noisy, t, transmitter.fs, transmitter.fc, save_path=output_path)
    
