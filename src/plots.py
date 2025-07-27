import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import __main__


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

class Plotter:
    def __init__(self):
        pass

    def time_domain(self, s1, s2, t, label1, label2, title1, title2, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        ax1.plot(t, s1, label=label1, color='blue')
        ax1.set_title(title1)
        ax1.set_xlim(0, 0.1)
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        leg1 = ax1.legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)

        ax2.plot(t, s2, label=label2, color='red')
        title = f'{title2}'
        ax2.set_title(title)
        ax2.set_xlim(0, 0.1)
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        leg2 = ax2.legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg2.get_frame().set_facecolor('white')
        leg2.get_frame().set_edgecolor('black')
        leg2.get_frame().set_alpha(1.0)

        self._save_or_show(fig, save_path)

    def frequency_domain(self, signal, signal_noise, fs, fc, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1/fs))

        fft_clean = np.fft.fftshift(np.fft.fft(signal))
        fft_clean_db = mag2db(fft_clean)
        ax1.plot(freqs, fft_clean_db, color='blue')
        ax1.set_title("Domínio da Frequência - Sem Ruído")
        ax1.set_ylim(-80, 5)
        ax1.set_xlim(-2.5 * fc, 2.5 * fc)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True)

        fft_noisy = np.fft.fftshift(np.fft.fft(signal_noise))
        fft_noisy_db = mag2db(fft_noisy)
        ax2.plot(freqs, fft_noisy_db, color='red')
        title = "Domínio da Frequência - Com Ruído"
        ax2.set_title(title)
        ax2.set_ylim(-80, 5)
        ax2.set_xlim(-2.5 * fc, 2.5 * fc)
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_xlabel("Frequência (Hz)")
        ax2.grid(True)

        self._save_or_show(fig, save_path)

    def _save_or_show(self, fig, path):
        if path:
            # Caminho baseado no diretório do script principal
            base_dir = os.path.dirname(os.path.abspath(__main__.__file__))
            full_path = os.path.join(base_dir, path)

            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            fig.savefig(full_path)
        else:
            plt.show()