import numpy as np
from plots import Plotter
from transmitter import Transmitter

class AddNoise:
    def __init__(self, snr=10):
        self.snr = snr
    
    def add_noise(self, signal):
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (self.snr / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise

if __name__ == "__main__":
    snr_db = 15
    add_noise = AddNoise(snr=snr_db)
    plotter = Plotter()

    transmitter = Transmitter(pcdid=1234, numblocks=2, output_print=False)
    t, s = transmitter.transmit()

    s_noisy = add_noise.add_noise(s)

    plotter.time_domain(s, 
                        s_noisy, 
                        t, 
                        r'$s(t)$', 
                        r'$s(t) + AWGN$', 
                        "Domínio do Tempo - Sem Ruído", 
                        f"Domínio do Tempo - Com Ruído (SNR = {snr_db} dB)", 
                        save_path="../out/receiver_add_noise_time.pdf"
    )
    plotter.frequency_domain(s,
                             s_noisy,
                             transmitter.fs,
                             transmitter.fc,
                             r'$S(f)$',
                             r'$S(f) + AWGN$',
                             "Domínio da Frequência - Sem Ruído",
                             "Domínio da Frequência - Com Ruído",
                             save_path="../out/receiver_add_noise_frequency.pdf"
    )
    
