import numpy as np
from plots import Plotter
from datagram import Datagram
from transmitter import Transmitter

class Noise:
    r"""
    Adiciona ruído branco gaussiano (AWGN) a um sinal.

    Args:
        snr (float): Relação sinal-ruído em decibéis (dB). Padrão é 10 dB.
    """
    def __init__(self, snr=10):
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

if __name__ == "__main__":
    datagram = Datagram(pcdnum=1234, numblocks=1)
    transmitter = Transmitter(datagram, output_print=False)
    t, s = transmitter.run()

    snr_db = 15
    add_noise = Noise(snr=snr_db)
    s_noisy = add_noise.add_noise(s)

    plotter = Plotter()
    plotter.plot_time_domain(s, 
                        s_noisy, 
                        t, 
                        r'$s(t)$', 
                        r'$s(t) + AWGN$', 
                        "Domínio do Tempo - Sem Ruído", 
                        f"Domínio do Tempo - Com Ruído (SNR = {snr_db} dB)", 
                        save_path="../out/example_addnoise_time.pdf"
    )
    plotter.plot_frequency_domain(s,
                             s_noisy,
                             transmitter.fs,
                             transmitter.fc,
                             r'$S(f)$',
                             r'$S(f) + AWGN$',
                             "Domínio da Frequência - Sem Ruído",
                             "Domínio da Frequência - Com Ruído",
                             save_path="../out/example_addnoise_frequency.pdf"
    )
    
