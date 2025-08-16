"""
Implementação de um receptor PTT-A3 com seus componentes.

Autor: Arthur Cadore
Data: 16-08-2025
"""

from datagram import Datagram
from modulator import Modulator
from scrambler import Scrambler
from encoder import Encoder
from plots import Plotter
from transmitter import Transmitter
from noise import Noise
from lowpassfilter import LPF
from matchedfilter import MatchedFilter
from sampler import Sampler
import numpy as np
from convolutional import DecoderViterbi

class Receiver:
    def __init__(self, fs=128_000, Rb=400, output_print=True, output_plot=True):
        r"""
        Classe que encapsula todo o processo de recepção, desde o recebimento do sinal com ruído (sinal do canal), até a recuperação do vetor de bit.

        Args:
            fs (int): Frequência de amostragem em Hz. Default é 128000 Hz.
            Rb (int): Taxa de bits em bps. Default é 400 bps.
            output_print (bool): Se True, imprime os vetores intermediários no console. Default é True.
            output_plot (bool): Se True, gera e salva os gráficos dos processos intermediários. Default é True.
        """
        self.fs = fs
        self.Rb = Rb
        self.output_print = output_print
        self.output_plot = output_plot
        self.plotter = Plotter()

    def demodulate(self, s):
        r"""
        Demodula o sinal $s'(t)$ com ruído recebido, recuperando os sinais $x'_{I}(t)$ e $y'_{Q}(t)$.

        Args:
            s (np.ndarray): Sinal $s'(t)$ a ser demodulado.

        Returns:
            xI_prime (np.ndarray): Sinal $x'_{I}(t)$ demodulado.
            yQ_prime (np.ndarray): Sinal $y'_{Q}(t)$ demodulado.
        """
        demodulator = Modulator(fc=self.fc, fs=self.fs)
        xI_prime, yQ_prime = demodulator.demodulate(s)

        if self.output_print:
            print("\n ==== DEMODULADOR ==== \n")
            print("x'I(t):", ''.join(map(str, xI_prime[:5])),"...")
            print("y'Q(t):", ''.join(map(str, yQ_prime[:5])),"...")
        if self.output_plot:
            self.plotter.plot_freq_receiver(
                xI_prime,
                yQ_prime,
                self.fs,
                self.fc,
                save_path="../out/receiver_freq.pdf"
            )
        return xI_prime, yQ_prime
    
    def lowpassfilter(self, cut_off, xI_prime, yQ_prime, t):
        r"""
        Aplica o filtro passa-baixa com resposta ao impuslo $h(t)$ aos sinais $x'_{I}(t)$ e $y'_{Q}(t)$.

        Args:
            cut_off (float): Frequência de corte do filtro.
            xI_prime (np.ndarray): Sinal $x'_{I}(t)$ a ser filtrado.
            yQ_prime (np.ndarray): Sinal $y'_{Q}(t)$ a ser filtrado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            dI_prime (np.ndarray): Sinal $d'_{I}(t)$ filtrado.
            dQ_prime (np.ndarray): Sinal $d'_{Q}(t)$ filtrado.
        """

        lpf = LPF(cut_off=cut_off, order=6, fs=self.fs, type="butter")
        impulse_response, t_impulse = lpf.calc_impulse_response()
        dI_prime = lpf.apply_filter(xI_prime)
        dQ_prime = lpf.apply_filter(yQ_prime)

        if self.output_print:
            print("\n ==== FILTRAGEM PASSA-BAIXA ==== \n")
            print("d'I(t):", ''.join(map(str, dI_prime[:5])),"...")
            print("d'Q(t):", ''.join(map(str, dQ_prime[:5])),"...")
        
        if self.output_plot:
            self.plotter.plot_lowpass_filter(
                t_impulse,
                impulse_response,
                t,
                dI_prime,
                dQ_prime,
                save_path="../out/receiver_lowpass_filter.pdf"
            )
            self.plotter.plot_lowpass_freq(
                t_impulse,
                impulse_response,
                xI_prime,
                yQ_prime,
                dI_prime,
                dQ_prime,
                self.fs,
                self.fc,
                save_path="../out/receiver_lowpass_freq.pdf"
            )

        return dI_prime, dQ_prime

    def matchedfilter(self, dI_prime, dQ_prime, t):
        r"""
        Aplica o filtro casado com resposta ao impuslo $h(t)$ aos sinais $d'_{I}(t)$ e $d'_{Q}(t)$.

        Args:
            dI_prime (np.ndarray): Sinal $d'_{I}(t)$ a ser filtrado.
            dQ_prime (np.ndarray): Sinal $d'_{Q}(t)$ a ser filtrado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            It_prime (np.ndarray): Sinal $I'(t)$ filtrado.
            Qt_prime (np.ndarray): Sinal $Q'(t)$ filtrado.
        """

        matched_filter = MatchedFilter(alpha=0.8, fs=self.fs, Rb=self.Rb, span=6, type="RRC-Inverted")
        It_prime = matched_filter.apply_filter(dI_prime)
        Qt_prime = matched_filter.apply_filter(dQ_prime)

        if self.output_print:
            print("\n ==== FILTRAGEM CASADA ==== \n")
            print("I'(t):", ''.join(map(str, It_prime[:5])),"...")
            print("Q'(t):", ''.join(map(str, Qt_prime[:5])),"...")

        if self.output_plot:
            self.plotter.plot_matched_filter(
                matched_filter.t_impulse,
                matched_filter.impulse_response,
                t,
                It_prime,
                Qt_prime,
                "Resposta ao Impulso - Filtro Casado",
                "Canal I - Filtro Casado",
                "Canal Q - Filtro Casado",
                "Resposta ao Impulso - Filtro Casado",
                "Canal I - Filtro Casado",
                "Canal Q - Filtro Casado",
                0.1,
                save_path="../out/receiver_matched_filter.pdf"
            )
            self.plotter.plot_matched_filter_freq(
                matched_filter.t_impulse,
                matched_filter.impulse_response,
                dI_prime,
                dQ_prime,
                It_prime,
                Qt_prime,
                self.fs,
                self.fc,
                "Resposta ao Impulso - Filtro Casado",
                "Canal I - Antes do Filtro Casado",
                "Canal Q - Antes do Filtro Casado",
                "Canal I - Depois do Filtro Casado",
                "Canal Q - Depois do Filtro Casado",
                "Resposta ao Impulso - Filtro Casado",
                "Canal I - Antes do Filtro Casado",
                "Canal Q - Antes do Filtro Casado",
                "Canal I - Depois do Filtro Casado",
                "Canal Q - Depois do Filtro Casado",
                save_path="../out/receiver_matched_filter_freq.pdf"
            )
        return It_prime, Qt_prime

    def sampler(self, It_prime, Qt_prime, t):
        r"""
        Realiza a decisão (amostragem e quantização) dos sinais $I'(t)$ e $Q'(t)$.

        Args:
            It_prime (np.ndarray): Sinal $I'(t)$ a ser amostrado e quantizado.
            Qt_prime (np.ndarray): Sinal $Q'(t)$ a ser amostrado e quantizado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            Xnrz_prime (np.ndarray): Sinal $X'_{NRZ}[n]$ amostrado e quantizado.
            Yman_prime (np.ndarray): Sinal $Y'_{MAN}[n]$ amostrado e quantizado.
        """ 
        sampler = Sampler(fs=self.fs, Rb=self.Rb, t=t)
        i_signal_sampled = sampler.sample(It_prime)
        q_signal_sampled = sampler.sample(Qt_prime)
        t_sampled = sampler.sample(t)

        Xnrz_prime = sampler.quantize(i_signal_sampled)
        Yman_prime = sampler.quantize(q_signal_sampled)

        if self.output_print:
            print("\n ==== DECISOR ==== \n")
            print("X'nrz:", ''.join(map(str, Xnrz_prime[:80])),"...")
            print("Y'man:", ''.join(map(str, Yman_prime[:80])),"...")

        if self.output_plot:
            self.plotter.plot_sampled_signals(t,
                                 It_prime,
                                 Qt_prime,
                                 t_sampled,
                                 i_signal_sampled,
                                 q_signal_sampled,                                 
                                 "Amostragem",
                                 "Canal I",
                                 "Canal Q",
                                 "Amostragem",
                                 "Canal I - Amostragem",
                                 "Canal Q - Amostragem",
                                 0.1,
                                 save_path="../out/receiver_sampler.pdf"
            )
        return Xnrz_prime, Yman_prime

    def decode(self, Xnrz_prime, Yman_prime):
        r"""
        Decodifica os sinais quantizados $X'_{NRZ}[n]$ e $Y'_{MAN}[n]$.

        Args:
            Xnrz_prime (np.ndarray): Sinal $X'_{NRZ}[n]$ quantizado.
            Yman_prime (np.ndarray): Sinal $Y'_{MAN}[n]$ quantizado.

        Returns:
            Xn_prime (np.ndarray): Sinal $X'n$ decodificado.
            Yn_prime (np.ndarray): Sinal $Y'n$ decodificado.
        """
        decoderNRZ = Encoder("nrz")
        decoderManchester = Encoder("manchester")
        i_quantized = np.array(Xnrz_prime)
        q_quantized = np.array(Yman_prime)
        
        Xn_prime = decoderNRZ.decode(i_quantized)
        Yn_prime = decoderManchester.decode(q_quantized)

        if self.output_print:
            print("\n ==== DECODIFICADOR DE LINHA ==== \n")
            print("X'n:", ''.join(map(str, Xn_prime)))
            print("Y'n:", ''.join(map(str, Yn_prime)))

        return Xn_prime, Yn_prime

    def remove_preamble(self, Xn_prime, Yn_prime):
        r"""
        Remove os 15 primeiros bits de cada sinal.

        Args:
            Xn_prime (np.ndarray): Sinal $X'n$ decodificado.
            Yn_prime (np.ndarray): Sinal $Y'n$ decodificado.

        Returns:
            Xn_prime (np.ndarray): Sinal $X'n$ sem preâmbulo.
            Yn_prime (np.ndarray): Sinal $Y'n$ sem preâmbulo.
        """
        # remove the first 15 bits from each signal
        Xn_prime = Xn_prime[15:]
        Yn_prime = Yn_prime[15:]

        if self.output_print:
            print("\n ==== REMOÇÃO DO PREÂMBULO ==== \n")
            print("X'n:", ''.join(map(str, Xn_prime)))
            print("Y'n:", ''.join(map(str, Yn_prime)))

        return Xn_prime, Yn_prime

    def descrambler(self, Xn_prime, Yn_prime):
        r"""
        Desembaralha os vetores de bits dos canais I e Q.

        Args:
            Xn_prime (np.ndarray): Vetor de bits $X'n$ embaralhados.
            Yn_prime (np.ndarray): Vetor de bits $Y'n$ embaralhados.

        Returns:
            vt0 (np.ndarray): Vetor de bits $v_{t}^{0}'$ desembaralhado.
            vt1 (np.ndarray): Vetor de bits $v_{t}^{1}'$ desembaralhado.
        """
        descrambler = Scrambler()
        vt0, vt1 = descrambler.descramble(Xn_prime, Yn_prime)

        if self.output_print:
            print("\n ==== DESEMBARALHADOR ==== \n")
            print("vt0':", ''.join(map(str, vt0)))
            print("vt1':", ''.join(map(str, vt1)))

        return vt0, vt1

    def conv_decoder(self, vt0, vt1):
        r"""
        Decodifica os vetores de bits dos canais I e Q.

        Args:
            vt0 (np.ndarray): Vetor de bits $v_{t}^{0}'$ desembaralhado.
            vt1 (np.ndarray): Vetor de bits $v_{t}^{1}'$ desembaralhado.

        Returns:
            ut (np.ndarray): Vetor de bits $u_{t}'$ decodificado.
        """
        conv_decoder = DecoderViterbi()
        ut = conv_decoder.decode(vt0, vt1)

        if self.output_print:
            print("\n ==== DECODIFICADOR VITERBI ==== \n")
            print("u't:", ''.join(map(str, ut)))

        return ut
    
    def run(self, s, t, fc=4000):
        r"""
        Executa o processo de recepção, retornando o resultado da recepção.

        Args:
            s (np.ndarray): Sinal $s(t)$ recebido.
            t (np.ndarray): Vetor de tempo.
            fc (float): Frequência de portadora.

        Returns:
            ut (np.ndarray): Vetor de bits $u_{t}'$ decodificado.
        """
        
        # TODO: Adicionar detecção de portadora;
        self.fc = fc

        xI_prime, yQ_prime = self.demodulate(s)
        dI_prime, dQ_prime= self.lowpassfilter(self.fc*0.6, xI_prime, yQ_prime, t)
        It_prime, Qt_prime = self.matchedfilter(dI_prime, dQ_prime, t)
        Xnrz_prime, Yman_prime = self.sampler(It_prime, Qt_prime, t)
        Xn_prime, Yn_prime = self.decode(Xnrz_prime, Yman_prime)
        Xn_prime, Yn_prime = self.remove_preamble(Xn_prime, Yn_prime)
        vt0, vt1 = self.descrambler(Xn_prime, Yn_prime)
        ut = self.conv_decoder(vt0, vt1)
        return ut 
    


if __name__ == "__main__":
    datagramTX = Datagram(pcdnum=1234, numblocks=1)
    bitsTX = datagramTX.streambits  
    transmitter = Transmitter(datagramTX, output_print=True)
    t, s = transmitter.run()

    snr_db = 0
    add_noise = Noise(snr=snr_db)
    s_noisy = add_noise.add_noise(s)

    print("\n ==== CANAL ==== \n")
    print("s(t):", ''.join(map(str, s_noisy[:5])), "...")
    print("t:   ", ''.join(map(str, t[:5])), "...")

    receiver = Receiver(output_print=True)
    bitsRX = receiver.run(s_noisy, t)

    try:
        datagramRX = Datagram(streambits=bitsRX)
        print("\n",datagramRX.parse_datagram())
    except Exception as e:
        print("Bits TX: ", ''.join(str(b) for b in bitsTX))
        print("Bits RX: ", ''.join(str(b) for b in bitsRX))
        
        # verifica quantos bits tem diferentes entre TX e RX
        # Verifica quantos bits são diferentes entre TX e RX
        num_errors = sum(1 for tx, rx in zip(bitsTX, bitsRX) if tx != rx)
        
        # Calcula a Taxa de Erro de Bit (BER)
        ber = num_errors / len(bitsTX)
        
        print(f"Número de erros: {num_errors}")
        print(f"Taxa de Erro de Bit (BER): {ber:.6f}")
