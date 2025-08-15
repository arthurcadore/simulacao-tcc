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
        self.fs = fs
        self.Rb = Rb
        self.output_print = output_print
        self.output_plot = output_plot
        self.plotter = Plotter()

    def demodulate(self, s):
        demodulator = Modulator(fc=self.fc, fs=self.fs)
        i_signal, q_signal = demodulator.demodulate(s)
        if self.output_print:
            print("i_signal:", ''.join(map(str, i_signal[:20])))
            print("q_signal:", ''.join(map(str, q_signal[:20])))
        if self.output_plot:
            self.plotter.plot_freq_receiver(
                i_signal,
                q_signal,
                self.fs,
                self.fc,
                save_path="../out/receiver_freq.pdf"
            )
        return i_signal, q_signal
    
    def lowpassfilter(self, cut_off, i_signal, q_signal, t):
        lpf = LPF(cut_off=cut_off, order=6, fs=self.fs, type="butter")
        impulse_response, t_impulse = lpf.calc_impulse_response()
        i_signal_filtered = lpf.apply_filter(i_signal)
        q_signal_filtered = lpf.apply_filter(q_signal)
        if self.output_plot:
            self.plotter.plot_lowpass_filter(
                t_impulse,
                impulse_response,
                t,
                i_signal_filtered,
                q_signal_filtered,
                save_path="../out/receiver_lowpass_filter.pdf"
            )
            self.plotter.plot_lowpass_freq(
                t_impulse,
                impulse_response,
                i_signal,
                q_signal,
                i_signal_filtered,
                q_signal_filtered,
                self.fs,
                self.fc,
                save_path="../out/receiver_lowpass_freq.pdf"
            )

        return i_signal_filtered, q_signal_filtered

    def matchedfilter(self, i_signal, q_signal, t):
        matched_filter = MatchedFilter(alpha=0.8, fs=self.fs, Rb=self.Rb, span=6, type="RRC-Inverted")

        i_signal_filtered = matched_filter.apply_filter(i_signal)
        q_signal_filtered = matched_filter.apply_filter(q_signal)
        
        if self.output_plot:
            self.plotter.plot_matched_filter(
                matched_filter.t_impulse,
                matched_filter.impulse_response,
                t,
                i_signal_filtered,
                q_signal_filtered,
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
                i_signal,
                q_signal,
                i_signal_filtered,
                q_signal_filtered,
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
        return i_signal_filtered, q_signal_filtered

    def sampler(self, i_signal, q_signal, t):
        sampler = Sampler(fs=self.fs, Rb=self.Rb, t=t)
        i_signal_sampled = sampler.sample(i_signal)
        q_signal_sampled = sampler.sample(q_signal)
        t_sampled = sampler.sample(t)

        i_signal_quantized = sampler.quantize(i_signal_sampled)
        q_signal_quantized = sampler.quantize(q_signal_sampled)
        
        if self.output_plot:
            self.plotter.plot_sampled_signals(t,
                                 i_signal,
                                 q_signal,
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
        return i_signal_quantized, q_signal_quantized

    def decode(self, i_signal_quantized, q_signal_quantized):
        decoderNRZ = Encoder("nrz")
        decoderManchester = Encoder("manchester")
        i_quantized = np.array(i_signal_quantized)
        q_quantized = np.array(q_signal_quantized)
        
        i_signal_decoded = decoderNRZ.decode(i_quantized)
        q_signal_decoded = decoderManchester.decode(q_quantized)
        return i_signal_decoded, q_signal_decoded

    def remove_preamble(self, i_signal_decoded, q_signal_decoded):
        # remove the first 15 bits from each signal
        i = i_signal_decoded[15:]
        q = q_signal_decoded[15:]
        return i, q

    def descrambler(self, X, Y):
        descrambler = Scrambler()
        vt0, vt1 = descrambler.descramble(X, Y)
        return vt0, vt1

    def conv_decoder(self, vt0, vt1):
        conv_decoder = DecoderViterbi()
        u = conv_decoder.decode(vt0, vt1)
        return u
    
    def run(self, s, t, fc=4000):
        
        # TODO: Adicionar detecção de portadora;
        self.fc = fc

        i, q = self.demodulate(s)
        i_filt, q_filt= self.lowpassfilter(self.fc*0.6, i, q, t)
        i_filt, q_filt = self.matchedfilter(i_filt, q_filt, t)
        i_quantized, q_quantized = self.sampler(i_filt, q_filt, t)
        i_decoded, q_decoded = self.decode(i_quantized, q_quantized)
        i, q = self.remove_preamble(i_decoded, q_decoded)
        i, q = self.descrambler(i, q)
        u = self.conv_decoder(i, q)
        return u 
    


if __name__ == "__main__":
    datagramTX = Datagram(pcdnum=1234, numblocks=8)
    bitsTX = datagramTX.streambits  
    transmitter = Transmitter(datagramTX, output_print=True)
    t, s = transmitter.run()

    snr_db = 0
    add_noise = Noise(snr=snr_db)
    s_noisy = add_noise.add_noise(s)
    
    receiver = Receiver(output_print=True)
    bitsRX = receiver.run(s_noisy, t)

    try:
        datagramRX = Datagram(streambits=bitsRX)
        print(datagramRX.parse_datagram())
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
