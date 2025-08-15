from formatter import Formatter
from convolutional import EncoderConvolutional
from datagram import Datagram
from modulator import Modulator
from preamble import Preamble
from scrambler import Scrambler
from multiplexer import Multiplexer
from encoder import Encoder
from plots import Plotter
from transmitter import Transmitter
from noise import Noise
from lowpassfilter import LPF
from matchedfilter import MatchedFilter

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
    
    def run(self, s, t, fc=4000):
        
        # TODO: Adicionar detecção de portadora;
        self.fc = fc

        i, q = self.demodulate(s)
        i_filt, q_filt= self.lowpassfilter(self.fc*0.6, i, q, t)
        i_filt, q_filt = self.matchedfilter(i_filt, q_filt, t)
        

if __name__ == "__main__":
    datagram = Datagram(pcdnum=1234, numblocks=1)
    transmitter = Transmitter(datagram, output_print=True)
    t, s = transmitter.run()

    snr_db = 15
    add_noise = Noise(snr=snr_db)
    s_noisy = add_noise.add_noise(s)
    
    receiver = Receiver(output_print=True)
    receiver.run(s_noisy, t)
    