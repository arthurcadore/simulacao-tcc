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

        
    
    def run(self, s, t, fc=4000):
        self.fc = fc
        self.demodulate(s)
        

if __name__ == "__main__":
    datagram = Datagram(pcdnum=1234, numblocks=1)
    transmitter = Transmitter(datagram, output_print=True)
    t, s = transmitter.run()

    snr_db = 15
    add_noise = Noise(snr=snr_db)
    s_noisy = add_noise.add_noise(s)
    
    receiver = Receiver(output_print=True)
    receiver.run(s_noisy, t)
    