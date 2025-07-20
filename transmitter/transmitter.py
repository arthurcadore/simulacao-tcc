import os
from formatter import Formatter
from convolutional import ConvolutionalEncoder
from datagram import Datagram
from modulator import Modulator
from preamble import Preamble
from scrambler import Scrambler
from multiplexer import Multiplexer
from encoder import Encoder


class Transmitter: 
    def __init__(self, pcdid, numblocks, fc=4000, fs=128_000): 
        self.pcdid = pcdid
        self.numblocks = numblocks
        self.fc = fc
        self.fs = fs
    
    def transmit(self):
        
        # Datagrama
        datagram = Datagram(self.pcdid, self.numblocks)
        ut = datagram.bits
        print("Datagrama: ", ''.join(str(b) for b in ut))
        output_path = os.path.join("out", "transmitter_datagram.pdf")
        datagram.plot_datagram(output_path)

        # Codificação convolucional
        encoder = ConvolutionalEncoder()
        vt0, vt1 = encoder.encode(ut)
        print("Saída vt0: ", ''.join(str(b) for b in vt0))
        print("Saída vt1: ", ''.join(str(b) for b in vt1))
        output_path = os.path.join("out", "transmitter_encoder.pdf")
        ConvolutionalEncoder.plot_encode(ut, vt0, vt1, output_path)

        # Embaralhamento
        scrambler = Scrambler()
        X, Y = scrambler.scramble(vt0, vt1)
        print("Xn embaralhado: ", ''.join(str(b) for b in X))
        print("Yn embaralhado: ", ''.join(str(b) for b in Y))  
        output_path = os.path.join("out", "transmitter_scrambler.pdf")
        scrambler.plot_scrambler(vt0, vt1, X, Y, output_path)

        # Preambulo
        sI, sQ = Preamble().generate_preamble()
        print("Preamble (S_i): ", ''.join(str(int(b)) for b in sI))
        print("Preamble (S_q): ", ''.join(str(int(b)) for b in sQ))

        # Multiplexação
        multiplexer = Multiplexer()
        Xn, Yn = multiplexer.concatenate(sI, sQ, X, Y)
        print("Xn concatenado: ", ''.join(str(b) for b in Xn))
        print("Yn concatenado: ", ''.join(str(b) for b in Yn))
        output_path = os.path.join("out", "transmitter_multiplexer.pdf")
        multiplexer.plot_concatenation(sI, sQ, X, Y, output_path)

        # Codificação de linha
        Xnrz = Encoder(Xn, "nrz").encode()
        Yman = Encoder(Yn, "manchester").encode()

        print("Xn codificado (NRZ): ", ''.join(str(b) for b in Xnrz))
        print("Yn codificado (MAN): ", ''.join(str(b) for b in Yman))

        # Formatação: 
        formatter = Formatter()
        dI = formatter.format(Xnrz)
        dQ = formatter.format(Yman)

        print("dI(t):", ''.join(str(b) for b in dI[:5]), "...")
        print("dQ(t):", ''.join(str(b) for b in dQ[:5]), "...")
        output_path = os.path.join("out", "transmitter_formatter.pdf")
        formatter.plot_format(dI, dQ, output_path)

        # Modulação
        modulator = Modulator(fc=self.fc, fs=self.fs)
        t, s = modulator.modulate(dI, dQ)
        print("s(t) :", ''.join(str(b) for b in s[:5]), "...")
        output_path = os.path.join("out", "transmitter_modulator.pdf")
        modulator.plot_modulation(dI, dQ, s, self.fs, t_xlim = 0.10, save_path=output_path)

        output_path = os.path.join("out", "transmitter_constellation.pdf")
        modulator.plot_iq(dI, dQ, output_path)

        return t, s

if __name__ == "__main__":

    transmitter = Transmitter(pcdid=1234, numblocks=2)
    t, s = transmitter.transmit()

    

