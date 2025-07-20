import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from formatter import Formatter
from convolutional import ConvolutionalEncoder
from datagram import Datagram
from modulator import Modulator
from preamble import Preamble
from scrambler import Scrambler
from multiplexer import Multiplexer
from encoder import Encoder

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

class Transmitter: 
    def __init__(self, pcdid, numblocks, fc=4000, fs=128_000): 
        self.pcdid = pcdid
        self.numblocks = numblocks
        self.fc = fc
        self.fs = fs
    
    def transmit(self):
        
        datagram = Datagram(self.pcdid, self.numblocks).bits
        print("Datagrama: ", ''.join(str(b) for b in datagram))

        # Codificação convolucional
        encoder = ConvolutionalEncoder()
        vt0, vt1 = encoder.encode(datagram)
        print("Saída vt0: ", ''.join(str(b) for b in vt0))
        print("Saída vt1: ", ''.join(str(b) for b in vt1))

        # Embaralhamento
        scrambler = Scrambler()
        Xn, Yn = scrambler.scramble(vt0, vt1)
        print("Xn embaralhado: ", ''.join(str(b) for b in Xn))
        print("Yn embaralhado: ", ''.join(str(b) for b in Yn))  

        # Preambulo
        sI, sQ = Preamble().generate_preamble()
        print("Preamble (S_i): ", ''.join(str(int(b)) for b in sI))
        print("Preamble (S_q): ", ''.join(str(int(b)) for b in sQ))

        # Multiplexação
        multiplexer = Multiplexer()
        Xn, Yn = multiplexer.concatenate(sI, sQ, Xn, Yn)
        print("Xn concatenado: ", ''.join(str(b) for b in Xn))
        print("Yn concatenado: ", ''.join(str(b) for b in Yn))

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

        # Modulação
        modulator = Modulator(fc=self.fc, fs=self.fs)
        t, s = modulator.modulate(dI, dQ)
        print("s(t) :", ''.join(str(b) for b in s[:5]), "...")

        return t, s

if __name__ == "__main__":

    transmitter = Transmitter(pcdid=1234, numblocks=1)
    t, s = transmitter.transmit()

    

