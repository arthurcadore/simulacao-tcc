from formatter import Formatter
from convolutional import EncoderConvolutional
from datagram import Datagram
from modulator import Modulator
from preamble import Preamble
from scrambler import Scrambler
from multiplexer import Multiplexer
from encoder import Encoder
from plots import Plotter


class Transmitter: 
    def __init__(self, pcdid, numblocks, fc=4000, fs=128_000, Rb=400, output_print=True, output_plot=True): 
        self.pcdid = pcdid
        self.numblocks = numblocks
        self.fc = fc
        self.fs = fs
        self.Rb = Rb
        self.output_print = output_print
        self.output_plot = output_plot
    
    def transmit(self):

        plotter = Plotter()

        # Datagrama
        datagram = Datagram(self.pcdid, self.numblocks)
        ut = datagram.streambits

        if self.output_print:
            print(datagram.parse_datagram())
        
        if self.output_plot:
            plotter.plot_bits([datagram.msglength, 
                               datagram.pcdid, 
                               datagram.blocks, 
                               datagram.tail],
                               sections=[("Message Length", len(datagram.msglength)),
                                         ("PCD ID", len(datagram.pcdid)),
                                         ("Dados de App.", len(datagram.blocks)),
                                         ("Tail", len(datagram.tail))],
                               colors=["green", "orange", "red", "blue"],
                               save_path="../out/transmitter_datagram.pdf"
            )

        # Codificação convolucional
        encoder = EncoderConvolutional()
        vt0, vt1 = encoder.encode(ut)

        if self.output_print:
            print("vt0: ", ''.join(str(b) for b in vt0))
            print("vt1: ", ''.join(str(b) for b in vt1))

        if self.output_plot:
            plotter.plot_conv(ut, 
                              vt0, 
                              vt1, 
                              "Entrada $u_t$", 
                              "Canal I $v_t^{(0)}$", 
                              "Canal Q $v_t^{(1)}$", 
                              "$u_t$", 
                              "$v_t^{(0)}$", 
                              "$v_t^{(1)}$", 
                              save_path="../out/transmitter_convolutional.pdf"
            )

        # Embaralhamento
        scrambler = Scrambler()
        X, Y = scrambler.scramble(vt0, vt1)
        
        if self.output_print:
            print("X: ", ''.join(str(b) for b in X))
            print("Y: ", ''.join(str(b) for b in Y))
    
        # Preambulo
        sI, sQ = Preamble().generate_preamble()

        if self.output_print:
            print("sI: ", ''.join(str(b) for b in sI))
            print("sQ: ", ''.join(str(b) for b in sQ))

        if self.output_plot:
            plotter.plot_preamble(sI,
                                  sQ,
                                  r"$S_i$",
                                  r"$S_q$",
                                  r"Canal $I$",
                                  r"Canal $Q$",
                                  save_path="../out/transmitter_preamble.pdf"
            )

        # Multiplexação
        multiplexer = Multiplexer()
        Xn, Yn = multiplexer.concatenate(sI, sQ, X, Y)

        if self.output_print:
            print("Xn: ", ''.join(str(b) for b in Xn))
            print("Yn: ", ''.join(str(b) for b in Yn))

        if self.output_plot:
            plotter.plot_mux(sI,
                             sQ,
                             X,
                             Y,
                             "Preambulo $S_I$",
                             "Canal I $(X_n)$",
                             "Preambulo $S_Q$",
                             "Canal Q $(Y_n)$",
                             "$X_n$",
                             "$Y_n$",
                             save_path="../out/transmitter_multiplexing.pdf"
            )



        # Codificação
        Xnrz = Encoder(Xn, "nrz").encode()
        Yman = Encoder(Yn, "manchester").encode()

        if self.output_print:
            print("Xnrz: ", ''.join(str(b) for b in Xnrz))
            print("Yman: ", ''.join(str(b) for b in Yman))

        if self.output_plot:
            plotter.plot_encode(Xn,
                                Yn,
                                Xnrz,
                                Yman,
                                "Canal I $(X_n)$",
                                "Canal Q $(Y_n)$",
                                "Canal I $(X_{NRZ}[n])$",
                                "Canal Q $(Y_{MAN}[n])$",
                                "$X_n$",
                                "$Y_n$",
                                "$X_{NRZ}[n]$",
                                "$Y_{MAN}[n]$",
                                save_path="../out/transmitter_encode.pdf"
            )

        # Formatação: 
        formatter = Formatter()
        dI = formatter.format(Xnrz)
        dQ = formatter.format(Yman)

        if self.output_print:
            print("dI: ", ''.join(str(b) for b in dI[:20]))
            print("dQ: ", ''.join(str(b) for b in dQ[:20]))

        if self.output_plot:
            plotter.plot_filter(formatter.g,
                                formatter.t_rc,
                                formatter.Tb,
                                formatter.span,
                                formatter.fs,
                                dI,
                                dQ,
                                fr'Pulso RRC ($\alpha={formatter.alpha}$)',
                                fr'$d_I(t)$',
                                fr'$d_Q(t)$',
                                'Pulso Root Raised Cosine (RRC)',
                                fr'Sinal $d_I(t)$',
                                fr'Sinal $d_Q(t)$',
                                0.05,
                                save_path="../out/transmitter_filter.pdf"
            )

        # Modulação
        modulator = Modulator(fc=self.fc, fs=self.fs)
        t, s = modulator.modulate(dI, dQ)

        if self.output_print:
            print("s: ", ''.join(str(b) for b in s[:20]))
            print("t: ", t[:20])

        if self.output_plot:
            plotter.plot_modulation_time(dI, 
                                      dQ, 
                                      s, 
                                      "dI(t)", 
                                      "dQ(t)",
                                      "s(t)",
                                      "Sinal $IQ$ - Formatados RRC",
                                      "Sinal Modulado $IQ$",
                                      fs=self.fs, 
                                      t_xlim=0.10, 
                                      save_path="../out/transmitter_modulator_time.pdf"
            )
            plotter.plot_modulation_freq(dI, 
                                        dQ, 
                                        s,
                                        "$D_I'(f)$",
                                        "$D_Q'(f)$",
                                        "$S(f)$",
                                        "Sinal Banda Base - Componente $I$",
                                        "Sinal Banda Base - Componente $Q$",
                                        "Sinal Modulado $IQ$",
                                        fs=self.fs, 
                                        fc=self.fc, 
                                        save_path="../out/transmitter_modulator_freq.pdf"
            )
            plotter.plot_modulation_iq(dI, 
                                       dQ, 
                                       fr'Amostras $IQ$',
                                       fr'Simbolos $QPSK$',
                                       fr'Plano $IQ$ (Scatter)',
                                       fr'Plano $IQ$ (Constelação)',
                                       save_path="../out/transmitter_modulator_iq.pdf"
            )

        return t, s

if __name__ == "__main__":

    transmitter = Transmitter(pcdid=1234, numblocks=1, output_print=True)
    t, s = transmitter.transmit()

    

