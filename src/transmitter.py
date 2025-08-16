"""
Implementação de um transmissor PTT-A3 com seus componentes.

Autor: Arthur Cadore
Data: 16-08-2025
"""

from formatter import Formatter
from convolutional import EncoderConvolutional
from datagram import Datagram
from modulator import Modulator
from preamble import Preamble
from scrambler import Scrambler
from multiplexer import Multiplexer
from encoder import Encoder
from plots import Plotter

# TODO: Implementar salvamento dos dados em formato binário para diminuir o tamanho do arquivo e facilitar a leitura posterior.
class TransmissionResult:
    r"""
    Classe para armazenar o resultado da transmissão, incluindo o sinal modulado e o vetor de tempo.

    Args:
        t (np.ndarray): Vetor de tempo.
        s (np.ndarray): Sinal modulado.
    """
    def __init__(self, t, s):
        self.time = t
        self.signal = s

    def save(self, path_prefix):
        with open(f"{path_prefix}_signal.txt", "w") as f:
            f.write(" ".join(map(str, self.signal)))
        with open(f"{path_prefix}_time.txt", "w") as f:
            f.write(" ".join(map(str, self.time)))


class Transmitter:
    def __init__(self, datagram: Datagram, fc=4000, fs=128_000, Rb=400, 
                 output_print=True, output_plot=True):
        r"""
        Classe que encapsula todo o processo de transmissão, desde a preparação do datagrama até a
        modulação do sinal.
    
        Args:
            datagram (Datagram): Instância do datagrama a ser transmitido.
            fc (float): Frequência da portadora em Hz. Default é 4000 Hz
            fs (float): Frequência de amostragem em Hz. Default é 128000 Hz.
            Rb (float): Taxa de bits em bps. Default é 400 b
            output_print (bool): Se True, imprime os vetores intermediários no console. Default é True.
            output_plot (bool): Se True, gera e salva os gráficos dos processos intermediários.
        """
        self.datagram = datagram
        self.fc = fc
        self.fs = fs
        self.Rb = Rb
        self.output_print = output_print
        self.output_plot = output_plot
        self.plotter = Plotter()

    def prepare_datagram(self):
        r"""
        Prepara o datagrama para transmissão, retornando o vetor de bits $u_t$.

        Returns:
            ut (np.ndarray): Vetor de bits do datagrama.
        """
        ut = self.datagram.streambits
        if self.output_print:
            print("\n ==== MONTAGEM DATAGRAMA ==== \n")
            print(self.datagram.parse_datagram())
            print("\nut:", ''.join(map(str, ut)))
        if self.output_plot:
            self.plotter.plot_bits(
                [self.datagram.msglength, self.datagram.pcdid, self.datagram.blocks, self.datagram.tail],
                sections=[
                    ("Message Length", len(self.datagram.msglength)),
                    ("PCD ID", len(self.datagram.pcdid)),
                    ("Dados de App.", len(self.datagram.blocks)),
                    ("Tail", len(self.datagram.tail))
                ],
                colors=["green", "orange", "red", "blue"],
                save_path="../out/transmitter_datagram.pdf"
            )
        return ut

    def encode_convolutional(self, ut):
        r"""
        Codifica o vetor de bits $u_t$ usando codificação convolucional.

        Args:
            ut (np.ndarray): Vetor de bits a ser codificado.

        Returns:
            vt0 (np.ndarray): Saída do canal I.
            vt1 (np.ndarray): Saída do canal Q.
        """
        encoder = EncoderConvolutional()
        vt0, vt1 = encoder.encode(ut)
        if self.output_print:
            print("\n ==== CODIFICADOR CONVOLUCIONAL ==== \n")
            print("vt0:", ''.join(map(str, vt0)))
            print("vt1:", ''.join(map(str, vt1)))
        if self.output_plot:
            self.plotter.plot_conv(
                ut, vt0, vt1, "Entrada $u_t$",
                "Canal I $v_t^{(0)}$", "Canal Q $v_t^{(1)}$",
                "$u_t$", "$v_t^{(0)}$", "$v_t^{(1)}$",
                save_path="../out/transmitter_convolutional.pdf"
            )
        return vt0, vt1

    def scramble(self, vt0, vt1):
        r"""
        Embaralha os vetores de bits dos canais I e Q.

        Args:
            vt0 (np.ndarray): Vetor de bits do canal I.
            vt1 (np.ndarray): Vetor de bits do canal Q.

        Returns:
            Xn (np.ndarray): Vetor embaralhado do canal I.
            Yn (np.ndarray): Vetor embaralhado do canal Q.
        """
        scrambler = Scrambler()
        X, Y = scrambler.scramble(vt0, vt1)
        if self.output_print:
            print("\n ==== EMBARALHADOR ==== \n")
            print("Xn:", ''.join(map(str, X)))
            print("Yn:", ''.join(map(str, Y)))
        return X, Y

    def generate_preamble(self):
        r"""
        Gera os vetores de preâmbulo $S_I$ e $S_Q$.

        Returns:
            sI (np.ndarray): Vetor do preâmbulo do canal I.
            sQ (np.ndarray): Vetor do preâmbulo do canal Q.
        """
        sI, sQ = Preamble().generate_preamble()
        if self.output_print:
            print("\n ==== MONTAGEM PREAMBULO ==== \n")
            print("sI:", ''.join(map(str, sI)))
            print("sQ:", ''.join(map(str, sQ)))
        if self.output_plot:
            self.plotter.plot_preamble(
                sI, sQ, r"$S_i$", r"$S_q$",
                r"Canal $I$", r"Canal $Q$",
                save_path="../out/transmitter_preamble.pdf"
            )
        return sI, sQ

    def multiplex(self, sI, sQ, X, Y):
        r"""
        Multiplexa os vetores de preâmbulo e dados dos canais I e Q.

        Args:
            sI (np.ndarray): Vetor do preâmbulo do canal I.
            sQ (np.ndarray): Vetor do preâmbulo do canal Q.
            X (np.ndarray): Vetor de dados do canal I.
            Y (np.ndarray): Vetor de dados do canal Q.
        
        Returns:
            Xn (np.ndarray): Vetor multiplexado do canal I.
            Yn (np.ndarray): Vetor multiplexado do canal Q.
        """

        multiplexer = Multiplexer()
        Xn, Yn = multiplexer.concatenate(sI, sQ, X, Y)
        if self.output_print:
            print("\n ==== MULTIPLEXADOR ==== \n")
            print("Xn:", ''.join(map(str, Xn)))
            print("Yn:", ''.join(map(str, Yn)))
        if self.output_plot:
            self.plotter.plot_mux(
                sI, sQ, X, Y,
                "Preambulo $S_I$", "Canal I $(X_n)$",
                "Preambulo $S_Q$", "Canal Q $(Y_n)$",
                "$X_n$", "$Y_n$",
                save_path="../out/transmitter_multiplexing.pdf"
            )
        return Xn, Yn

    def encode_channels(self, Xn, Yn):
        r"""
        Codifica os vetores dos canais I e Q usando NRZ e Manchester, respectivamente.

        Args:
            Xn (np.ndarray): Vetor do canal I a ser codificado.
            Yn (np.ndarray): Vetor do canal Q a ser codificado.
        
        Returns:
            Xnrz (np.ndarray): Vetor codificado do canal I (NRZ).
            Yman (np.ndarray): Vetor codificado do canal Q (Manchester).
        """

        encoderNRZ = Encoder("nrz")
        encoderManchester = Encoder("manchester")
        Xnrz = encoderNRZ.encode(Xn)
        Yman = encoderManchester.encode(Yn)
        if self.output_print:
            print("\n ==== CODIFICAÇÃO DE LINHA ==== \n")
            print("Xnrz:", ''.join(map(str, Xnrz[:80])),"...")
            print("Yman:", ''.join(map(str, Yman[:80])),"...")
        if self.output_plot:
            self.plotter.plot_encode(
                Xn, Yn, Xnrz, Yman,
                "Canal I $(X_n)$", "Canal Q $(Y_n)$",
                "Canal I $(X_{NRZ}[n])$", "Canal Q $(Y_{MAN}[n])$",
                "$X_n$", "$Y_n$", "$X_{NRZ}[n]$", "$Y_{MAN}[n]$",
                save_path="../out/transmitter_encode.pdf"
            )
        return Xnrz, Yman

    def format_signals(self, Xnrz, Yman):
        r"""
        Formata os vetores dos canais I e Q usando filtro RRC.

        Args:
            Xnrz (np.ndarray): Vetor do canal I a ser formatado.
            Yman (np.ndarray): Vetor do canal Q a ser formatado.
        
        Returns:
            dI (np.ndarray): Vetor formatado do canal I.
            dQ (np.ndarray): Vetor formatado do canal Q.
        """
        formatter = Formatter()
        dI = formatter.apply_format(Xnrz)
        dQ = formatter.apply_format(Yman)
        if self.output_print:
            print("\n ==== FORMATADOR ==== \n")
            print("dI:", ''.join(map(str, dI[:5])),"...")
            print("dQ:", ''.join(map(str, dQ[:5])),"...")
        if self.output_plot:
            self.plotter.plot_filter(
                formatter.g, formatter.t_rc, formatter.Tb,
                formatter.span, formatter.fs, dI, dQ,
                fr'Pulso RRC ($\alpha={formatter.alpha}$)',
                fr'$d_I(t)$', fr'$d_Q(t)$',
                'Pulso Root Raised Cosine (RRC)',
                fr'Sinal $d_I(t)$', fr'Sinal $d_Q(t)$',
                0.05, save_path="../out/transmitter_filter.pdf"
            )
        return dI, dQ

    def modulate(self, dI, dQ):
        r"""
        Modula os vetores formatados dos canais I e Q usando modulação QPSK.

        Args:
            dI (np.ndarray): Vetor formatado do canal I.
            dQ (np.ndarray): Vetor formatado do canal Q.
        
        Returns:

            t (np.ndarray): Vetor de tempo.
            s (np.ndarray): Sinal modulado.
        """
        modulator = Modulator(fc=self.fc, fs=self.fs)
        t, s = modulator.modulate(dI, dQ)
        if self.output_print:
            print("\n ==== MODULADOR ==== \n")
            print("s(t):", ''.join(map(str, s[:5])),"...")
            print("t:   ", ''.join(map(str, t[:5])),"...")
        if self.output_plot:
            self.plotter.plot_modulation_time(
                dI, dQ, s, "dI(t)", "dQ(t)", "s(t)",
                "Sinal $IQ$ - Formatados RRC", "Sinal Modulado $IQ$",
                fs=self.fs, t_xlim=0.10,
                save_path="../out/transmitter_modulator_time.pdf"
            )
            self.plotter.plot_modulation_freq(
                dI, dQ, s,
                "$D_I'(f)$", "$D_Q'(f)$", "$S(f)$",
                "Sinal Banda Base - Componente $I$",
                "Sinal Banda Base - Componente $Q$",
                "Sinal Modulado $IQ$",
                fs=self.fs, fc=self.fc,
                save_path="../out/transmitter_modulator_freq.pdf"
            )
            self.plotter.plot_modulation_iq(
                dI, dQ,
                fr'Amostras $IQ$', fr'Simbolos $QPSK$',
                fr'Plano $IQ$ (Scatter)', fr'Plano $IQ$ (Constelação)',
                save_path="../out/transmitter_modulator_iq.pdf"
            )
        return t, s

    def run(self):
        r"""
        Executa o processo de transmissão, retornando o resultado da transmissão.

        Returns:
            TransmissionResult: Instância contendo o vetor de tempo e o sinal modulado.
        """
        ut = self.prepare_datagram()
        vt0, vt1 = self.encode_convolutional(ut)
        X, Y = self.scramble(vt0, vt1)
        sI, sQ = self.generate_preamble()
        Xn, Yn = self.multiplex(sI, sQ, X, Y)
        Xnrz, Yman = self.encode_channels(Xn, Yn)
        dI, dQ = self.format_signals(Xnrz, Yman)
        t, s = self.modulate(dI, dQ)
        return t, s


if __name__ == "__main__":
    datagram = Datagram(pcdnum=1234, numblocks=1)
    transmitter = Transmitter(datagram, output_print=True, output_plot=True)
    t, s = transmitter.run()
    result = TransmissionResult(t, s)
    result.save("../out/transmitter")
