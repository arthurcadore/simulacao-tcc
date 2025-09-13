"""
Implementação de um transmissor PTT-A3 com seus componentes.

Autor: Arthur Cadore
Data: 16-08-2025
"""
import numpy as np
from .formatter import Formatter
from .convolutional import EncoderConvolutional
from .datagram import Datagram
from .modulator import Modulator
from .preamble import Preamble
from .scrambler import Scrambler
from .multiplexer import Multiplexer
from .encoder import Encoder
from .data import ExportData, ImportData
from .plotter import create_figure, save_figure, BitsPlot, EncodedBitsPlot, ImpulseResponsePlot, TimePlot, FrequencyPlot, ConstellationPlot, PhasePlot

class Transmitter:
    def __init__(self, datagram: Datagram, fc=4000, fs=128_000, Rb=400, 
                 output_print=True, output_plot=True):
        r"""
        Classe que encapsula todo o processo de transmissão no padrão PTT-A3. A estrutura do transmissor é representada pelo diagrama de blocos abaixo.

        ![pageplot](../assets/blocos_modulador.svg)
    
        Args:
            datagram (Datagram): Instância do datagrama a ser transmitido.
            fc (float): Frequência da portadora em Hz. 
            fs (float): Frequência de amostragem em Hz. 
            Rb (float): Taxa de bits em bps.
            output_print (bool): Se `True`, imprime os vetores intermediários no console.
            output_plot (bool): Se `True`, gera e salva os gráficos dos processos intermediários.

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.1 e 3.2)
        </div>
        """
        self.datagram = datagram
        self.fc = fc
        self.fs = fs
        self.Rb = Rb
        self.output_print = output_print
        self.output_plot = output_plot

    def prepare_datagram(self):
        r"""
        Gera o datagrama para transmissão, retornando o vetor de bits $u_t$.

        Returns:
            ut (np.ndarray): Vetor de bits do datagrama.

        Exemplo:
            ![pageplot](assets/transmitter_datagram_time.svg)
        """
        ut = self.datagram.streambits
        if self.output_print:
            print("\n ==== MONTAGEM DATAGRAMA ==== \n")
            print(self.datagram.parse_datagram())
            print("\nut:", ''.join(map(str, ut)))
        if self.output_plot:
            fig_datagram, grid = create_figure(1, 1, figsize=(16, 5))

            BitsPlot(
                fig_datagram, grid, (0, 0),
                bits_list=[self.datagram.msglength, 
                           self.datagram.pcdid, 
                           self.datagram.blocks, 
                           self.datagram.tail],
                sections=[("Message Length", len(self.datagram.msglength)),
                          ("PCD ID", len(self.datagram.pcdid)),
                          ("Dados de App.", len(self.datagram.blocks)),
                          ("Tail", len(self.datagram.tail))],
                colors=["green", "orange", "red", "blue"]
            ).plot(xlabel="Index de Bit")

            fig_datagram.tight_layout()
            save_figure(fig_datagram, "transmitter_datagram_time.pdf")

        return ut

    def encode_convolutional(self, ut):
        r"""
        Codifica o vetor de bits $u_t$ usando codificação convolucional, retornando os vetores de bits $v_t^{(0)}$ e $v_t^{(1)}$.

        Args:
            ut (np.ndarray): Vetor de bits a ser codificado.

        Returns:
            vt0 (np.ndarray): Saída do canal I.
            vt1 (np.ndarray): Saída do canal Q.

        Exemplo:
            ![pageplot](assets/transmitter_conv_time.svg)
        """
        encoder = EncoderConvolutional()
        vt0, vt1 = encoder.encode(ut)
        if self.output_print:
            print("\n ==== CODIFICADOR CONVOLUCIONAL ==== \n")
            print("vt0:", ''.join(map(str, vt0)))
            print("vt1:", ''.join(map(str, vt1)))
        if self.output_plot:
            fig_conv, grid_conv = create_figure(3, 1, figsize=(16, 9))

            BitsPlot(
                fig_conv, grid_conv, (0, 0),
                bits_list=[ut],
                sections=[("$u_t$", len(ut))],
                colors=["darkred"]
            ).plot(ylabel="$u_t$")

            BitsPlot(
                fig_conv, grid_conv, (1, 0),
                bits_list=[vt0],
                sections=[("$v_t^{(0)}$", len(vt0))],
                colors=["darkgreen"]
            ).plot(ylabel="$v_t^{(0)}$")

            BitsPlot(
                fig_conv, grid_conv, (2, 0),
                bits_list=[vt1],
                sections=[("$v_t^{(1)}$", len(vt1))],
                colors=["navy"]
            ).plot(ylabel="$v_t^{(1)}$", xlabel="Index de Bit")

            fig_conv.tight_layout()
            save_figure(fig_conv, "transmitter_conv_time.pdf")       
        return vt0, vt1

    def scramble(self, vt0, vt1):
        r"""
        Embaralha os vetores de bits $v_t^{(0)}$ e $v_t^{(1)}$, criando os vetores $X_n$ e $Y_n$ embaralhados.

        Args:
            vt0 (np.ndarray): Vetor de bits do canal I.
            vt1 (np.ndarray): Vetor de bits do canal Q.

        Returns:
            Xn (np.ndarray): Vetor embaralhado do canal I.
            Yn (np.ndarray): Vetor embaralhado do canal Q.

        Exemplo:
            ![pageplot](assets/transmitter_scrambler_time.svg)
        """
        scrambler = Scrambler()
        X, Y = scrambler.scramble(vt0, vt1)
        if self.output_print:
            print("\n ==== EMBARALHADOR ==== \n")
            print("Xn:", ''.join(map(str, X)))
            print("Yn:", ''.join(map(str, Y)))
        if self.output_plot:
            fig_scrambler, grid_scrambler = create_figure(2, 2, figsize=(16, 9))

            BitsPlot(
                fig_scrambler, grid_scrambler, (0, 0),
                bits_list=[vt0],
                sections=[("$v_t^{0}$", len(vt0))],
                colors=["darkgreen"]
            ).plot(ylabel="Original")

            BitsPlot(
                fig_scrambler, grid_scrambler, (0, 1),
                bits_list=[vt1],
                sections=[("$v_t^{1}$", len(vt1))],
                colors=["navy"]
            ).plot()

            BitsPlot(
                fig_scrambler, grid_scrambler, (1, 0),
                bits_list=[X],
                sections=[("$X_n$", len(X))],
                colors=["darkgreen"]
            ).plot(ylabel="Embaralhado", xlabel="Index de Bit")

            BitsPlot(
                fig_scrambler, grid_scrambler, (1, 1),
                bits_list=[Y],
                sections=[("$Y_n$", len(Y))],
                colors=["navy"]
            ).plot(xlabel="Index de Bit")

            fig_scrambler.tight_layout()
            save_figure(fig_scrambler, "transmitter_scrambler_time.pdf")

        return X, Y

    def generate_preamble(self):
        r"""
        Gera os vetores de preâmbulo $S_I$ e $S_Q$.

        Returns:
            sI (np.ndarray): Vetor do preâmbulo do canal I.
            sQ (np.ndarray): Vetor do preâmbulo do canal Q.

        Exemplo:
            ![pageplot](assets/transmitter_preamble_time.svg)
        """
        sI, sQ = Preamble().generate_preamble()
        if self.output_print:
            print("\n ==== MONTAGEM PREAMBULO ==== \n")
            print("sI:", ''.join(map(str, sI)))
            print("sQ:", ''.join(map(str, sQ)))
        if self.output_plot:
            fig_preamble, grid_preamble = create_figure(2, 1, figsize=(16, 9))

            BitsPlot(
                fig_preamble, grid_preamble, (0,0),
                bits_list=[sI],
                sections=[("Preambulo $S_I$", len(sI))],
                colors=["darkgreen"]
            ).plot(ylabel="Canal $I$")
            
            BitsPlot(
                fig_preamble, grid_preamble, (1,0),
                bits_list=[sQ],
                sections=[("Preambulo $S_Q$", len(sQ))],
                colors=["navy"]
            ).plot(xlabel="Index de Bit", ylabel="Canal $Q$")

            fig_preamble.tight_layout()
            save_figure(fig_preamble, "transmitter_preamble_time.pdf")
        return sI, sQ

    def multiplex(self, sI, sQ, X, Y):
        r"""
        Multiplexa os vetores de preâmbulo $S_I$ e $S_Q$ com os vetores de dados $X$ e $Y$, retornando os vetores multiplexados $X_n$ e $Y_n$.

        Args:
            sI (np.ndarray): Vetor do preâmbulo do canal I.
            sQ (np.ndarray): Vetor do preâmbulo do canal Q.
            X (np.ndarray): Vetor de dados do canal I.
            Y (np.ndarray): Vetor de dados do canal Q.
        
        Returns:
            Xn (np.ndarray): Vetor multiplexado do canal I.
            Yn (np.ndarray): Vetor multiplexado do canal Q.

        Exemplo:
            ![pageplot](assets/transmitter_mux_time.svg)
        """

        multiplexer = Multiplexer()
        Xn, Yn = multiplexer.concatenate(sI, sQ, X, Y)
        if self.output_print:
            print("\n ==== MULTIPLEXADOR ==== \n")
            print("Xn:", ''.join(map(str, Xn)))
            print("Yn:", ''.join(map(str, Yn)))
        if self.output_plot:
            fig_mux, grid_mux = create_figure(2, 1, figsize=(16, 9))
            BitsPlot(
                fig_mux, grid_mux, (0,0),
                bits_list=[sI, X],
                sections=[("Preambulo $S_I$", len(sI)),
                          ("Canal I $(X_n)$", len(X))],
                colors=["darkred", "darkgreen"]
            ).plot(ylabel="Canal $I$")

            BitsPlot(
                fig_mux, grid_mux, (1,0),
                bits_list=[sQ, Y],
                sections=[("Preambulo $S_Q$", len(sQ)),
                          ("Canal Q $(Y_n)$", len(Y))],
                colors=["darkred", "navy"]
            ).plot(xlabel="Index de Bit", ylabel="Canal $Q$")

            fig_mux.tight_layout()
            save_figure(fig_mux, "transmitter_mux_time.pdf")   
        return Xn, Yn

    def encode_channels(self, Xn, Yn):
        r"""
        Codifica os vetores dos canais $X_n$ e $Y_n$ usando $NRZ$ e $Manchester$, respectivamente, retornando os vetores de sinal codificados $X_{NRZ}$ e $Y_{MAN}$.

        Args:
            Xn (np.ndarray): Vetor do canal $X_n$ a ser codificado.
            Yn (np.ndarray): Vetor do canal $Y_n$ a ser codificado.
        
        Returns:
            Xnrz (np.ndarray): Vetor de sinal codificado do canal I $NRZ$. 
            Yman (np.ndarray): Vetor de sinal codificado do canal Q $Manchester$. 

        Exemplo:
            ![pageplot](assets/transmitter_encoder_time.svg)
        """

        encoderNRZ = Encoder("nrz")
        encoderManchester = Encoder("nrz2")
        Xnrz = encoderNRZ.encode(Xn)
        Yman = encoderManchester.encode(Yn)

        if self.output_print:
            print("\n ==== CODIFICAÇÃO DE LINHA ==== \n")
            print("Xnrz:", ' '.join(f"{x:+d}" for x in Xnrz[:40]),"...")
            print("Yman:", ' '.join(f"{y:+d}" for y in Yman[:40]),"...")
        if self.output_plot:
            fig_encoder, grid = create_figure(4, 1, figsize=(16, 9))

            BitsPlot(
                fig_encoder, grid, (0, 0),
                bits_list=[Xn],
                sections=[("$X_n$", len(Xn))],
                colors=["darkgreen"]
            ).plot(xlabel="Index de Bit", ylabel="$X_n$", xlim=(0, len(Xn)/2))

            EncodedBitsPlot(
                fig_encoder, grid, (1, 0),
                bits=Xnrz,
                color='darkgreen',
            ).plot(xlabel="Index de Simbolo", ylabel="$X_{NRZ}[n]$", label="$X_{NRZ}[n]$", xlim=(0, len(Xnrz)/2))

            BitsPlot(
                fig_encoder, grid, (2, 0),
                bits_list=[Yn],
                sections=[("$Y_n$", len(Yn))],
                colors=["navy"]
            ).plot(xlabel="Index de Bit", ylabel="$Y_n$", xlim=(0, len(Yn)/2))

            EncodedBitsPlot(
                fig_encoder, grid, (3, 0),
                bits=Yman,
                color="navy",
            ).plot(xlabel="Index de Simbolo", ylabel="$Y_{MAN}[n]$", label="$Y_{MAN}[n]$", xlim=(0, len(Yman)/2))

            fig_encoder.tight_layout()
            save_figure(fig_encoder, "transmitter_encoder_time.pdf")
        return Xnrz, Yman

    def format_signals(self, Xnrz, Yman):
        r"""
        Formata os vetores de sinal codificados $X_{NRZ}$ e $Y_{MAN}$ usando filtro RRC, retornando os vetores formatados $d_I$ e $d_Q$.

        Args:
            Xnrz (np.ndarray): Vetor do canal $X_{NRZ}$ a ser formatado.
            Yman (np.ndarray): Vetor do canal $Y_{MAN}$ a ser formatado.
        
        Returns:
            dI (np.ndarray): Vetor formatado do canal I, $d_I$.
            dQ (np.ndarray): Vetor formatado do canal Q, $d_Q$.

        Exemplo:
            - Tempo: ![pageplot](assets/transmitter_formatter_time.svg)
            - Frequência: ![pageplot](assets/transmitter_formatter_freq.svg)
        """

        formatterI = Formatter(fs=self.fs, Rb=self.Rb, type="RRC", channel="I", bits_per_symbol=1)
        formatterQ = Formatter(fs=self.fs, Rb=self.Rb, type="Manchester", channel="Q", bits_per_symbol=2)

        dI = formatterI.apply_format(Xnrz)
        dQ = formatterQ.apply_format(Yman)
        
        if self.output_print:
            print("\n ==== FORMATADOR ==== \n")
            print("dI:", ''.join(map(str, dI[:5])),"...")
            print("dQ:", ''.join(map(str, dQ[:5])),"...")
            
        if self.output_plot:
            fig_format, grid_format = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_format, grid_format, (0, 0),
                formatterI.t_rc, formatterI.g,
                t_unit="ms",
                colors="darkorange",
            ).plot(label="$g(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

            ImpulseResponsePlot(
                fig_format, grid_format, (0, 1),
                formatterQ.t_rc, formatterQ.g,
                t_unit="ms",
                colors="darkorange",
            ).plot(label="$g(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

            TimePlot(
                fig_format, grid_format, (1,0),
                t= np.arange(len(dI)) / formatterI.fs,
                signals=[dI],
                labels=["$d_I(t)$"],
                title="Canal $I$",
                xlim=(40, 200),
                colors="darkgreen",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            TimePlot(
                fig_format, grid_format, (1,1),
                t= np.arange(len(dQ)) / formatterQ.fs,
                signals=[dQ],
                labels=["$d_Q(t)$"],
                title="Canal $Q$",
                xlim=(40, 200),
                colors="darkblue",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            fig_format.tight_layout()
            save_figure(fig_format, "transmitter_formatter_time.pdf")

            fig_format_freq, grid_format_freq = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_format_freq, grid_format_freq, (0, 0),
                formatterI.t_rc, formatterI.g,
                t_unit="ms",
                colors="darkorange",
            ).plot(label="$g(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

            ImpulseResponsePlot(
                fig_format_freq, grid_format_freq, (0, 1),
                formatterQ.t_rc, formatterQ.g,
                t_unit="ms",
                colors="darkorange",
            ).plot(label="$g(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

            FrequencyPlot(
                fig_format_freq, grid_format_freq, (1, 0),
                fs=self.fs,
                signal=dI,
                fc=self.fc,
                labels=["$D_I(f)$"],
                title="Canal $I$",
                xlim=(-1.5, 1.5),
                colors="darkgreen",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_format_freq, grid_format_freq, (1, 1),
                fs=self.fs,
                signal=dQ,
                fc=self.fc,
                labels=["$D_Q(f)$"],
                title="Canal $Q$",
                xlim=(-1.5, 1.5),
                colors="navy",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            fig_format_freq.tight_layout()
            save_figure(fig_format_freq, "transmitter_formatter_freq.pdf")

        return dI, dQ

    def modulate(self, dI, dQ):
        r"""
        Modula os vetores de sinal $d_I(t)$ e $d_Q(t)$ usando modulação QPSK, retornando o sinal modulado $s(t)$.

        Args:
            dI (np.ndarray): Vetor formatado do canal I, $d_I(t)$.
            dQ (np.ndarray): Vetor formatado do canal Q, $d_Q(t)$.
        
        Returns:
            t (np.ndarray): Vetor de tempo, $t$.
            s (np.ndarray): Sinal modulado, $s(t)$.

        Exemplo:
            - Tempo: ![pageplot](assets/transmitter_modulator_time.svg)
            - Frequência: ![pageplot](assets/transmitter_modulator_freq.svg)
            - Portadora: ![pageplot](assets/transmitter_modulator_portadora.svg)
            - Fase e Constelação: ![pageplot](assets/transmitter_modulator_constellation.svg)
        """
        modulator = Modulator(fc=self.fc, fs=self.fs)
        t, s = modulator.modulate(dI, dQ)
        if self.output_print:
            print("\n ==== MODULADOR ==== \n")
            print("s(t):", ''.join(map(str, s[:5])),"...")
            print("t:   ", ''.join(map(str, t[:5])),"...")
        if self.output_plot:
            fig_time, grid = create_figure(2, 1, figsize=(16, 9))
            TimePlot(
                fig_time, grid, (0, 0),
                t=t,
                signals=[dI, dQ],
                labels=["$d_I(t)$", "$d_Q(t)$"],
                title="Componentes $IQ$ - Demoduladas",
                xlim=(40, 200),
                colors=["darkgreen", "navy"],
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            TimePlot(
                fig_time, grid, (1, 0),
                t=t,
                signals=[s],
                labels=["$s(t)$"],
                title="Sinal Modulado $IQ$",
                xlim=(40, 200),
                colors="darkred",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            fig_time.tight_layout()
            save_figure(fig_time, "transmitter_modulator_time.pdf")

            fig_freq, grid = create_figure(2, 2, figsize=(16, 9))
            FrequencyPlot(
                fig_freq, grid, (0, 0),
                fs=self.fs,
                signal=dI,
                fc=self.fc,
                labels=["$D_I(f)$"],
                title="Componente I",
                xlim=(-1.5, 1.5),
                colors="darkgreen",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()
        
            FrequencyPlot(
                fig_freq, grid, (0, 1),
                fs=self.fs,
                signal=dQ,
                fc=self.fc,
                labels=["$D_Q(f)$"],
                title="Componente Q",
                xlim=(-1.5, 1.5),
                colors="navy",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()
        
            FrequencyPlot(
                fig_freq, grid, (1, slice(0, 2)),
                fs=self.fs,
                signal=s,
                fc=self.fc,
                labels=["$S(f)$"],
                title="Sinal Modulado $IQ$",
                xlim=(-10, 10),
                colors="darkred",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()
        
            fig_freq.tight_layout()
            save_figure(fig_freq, "transmitter_modulator_freq.pdf")

            # PLOT 3 - Constelação
            fig_const, grid = create_figure(1, 2, figsize=(16, 8))
            PhasePlot(
                fig_const, grid, (0, 0),
                t=t,
                signals=[dI, dQ],
                labels=["Fase $I + jQ$"],
                title="Fase $I + jQ$",
                xlim=(40, 200),
                ylim=(-np.pi, np.pi),
                colors=["darkred"],
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            ConstellationPlot(
                fig_const, grid, (0, 1),
                dI=dI[:40000:5],
                dQ=dQ[:40000:5],
                xlim=(-1.1, 1.1),
                ylim=(-1.1, 1.1),
                title="Constelação $IQ$",
                colors=["darkred"],
                style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot(show_ideal_points=False)

            fig_const.tight_layout()
            save_figure(fig_const, "transmitter_modulator_constellation.pdf") 

            # PLOT 4 - Portadora pura e sinal modulado
            fig_portadora, grid = create_figure(1, 2, figsize=(16, 8))
            FrequencyPlot(
                fig_portadora, grid, (0, 0),
                fs=self.fs,
                signal=s[0:(int(round(0.082 * self.fs)))],
                fc=self.fc,
                labels=["$S(f)$"],
                title="Portadora Pura - $0$ a $80$ms",
                xlim=(-10, 10),
                colors="darkred",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_portadora, grid, (0, 1),
                fs=self.fs,
                signal=s[(int(round(0.082 * self.fs))):],
                fc=self.fc,
                labels=["$S(f)$"],
                title="Sinal Modulado - $80$ms em diante",
                xlim=(-10, 10),
                colors="darkred",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            fig_portadora.tight_layout()
            save_figure(fig_portadora, "transmitter_modulator_portadora.pdf")

        return t, s

    def run(self):
        r"""
        Executa o processo de transmissão, retornando o sinal modulado $s(t)$ e o vetor de tempo $t$.

        Returns:
            t (np.ndarray): Vetor de tempo, $t$.
            s (np.ndarray): Sinal modulado, $s(t)$.
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

    ExportData([s, t], "transmitter_st").save()

    # ## TESTE DE IMPORT:

    # # Importa os dados    
    # import_data = ImportData("transmitter_st")
    # st = import_data.load()
    
    # print("s(t):", ''.join(map(str, s[:5])),"...")
    # print("t:   ", ''.join(map(str, t[:5])),"...")


    # # Verifica se os dados importados são iguais aos dados exportados
    # if np.array_equal(s, st[0]) and np.array_equal(t, st[1]):
    #     print("\nOs dados importados são iguais aos dados exportados.")
    