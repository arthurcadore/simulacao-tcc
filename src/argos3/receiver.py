# """
# Implementação de um receptor PTT-A3 com seus componentes.

# Autor: Arthur Cadore
# Data: 16-08-2025
# """

import numpy as np
from .datagram import Datagram
from .modulator import Modulator
from .scrambler import Scrambler
from .encoder import Encoder
from .transmitter import Transmitter
from .noise import NoiseEBN0
from .lowpassfilter import LPF
from .matchedfilter import MatchedFilter
from .sampler import Sampler
from .convolutional import DecoderViterbi
from .synchronizer import Synchronizer
from .plotter import save_figure, create_figure, TimePlot, FrequencyPlot, ImpulseResponsePlot, SampledSignalPlot, BitsPlot, PhasePlot, ConstellationPlot, FrequencyResponsePlot, SincronizationPlot, CorrelationPlot, SymbolsPlot

class Receiver:
    def __init__(self, fs=128_000, Rb=400, fc=None, lpf_cutoff=600, preamble="2BEEEEBF", channel_encode=("nrz", "man"), G=np.array([[0b1111001, 0b1011011]]), output_print=True, output_plot=True):
        r"""
        Classe que encapsula todo o processo de recepção no padrão ARGOS-3. A estrutura do receptor é representada pelo diagrama de blocos abaixo.

        ![pageplot](../assets/blocos_demodulador.svg)

        Args:
            fs (int): Frequência de amostragem em Hz.
            Rb (int): Taxa de bits em bps.
            fc (int): Frequência de portadora em Hz.
            lpf_cutoff (int): Frequência de corte do filtro passa-baixa em Hz.
            preamble (str): String de preâmbulo em hex.
            channel_encode (tuple): Tupla com o tipo de codificação dos canais I e Q respectivamente.
            G (np.ndarray): Matriz de geração para codificação convolucional.
            output_print (bool): Se `True`, imprime os vetores intermediários no console. 
            output_plot (bool): Se `True`, gera e salva os gráficos dos processos intermediários.

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-2097-CNES (seção 3.1 e 3.2)
        </div>
        """

        # Validar os valores de channel_encode
        valid_encodings = ["nrz", "man"]
        if channel_encode[0] not in valid_encodings or channel_encode[1] not in valid_encodings:
            raise ValueError("Os tipos de codificação devem ser 'nrz' ou 'manchester'.")

        # Parâmetros
        self.fs = fs
        self.Rb = Rb
        self.fc = fc
        self.lpf_cutoff = lpf_cutoff
        self.output_print = output_print
        self.output_plot = output_plot
        self.preamble = preamble
        self.G = G

        # Parâmetros fixos
        self.alpha = 0.8
        self.span = 24
        self.lpf_order = 6
        self.delayI = 0
        self.delayQ = 0
        self.cI_encoder = "nrz"
        self.cQ_encoder = "nrz"

        # Codificação I e Q
        self.cI_type = channel_encode[0]
        self.cQ_type = channel_encode[1]

        # Mapeamento das configurações de codificação
        encoding_params = {
            "nrz": {"format": "RRC", "bits_per_symbol": 1, "Rb_multiplier": 1, "matched": "RRC-Inverted"},
            "man": {"format": "Manchester", "bits_per_symbol": 2, "Rb_multiplier": 2, "matched": "Manchester-Inverted"}
        }

        # Parâmetros para o canal I e Q
        cI_params = encoding_params[self.cI_type]
        self.cI_format = cI_params["format"]
        self.cI_bits_per_symbol = cI_params["bits_per_symbol"]
        self.cI_Rb = self.Rb
        self.cI_matched = cI_params["matched"]
        cQ_params = encoding_params[self.cQ_type]
        self.cQ_format = cQ_params["format"]
        self.cQ_bits_per_symbol = cQ_params["bits_per_symbol"]
        self.cQ_Rb = self.Rb
        self.cQ_matched = cQ_params["matched"]


        # Submodulos
        self.demodulator = Modulator(fc=self.fc, fs=self.fs)
        self.lpf = LPF(cut_off=self.lpf_cutoff, order=self.lpf_order, fs=self.fs, type="butter")
        self.matched_filterI = MatchedFilter(alpha=self.alpha, fs=self.fs, Rb=self.Rb, span=self.span, type="RRC-Inverted", channel="I", bits_per_symbol=1)
        self.matched_filterQ = MatchedFilter(alpha=self.alpha, fs=self.fs, Rb=self.Rb, span=self.span, type="Manchester-Inverted", channel="Q", bits_per_symbol=2)
        self.synchronizerI = Synchronizer(fs=self.fs, Rb=self.Rb, sync_word=self.preamble)
        self.synchronizerQ = Synchronizer(fs=self.fs, Rb=self.Rb, sync_word=self.preamble)
        self.samplerI = Sampler(fs=self.fs, Rb=self.Rb, delay=self.delayI)
        self.samplerQ = Sampler(fs=self.fs, Rb=self.Rb, delay=self.delayQ)
        self.decoderI = Encoder("nrz")
        self.decoderQ = Encoder("nrz")
        self.unscrambler = Scrambler()
        self.conv_viterbi = DecoderViterbi(G=self.G)

    def demodulate(self, s, t):
        r"""normalmente
        Demodula o sinal $s'(t)$ com ruído recebido, recuperando os sinais $x'_{I}(t)$ e $y'_{Q}(t)$.

        Args:
            s (np.ndarray): Sinal $s'(t)$ a ser demodulado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            xI_prime (np.ndarray): Sinal $x'_{I}(t)$ demodulado.
            yQ_prime (np.ndarray): Sinal $y'_{Q}(t)$ demodulado.
        
        Example:
            - Tempo: ![pageplot](assets/receiver_demodulator_time.svg)
            - Frequência: ![pageplot](assets/receiver_demodulator_freq.svg)
        """

        xI_prime, yQ_prime = self.demodulator.demodulate(s)

        if self.output_print:
            print("\n ==== DEMODULADOR ==== \n")
            print("x'I(t):", ''.join(map(str, xI_prime[:5])),"...")
            print("y'Q(t):", ''.join(map(str, yQ_prime[:5])),"...")

        if self.output_plot:
            fig_time, grid = create_figure(2, 1, figsize=(16, 9))

            TimePlot(
                fig_time, grid, (0, 0),
                t=t,
                signals=[s],
                labels=[r"$s(t)$ + AWGN"],
                title=r"Sinal Modulado + Ruído $Eb/N_0$ 20 $dB$",
                xlim=(40, 200),
                amp_norm=True,
                colors="darkred",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            TimePlot(
                fig_time, grid, (1, 0),
                t=t,
                signals=[xI_prime, yQ_prime],
                labels=[r"$xI'(t)$", r"$yQ'(t)$"],
                title=r"Componentes $IQ$ - Demoduladas",
                xlim=(40, 200),
                amp_norm=True,
                colors=["darkgreen", "navy"],
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            fig_time.tight_layout()
            save_figure(fig_time, "receiver_demodulator_time.pdf")

            fig_freq, grid = create_figure(3, 1, figsize=(16, 9))

            FrequencyPlot(
                fig_freq, grid, (0, 0),
                fs=self.fs,
                signal=s,
                fc=self.fc,
                labels=[r"$S(f)$"],
                title=r"Sinal Modulado $IQ$",
                xlim=(-10, 10),
                colors="darkred",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_freq, grid, (1, 0),
                fs=self.fs,
                signal=xI_prime,
                fc=self.fc,
                labels=[r"$X_I'(f)$"],
                title=r"Componente $I$ - Demodulada",
                xlim=(-10, 10),
                colors="darkgreen",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_freq, grid, (2, 0),
                fs=self.fs,
                signal=yQ_prime,
                fc=self.fc,
                labels=[r"$Y_Q'(f)$"],
                title=r"Componente $Q$ - Demodulada",
                xlim=(-10, 10),
                colors="navy",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            fig_freq.tight_layout()
            save_figure(fig_freq, "receiver_demodulator_freq.pdf")

        return xI_prime, yQ_prime
    
    def lowpassfilter(self, xI_prime, yQ_prime, t):
        r"""
        Aplica o filtro passa-baixa com resposta ao impuslo $h(t)$ aos sinais $x'_{I}(t)$ e $y'_{Q}(t)$, retornando os sinais filtrados $d'_{I}(t)$ e $d'_{Q}(t)$.

        Args:
            xI_prime (np.ndarray): Sinal $x'_{I}(t)$ a ser filtrado.
            yQ_prime (np.ndarray): Sinal $y'_{Q}(t)$ a ser filtrado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            dI_prime (np.ndarray): Sinal $d'_{I}(t)$ filtrado.
            dQ_prime (np.ndarray): Sinal $d'_{Q}(t)$ filtrado.

        Example:
            - Tempo: ![pageplot](assets/receiver_lpf_time.svg)
            - Frequência: ![pageplot](assets/receiver_lpf_freq.svg)
        """

        impulse_response, t_impulse = self.lpf.calc_impulse_response()

        dI_prime = self.lpf.apply_filter(xI_prime)
        dQ_prime = self.lpf.apply_filter(yQ_prime)

        if self.output_print:
            print("\n ==== FILTRAGEM PASSA-BAIXA ==== \n")
            print("d'I(t):", ''.join(map(str, dI_prime[:5])),"...")
            print("d'Q(t):", ''.join(map(str, dQ_prime[:5])),"...")
        
        if self.output_plot:
            fig_signal, grid_signal = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_signal, grid_signal, (0, slice(0, 2)),
                t_impulse, impulse_response,
                t_unit="ms",
                colors="darkorange",
                label=r"$h(t)$", 
                xlabel=r"Tempo ($ms$)", 
                ylabel="Amplitude", 
                xlim=(0, 8),
                amp_norm=True,
            ).plot()

            TimePlot(
                fig_signal, grid_signal, (1, 0),
                t=t, 
                signals=[dI_prime],
                labels=["$d_I'(t)$"],  
                title="Sinal filtrado - Componente $I$",
                xlim=(40, 200),
                ylim=(-1, 1),
                amp_norm=True,
                colors="darkgreen"
            ).plot()

            TimePlot(
                fig_signal, grid_signal, (1, 1),
                t=t, 
                signals=[dQ_prime],
                labels=["$d_Q'(t)$"],
                title="Sinal filtrado - Componente $Q$",
                xlim=(40, 200),
                ylim=(-1, 1),
                amp_norm=True,
                colors="navy"
            ).plot()

            fig_signal.tight_layout()
            save_figure(fig_signal, "receiver_lpf_time.pdf")

            fig_freq, grid_freq = create_figure(3, 2, figsize=(16, 9))

            FrequencyResponsePlot(
                fig_freq, grid_freq, (0, slice(0, 2)),
                self.lpf.b,
                self.lpf.a,
                self.fs,
                f_cut=self.lpf_cutoff,
                xlim=(0, 3*self.lpf_cutoff),
            ).plot()

            FrequencyPlot(
                fig_freq, grid_freq, (1, 0), 
                fs=self.fs,
                signal=xI_prime,
                fc=self.fc,
                labels=[r"$X_I'(f)$"],
                title=r"Componente $I$ - Demodulada",
                xlim=(-10, 10),
                colors="darkgreen",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_freq, grid_freq, (1, 1), 
                fs=self.fs,
                signal=yQ_prime,
                fc=self.fc,
                labels=[r"$Y_Q'(f)$"],
                title=r"Componente $Q$ - Demodulada",
                xlim=(-10, 10),
                colors="navy",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_freq, grid_freq, (2, 0), 
                fs=self.fs,
                signal=dI_prime,
                fc=self.fc,
                labels=[r"$d_I'(f)$"],
                title=r"Componente $I$ - Filtrada",
                xlim=(-10, 10),
                colors="darkgreen",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_freq, grid_freq, (2, 1), 
                fs=self.fs,
                signal=dQ_prime,
                fc=self.fc,
                labels=[r"$d_Q'(f)$"],
                title=r"Componente $Q$ - Filtrada",
                xlim=(-10, 10),
                colors="navy",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            fig_freq.tight_layout()
            save_figure(fig_freq, "receiver_lpf_freq.pdf")


        return dI_prime, dQ_prime

    def matchedfilter(self, dI_prime, dQ_prime, t):
        r"""
        Aplica o filtro casado com resposta ao impuslo $g(-t)$ aos sinais $d'_{I}(t)$ e $d'_{Q}(t)$, retornando os sinais filtrados $I'(t)$ e $Q'(t)$.

        Args:
            dI_prime (np.ndarray): Sinal $d'_{I}(t)$ a ser filtrado.
            dQ_prime (np.ndarray): Sinal $d'_{Q}(t)$ a ser filtrado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            It_prime (np.ndarray): Sinal $I'(t)$ filtrado.
            Qt_prime (np.ndarray): Sinal $Q'(t)$ filtrado.

        Example:
            - Tempo: ![pageplot](assets/receiver_mf_time.svg)
            - Frequência: ![pageplot](assets/receiver_mf_freq.svg)
        """

        It_prime = self.matched_filterI.apply_filter(dI_prime)
        Qt_prime = self.matched_filterQ.apply_filter(dQ_prime)

        if self.output_print:
            print("\n ==== FILTRAGEM CASADA ==== \n")
            print("I'(t):", ''.join(map(str, It_prime[:5])),"...")
            print("Q'(t):", ''.join(map(str, Qt_prime[:5])),"...")

        if self.output_plot:
            fig_matched, grid_matched = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_matched, grid_matched, (0, 0),
                self.matched_filterI.t_rc, self.matched_filterI.g_inverted,
                t_unit="ms",
                colors="darkorange",
                label=r"$g(-t)$", 
                xlabel=r"Tempo ($ms$)", 
                ylabel="Amplitude", 
                xlim=(-15, 15),
                amp_norm=True,
            ).plot()

            ImpulseResponsePlot(
                fig_matched, grid_matched, (0, 1),
                self.matched_filterQ.t_rc, self.matched_filterQ.g_inverted,
                t_unit="ms",
                colors="darkorange",
                label=r"$g(-t)$", 
                xlabel=r"Tempo ($ms$)", 
                ylabel="Amplitude", 
                xlim=(-15, 15),
                amp_norm=True,
            ).plot()

            TimePlot(
                fig_matched, grid_matched, (1, 0),
                t,
                It_prime,
                labels=[r"$I'(t)$"],
                title=r"Sinal filtrado - Componente $I$",
                xlim=(40, 200),
                amp_norm=True,
                colors="darkgreen"
            ).plot()

            TimePlot(
                fig_matched, grid_matched, (1, 1),
                t,
                Qt_prime,
                labels=[r"$Q'(t)$"],
                title=r"Sinal filtrado - Componente $Q$",
                xlim=(40, 200),
                amp_norm=True,
                colors="navy"
            ).plot()

            fig_matched.tight_layout()
            save_figure(fig_matched, "receiver_mf_time.pdf")

            fig_matched_freq, grid_matched_freq = create_figure(3, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_matched_freq, grid_matched_freq, (0, 0),
                self.matched_filterI.t_rc, self.matched_filterI.g_inverted,
                t_unit="ms",
                colors="darkorange",
                label=r"$g(-t)$", 
                xlabel=r"Tempo ($ms$)", 
                ylabel="Amplitude", 
                xlim=(-15, 15),
                amp_norm=True,
            ).plot()

            ImpulseResponsePlot(
                fig_matched_freq, grid_matched_freq, (0, 1),
                self.matched_filterQ.t_rc, self.matched_filterQ.g_inverted,
                t_unit="ms",
                colors="darkorange",
                label=r"$g(-t)$", 
                xlabel=r"Tempo ($ms$)", 
                ylabel="Amplitude", 
                xlim=(-15, 15),
                amp_norm=True,
            ).plot()

            FrequencyPlot(
                fig_matched_freq, grid_matched_freq, (1, 0),
                fs=self.fs,
                signal=dI_prime,
                fc=self.fc,
                labels=[r"$d_I'(f)$"],
                title=r"Componente $I$ - Fitragem Passa-Baixa",
                xlim=(-10, 10),
                colors="darkgreen",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_matched_freq, grid_matched_freq, (1, 1),
                fs=self.fs,
                signal=dQ_prime,
                fc=self.fc,
                labels=[r"$d_Q'(f)$"],
                title=r"Componente $Q$ - Fitragem Passa-Baixa",
                xlim=(-10, 10),
                colors="navy",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_matched_freq, grid_matched_freq, (2, 0),
                fs=self.fs,
                signal=It_prime,
                fc=self.fc,
                labels=[r"$I'(f)$"],
                title=r"Componente $I$ - Fitragem Casada",
                xlim=(-10, 10),
                colors="darkgreen",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            FrequencyPlot(
                fig_matched_freq, grid_matched_freq, (2, 1),
                fs=self.fs,
                signal=Qt_prime,
                fc=self.fc,
                labels=[r"$Q'(f)$"],
                title=r"Componente $Q$ - Fitragem Casada",
                xlim=(-10, 10),
                colors="navy",
                style={"line": {"linewidth": 1, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot()

            fig_matched_freq.tight_layout()
            save_figure(fig_matched_freq, "receiver_mf_freq.pdf")

        return It_prime, Qt_prime

    def synchronizer(self, It_prime, Qt_prime):
        r"""
        Realiza a sincronização do sinal recebido, retornando o sinal sincronizado.

        Args:
            It_prime (np.ndarray): Sinal $I'(t)$ a ser sincronizado.
            Qt_prime (np.ndarray): Sinal $Q'(t)$ a ser sincronizado.

        Returns:
            delayI (float): Delay do sinal $I'(t)$.
            delayQ (float): Delay do sinal $Q'(t)$.

        Example:
            Tempo: ![pageplot](assets/receiver_sync_time.svg)
            Módulo Correlação: ![pageplot](assets/receiver_sync_corr.svg)
        """

        delayI_min, delayI_max, delayI, corr_vec_I = self.synchronizerI.correlation(It_prime, "I")
        delayQ_min, delayQ_max, delayQ, corr_vec_Q = self.synchronizerQ.correlation(Qt_prime, "Q")

        if self.output_print:
            print("\n ==== SINCRONIZADOR ==== \n")
            print("Delay Q Min  :", delayQ_min)
            print("Delay Q Max  :", delayQ_max)
            print("Delay Q Corr :", delayQ)
            print("Delay I Min  :", delayI_min)
            print("Delay I Max  :", delayI_max)
            print("Delay I Corr :", delayI)

        # Nota: delayI e delayQ devem ser iguais, portanto, delayI é ajustado para ser igual a delayQ
        # O ajuste deve ser feito pois o canal I não suporta sincronização.
        delayI_min, delayI_max, delayI = delayQ_min, delayQ_max, delayQ
        
        if self.output_plot:
            fig_corr, grid_corr = create_figure(1, 1, figsize=(16, 9))
            CorrelationPlot(
                fig_corr, grid_corr, (0, 0),
                corr_vec=corr_vec_Q,  
                fs=self.fs,
                xlim_ms=(40, 200),
                colors="darkblue",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                },
            ).plot()
            fig_corr.tight_layout()
            
            save_figure(fig_corr, "receiver_sync_corr.pdf")
    
            fig_sync, grid_sync = create_figure(2,1, figsize=(16, 9))

            SincronizationPlot(
                fig_sync, grid_sync, (0,0),
                t= np.arange(len(It_prime)) / self.fs,
                signal=It_prime,
                sync_start=delayI_min,
                sync_end=delayI_max,
                max_corr=delayI,
                title=r"Canal $I$",
                labels=[r"$d_I(t)$"],
                colors="darkgreen",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                },
                xlim=(40, 200),
            ).plot()

            SincronizationPlot(
                fig_sync, grid_sync, (1,0),
                t=np.arange(len(Qt_prime)) / self.fs,
                signal=Qt_prime,
                sync_start=delayQ_min,
                sync_end=delayQ_max,
                max_corr=delayQ,
                title=r"Canal $Q$",
                labels=[r"$d_Q(t)$"],
                colors="darkblue",
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                },
                xlim=(40, 200),
            ).plot()

            fig_sync.tight_layout()
            save_figure(fig_sync, "receiver_sync_time.pdf")

        return delayI_max, delayQ_max

    def sampler(self, It_prime, Qt_prime, t):
        r"""
        Realiza a decisão (amostragem e quantização) dos sinais $I'(t)$ e $Q'(t)$, retornando os vetores de simbolos $X'_{NRZ}[n]$ e $Y'_{MAN}[n]$.

        Args:
            It_prime (np.ndarray): Sinal $I'(t)$ a ser amostrado e quantizado.
            Qt_prime (np.ndarray): Sinal $Q'(t)$ a ser amostrado e quantizado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            Xnrz_prime (np.ndarray): Sinal $X'_{NRZ}[n]$ amostrado e quantizado.
            Yman_prime (np.ndarray): Sinal $Y'_{MAN}[n]$ amostrado e quantizado.
        
        Example:
            - Tempo: ![pageplot](assets/receiver_sampler_time.svg)
            - Constelação: ![pageplot](assets/receiver_sampler_const.svg)  
            - Fase: ![pageplot](assets/receiver_sampler_phase.svg)  
        """ 

        s_sampledI = self.samplerI.sample(It_prime)
        t_sampledI = self.samplerI.sample(t)
        Xi_prime = self.samplerI.quantize(s_sampledI)

        s_sampledQ = self.samplerQ.sample(Qt_prime)
        t_sampledQ = self.samplerQ.sample(t)
        Yq_prime = self.samplerQ.quantize(s_sampledQ)

        if self.output_print:
            print("\n ==== DECISOR ==== \n")
            print("X'i:", ' '.join(f"{x:+d}" for x in Xi_prime[:20]),"...")
            print("Y'q:", ' '.join(f"{y:+d}" for y in Yq_prime[:20]),"...")

        if self.output_plot:
            fig_sampler, grid_sampler = create_figure(2, 1, figsize=(16, 9))

            SampledSignalPlot(
                fig_sampler, grid_sampler, (0, 0),
                t,
                It_prime,
                t_sampledI,
                s_sampledI,
                colors='darkgreen',
                label_signal="Sinal original", 
                label_samples="Amostras", 
                xlim=(80, 240), 
                title="Componente $I$ amostrado"
            ).plot()

            SampledSignalPlot(
                fig_sampler, grid_sampler, (1, 0),
                t,
                Qt_prime,
                t_sampledQ,
                s_sampledQ,
                colors='navy',
                label_signal="Sinal original", 
                label_samples="Amostras", 
                xlim=(80, 240), 
                title="Componente $Q$ amostrado"
            ).plot()

            fig_sampler.tight_layout()
            save_figure(fig_sampler, "receiver_sampler_time.pdf")            

            fig_const, grid_const = create_figure(1, 2, figsize=(16, 9))

            ConstellationPlot(
                fig_const, grid_const, (0, 0),
                dI=It_prime[:40000:5],
                dQ=Qt_prime[:40000:5],
                xlim=(-1.4, 1.4),
                ylim=(-1.4, 1.4),
                title="Constelação $IQ$",
                colors=["darkred"],
                style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot() 

            ConstellationPlot(
                fig_const, grid_const, (0, 1),
                dI=s_sampledI,
                dQ=s_sampledQ,
                xlim=(-1.4, 1.4),
                ylim=(-1.4, 1.4),
                title="Constelação $IQ - Amostrado$",
                colors=["darkred"],
                style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot() 

            fig_const.tight_layout()
            save_figure(fig_const, "receiver_sampler_const.pdf")

            fig_phase, grid_phase = create_figure(1, 2, figsize=(16, 9))

            PhasePlot(
                fig_phase, grid_phase, (0, 0),
                t=t,
                signals=[It_prime, Qt_prime],
                labels=["Fase $I + jQ$"],
                title="Fase $I + jQ$",
                xlim=(40, 320),
                colors=["darkred"],
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            PhasePlot(
                fig_phase, grid_phase, (0, 1),
                t=t_sampledI,
                signals=[np.array(Xi_prime), np.array(Yq_prime)],
                labels=["Fase $I + jQ$"],
                title="Fase $I + jQ$ - Decidido",
                xlim=(40, 320),
                colors=["darkred"],
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            fig_phase.tight_layout()
            save_figure(fig_phase, "receiver_sampler_phase.pdf")

        return Xi_prime, Yq_prime

    def decode(self, Xnrz_prime, Yman_prime):
        r"""
        Decodifica os vetores de simbolos codificados $X'_{NRZ}[n]$ e $Y'_{MAN}[n]$, retornando os vetores de bits $X'n$ e $Y'n$.

        Args:
            Xnrz_prime (np.ndarray): Sinal $X'_{NRZ}[n]$ quantizado.
            Yman_prime (np.ndarray): Sinal $Y'_{MAN}[n]$ quantizado.

        Returns:
            Xn_prime (np.ndarray): Sinal $X'n$ decodificado.
            Yn_prime (np.ndarray): Sinal $Y'n$ decodificado.
        
        Example:
            - Tempo: ![pageplot](assets/receiver_decoder_time.svg)
        """

        i_quantized = np.array(Xnrz_prime)
        q_quantized = np.array(Yman_prime)
        
        Xn_prime = self.decoderI.decode(i_quantized)
        Yn_prime = self.decoderQ.decode(q_quantized)

        if self.output_print:
            print("\n ==== DECODIFICADOR DE LINHA ==== \n")
            print("X'n:", ''.join(map(str, Xn_prime)))
            print("Y'n:", ''.join(map(str, Yn_prime)))
        
        if self.output_plot:
            fig_decoder, grid_decoder = create_figure(4, 1, figsize=(16, 9))

            SymbolsPlot(
                fig_decoder, grid_decoder, (0, 0),
                symbols_list=[Xnrz_prime],
                samples_per_symbol=1,
                colors=["darkgreen"],
                xlabel="Index de Simbolo",
                xlim=(0, 60),
                ylabel="$X_{NRZ}[n]$", 
                label="$X_{NRZ}[n]$"
            ).plot()

            BitsPlot(
                fig_decoder, grid_decoder, (1, 0),
                bits_list=[Xn_prime],
                sections=[("$X_n$", len(Xn_prime))],
                colors=["darkgreen"],
                xlabel="Index de Bit", 
                ylabel="$X_n$", 
                xlim=(0, 60)
            ).plot()

            SymbolsPlot(
                fig_decoder, grid_decoder, (2, 0),
                symbols_list=[Yman_prime],
                samples_per_symbol=1,
                colors=["navy"],
                xlabel="Index de Simbolo",
                xlim=(0, 60),
                ylabel="$Y_{MAN}[n]$", 
                label="$Y_{MAN}[n]$"
            ).plot()

            BitsPlot(
                fig_decoder, grid_decoder, (3, 0),
                bits_list=[Yn_prime],
                sections=[("$Y_n$", len(Yn_prime))],
                colors=["navy"],
                xlabel="Index de Bit", 
                ylabel="$Y_n$", 
                xlim=(0, 60)
            ).plot()

            fig_decoder.tight_layout()
            save_figure(fig_decoder, "receiver_decoder_time.pdf")
                 
        return Xn_prime, Yn_prime

    def descrambler(self, Xn_prime, Yn_prime):
        r"""
        Desembaralha os vetores de bits $X'n$ e $Y'n$, retornando os vetores de bits $v_{t}^{0'}$ e $v_{t}^{1'}$.

        Args:
            Xn_prime (np.ndarray): Vetor de bits $X'n$ embaralhados.
            Yn_prime (np.ndarray): Vetor de bits $Y'n$ embaralhados.

        Returns:
            vt0 (np.ndarray): Vetor de bits $v_{t}^{0'}$ desembaralhado.
            vt1 (np.ndarray): Vetor de bits $v_{t}^{1'}$ desembaralhado.

        Example:
            - Tempo: ![pageplot](assets/receiver_descrambler_time.svg)
        """

        vt0, vt1 = self.unscrambler.descramble(Xn_prime, Yn_prime)

        if self.output_print:
            print("\n ==== DESEMBARALHADOR ==== \n")
            print("vt0':", ''.join(map(str, vt0)))
            print("vt1':", ''.join(map(str, vt1)))
        
        if self.output_plot:
            fig_descrambler, grid_descrambler = create_figure(4, 1, figsize=(16, 9))

            BitsPlot(
                fig_descrambler, grid_descrambler, (0, 0),
                bits_list=[Xn_prime],
                sections=[("$X_n$", len(Xn_prime))],
                colors=["darkgreen"],
                ylabel="Embaralhado",
                xlim=(0, 60)
            ).plot()

            BitsPlot(
                fig_descrambler, grid_descrambler, (1, 0),
                bits_list=[vt0],
                sections=[("$v_t^{0}$", len(vt0))],
                colors=["darkgreen"],
                ylabel="Restaurado", 
                xlim=(0, 60)
            ).plot()

            BitsPlot(
                fig_descrambler, grid_descrambler, (2, 0),
                bits_list=[Yn_prime],
                sections=[("$Y_n$", len(Yn_prime))],
                colors=["navy"],
                ylabel="Embaralhado",
                xlim=(0, 60)
            ).plot()

            BitsPlot(
                fig_descrambler, grid_descrambler, (3, 0),
                bits_list=[vt1],
                sections=[("$v_t^{1}$", len(vt1))],
                colors=["navy"],
                ylabel="Restaurado", 
                xlabel="Index de Bit",
                xlim=(0, 60)
            ).plot()

            fig_descrambler.tight_layout()
            save_figure(fig_descrambler, "receiver_descrambler_time.pdf")     

        return vt0, vt1

    def conv_decoder(self, vt0, vt1):
        r"""
        Decodifica os vetores de bits $v_{t}^{0'}$ e $v_{t}^{1'}$, retornando o vetor de bits $u_{t}'$.

        Args:
            vt0 (np.ndarray): Vetor de bits $v_{t}^{0'}$ desembaralhado.
            vt1 (np.ndarray): Vetor de bits $v_{t}^{1'}$ desembaralhado.

        Returns:
            ut (np.ndarray): Vetor de bits $u_{t}'$ decodificado.
        
        Example:
            - Tempo: ![pageplot](assets/receiver_conv_time.svg)
        """

        ut = self.conv_viterbi.decode(vt0, vt1)

        if self.output_print:
            print("\n ==== DECODIFICADOR VITERBI ==== \n")
            print("u't:", ''.join(map(str, ut)))
        
        if self.output_plot:
            fig_conv_decoder, grid_conv_decoder = create_figure(3, 1, figsize=(16, 9))

            BitsPlot(
                fig_conv_decoder, grid_conv_decoder, (0, 0),
                bits_list=[vt0],
                sections=[("$v_t^{0}$", len(vt0))],
                colors=["darkgreen"],
                ylabel="Canal $I$",
                xlim=(0, 60)
            ).plot()

            BitsPlot(
                fig_conv_decoder, grid_conv_decoder, (1, 0),
                bits_list=[vt1],
                sections=[("$v_t^{1}$", len(vt1))],
                colors=["navy"],
                ylabel="Canal $Q$",
                xlim=(0, 60)
            ).plot()

            BitsPlot(
                fig_conv_decoder, grid_conv_decoder, (2, 0),
                bits_list=[ut],
                sections=[("$u_t'$", len(ut))],
                colors=["darkred"],
                ylabel="Decodificado", 
                xlabel="Index de Bit",
                xlim=(0, 60)
            ).plot()

            fig_conv_decoder.tight_layout()
            save_figure(fig_conv_decoder, "receiver_conv_time.pdf")     

        return ut

    
    def datagram(self, ut):
        r"""
        Recebe um vetor de bits $u_{t}'$ decodificado e retorna um datagrama no padrão ARGOS-3, ou o vetor de bits $u_{t}'$ se houver erro.

        Args:
            ut (np.ndarray): Vetor de bits $u_{t}'$ decodificado.

        Returns:
            datagram (np.ndarray): Datagrama gerado, ou o vetor de bits $u_{t}'$ se houver erro.
            success (bool): Indica se a operação foi bem-sucedida.

        Example:
            - Tempo: ![pageplot](assets/receiver_datagram_time.svg)
        """
        try:
            datagramRX = Datagram(streambits=ut)

            if self.output_print:
                print("\n ==== DATAGRAMA ==== \n")
                print("\n",datagramRX.parse_datagram())

            if self.output_plot:
                fig_datagram, grid = create_figure(1, 1, figsize=(16, 5))
                BitsPlot(
                    fig_datagram, grid, (0, 0),
                    bits_list=[datagramRX.msglength, 
                               datagramRX.pcdid, 
                               datagramRX.blocks, 
                               datagramRX.tail],
                    sections=[("Message Length", len(datagramRX.msglength)),
                              ("PCD ID", len(datagramRX.pcdid)),
                              ("Dados de App.", len(datagramRX.blocks)),
                              ("Tail", len(datagramRX.tail))],
                    colors=["green", "orange", "red", "blue"],
                    xlabel="Index de Bit"
                ).plot()
                fig_datagram.tight_layout()
                save_figure(fig_datagram, "receiver_datagram_time.pdf")
            
            return datagramRX, True

        except Exception as e:
            return ut, False
    
    def receive(self, s):
        r"""
        Executa o processo de recepção, retornando o resultado da recepção.

        Args:
            s (np.ndarray): Sinal $s(t)$ recebido.

        Returns:
            datagramRX (np.ndarray): Datagrama gerado, ou vetor ut se houver erro.
        """

        t = np.arange(0, len(s)/self.fs, 1/self.fs)

        xI_prime, yQ_prime = self.demodulate(s, t)
        dI_prime, dQ_prime= self.lowpassfilter(xI_prime, yQ_prime, t)
        It_prime, Qt_prime = self.matchedfilter(dI_prime, dQ_prime, t)
        self.delayI, self.delayQ = self.synchronizer(It_prime, Qt_prime)

        # Atualiza o delay do sampler
        self.samplerI.update_sampler(self.delayI, t)
        self.samplerQ.update_sampler(self.delayQ, t)

        Xnrz_prime, Yman_prime = self.sampler(It_prime, Qt_prime, t)
        Xn_prime, Yn_prime = self.decode(Xnrz_prime, Yman_prime)
        vt0, vt1 = self.descrambler(Xn_prime, Yn_prime)
        ut = self.conv_decoder(vt0, vt1)

        datagramRX, success = self.datagram(ut)
        return datagramRX, success 


if __name__ == "__main__":

    datagramTX = Datagram(pcdnum=1234, numblocks=1)
    transmitter = Transmitter(fc=4000, output_print=True, output_plot=True)
    t, s = transmitter.transmit(datagramTX)

    # Adicionando ruído ao sinal
    ebn0_db = 20
    add_noise = NoiseEBN0(ebn0_db=ebn0_db, seed=11)
    s = add_noise.add_noise(s)
    
    print("\n ==== CANAL ==== \n")
    print("s(t):", ''.join(map(str, s[:5])), "...")
    print("t:   ", ''.join(map(str, t[:5])), "...")

    receiver = Receiver(fc=4000, output_print=True)
    datagramRX, success = receiver.receive(s)
        
    if not success:
        bitsTX = datagramTX.streambits 
        bitsRX = datagramRX
        print("Bits TX: ", ''.join(str(b) for b in bitsTX))
        print("Bits RX: ", ''.join(str(b) for b in bitsRX))

        # Calcula a Taxa de Erro de Bit (BER)
        num_errors = sum(1 for tx, rx in zip(bitsTX, bitsRX) if tx != rx)
        ber = num_errors / len(bitsTX)

        print(f"Número de erros: {num_errors}")
        print(f"Taxa de Erro de Bit (BER): {ber:.6f}")

