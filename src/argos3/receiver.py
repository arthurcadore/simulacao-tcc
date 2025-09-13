"""
Implementação de um receptor PTT-A3 com seus componentes.

Autor: Arthur Cadore
Data: 16-08-2025
"""

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
from .plotter import save_figure, create_figure, TimePlot, FrequencyPlot, ImpulseResponsePlot, SampledSignalPlot, BitsPlot, EncodedBitsPlot, PhasePlot, ConstellationPlot, FrequencyResponsePlot, SincronizationPlot, CorrelationPlot

class Receiver:
    def __init__(self, fs=128_000, Rb=400, fc=None, output_print=True, output_plot=True):
        r"""
        Classe que encapsula todo o processo de recepção no padrão ARGOS-3. A estrutura do receptor é representada pelo diagrama de blocos abaixo.

        ![pageplot](../assets/blocos_demodulador.svg)

        Args:
            fs (int): Frequência de amostragem em Hz.
            Rb (int): Taxa de bits em bps.
            output_print (bool): Se `True`, imprime os vetores intermediários no console. 
            output_plot (bool): Se `True`, gera e salva os gráficos dos processos intermediários.

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-2097-CNES (seção 3.1 e 3.2)
        </div>
        """
        self.fs = fs
        self.Rb = Rb
        self.fc = fc
        self.output_print = output_print
        self.output_plot = output_plot

    def demodulate(self, s, t):
        r"""
        Demodula o sinal $s'(t)$ com ruído recebido, recuperando os sinais $x'_{I}(t)$ e $y'_{Q}(t)$.

        Args:
            s (np.ndarray): Sinal $s'(t)$ a ser demodulado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            xI_prime (np.ndarray): Sinal $x'_{I}(t)$ demodulado.
            yQ_prime (np.ndarray): Sinal $y'_{Q}(t)$ demodulado.
        
        Exemplo:
            - Tempo: ![pageplot](assets/receiver_demodulator_time.svg)
            - Frequência: ![pageplot](assets/receiver_demodulator_freq.svg)
        """
        demodulator = Modulator(fc=self.fc, fs=self.fs)
        xI_prime, yQ_prime = demodulator.demodulate(s)

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
    
    def lowpassfilter(self, cut_off, xI_prime, yQ_prime, t):
        r"""
        Aplica o filtro passa-baixa com resposta ao impuslo $h(t)$ aos sinais $x'_{I}(t)$ e $y'_{Q}(t)$, retornando os sinais filtrados $d'_{I}(t)$ e $d'_{Q}(t)$.

        Args:
            cut_off (float): Frequência de corte do filtro.
            xI_prime (np.ndarray): Sinal $x'_{I}(t)$ a ser filtrado.
            yQ_prime (np.ndarray): Sinal $y'_{Q}(t)$ a ser filtrado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            dI_prime (np.ndarray): Sinal $d'_{I}(t)$ filtrado.
            dQ_prime (np.ndarray): Sinal $d'_{Q}(t)$ filtrado.

        Exemplo:
            - Tempo: ![pageplot](assets/receiver_lpf_time.svg)
            - Frequência: ![pageplot](assets/receiver_lpf_freq.svg)
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
            fig_signal, grid_signal = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_signal, grid_signal, (0, slice(0, 2)),
                t_impulse, impulse_response,
                t_unit="ms",
                colors="darkorange",
            ).plot(label=r"$h(t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(0, 8))

            TimePlot(
                fig_signal, grid_signal, (1, 0),
                t, 
                dI_prime,
                labels=[r"$d_I'(t)$"],  
                title=r"Sinal filtrado - Componente $I$",
                xlim=(40, 200),
                ylim=(-1,1),
                colors="darkgreen"
            ).plot()

            TimePlot(
                fig_signal, grid_signal, (1, 1),
                t, 
                dQ_prime,
                labels=[r"$d_Q'(t)$"],
                title=r"Sinal filtrado - Componente $Q$",
                xlim=(40, 200),
                ylim=(-1,1),
                colors="navy"
            ).plot()

            fig_signal.tight_layout()
            save_figure(fig_signal, "receiver_lpf_time.pdf")


            fig_freq, grid_freq = create_figure(3, 2, figsize=(16, 9))

            FrequencyResponsePlot(
                fig_freq, grid_freq, (0, slice(0, 2)),
                lpf.b,
                lpf.a,
                self.fs,
                f_cut=cut_off,
                xlim=(0, 3*cut_off),
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

        Exemplo:
            - Tempo: ![pageplot](assets/receiver_mf_time.svg)
            - Frequência: ![pageplot](assets/receiver_mf_freq.svg)
        """

        matched_filter_I = MatchedFilter(alpha=0.8, fs=self.fs, Rb=self.Rb, span=6, type="RRC-Inverted", channel="I", bits_per_symbol=1)
        matched_filter_Q = MatchedFilter(alpha=0.8, fs=self.fs, Rb=self.Rb, span=6, type="Manchester-Inverted", channel="Q", bits_per_symbol=2)
        It_prime = matched_filter_I.apply_filter(dI_prime)
        Qt_prime = matched_filter_Q.apply_filter(dQ_prime)

        if self.output_print:
            print("\n ==== FILTRAGEM CASADA ==== \n")
            print("I'(t):", ''.join(map(str, It_prime[:5])),"...")
            print("Q'(t):", ''.join(map(str, Qt_prime[:5])),"...")

        if self.output_plot:
            fig_matched, grid_matched = create_figure(2, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_matched, grid_matched, (0, 0),
                matched_filter_I.t_rc, matched_filter_I.g_inverted,
                t_unit="ms",
                colors="darkorange",
            ).plot(label=r"$g(-t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

            ImpulseResponsePlot(
                fig_matched, grid_matched, (0, 1),
                matched_filter_Q.t_rc, matched_filter_Q.g_inverted,
                t_unit="ms",
                colors="darkorange",
            ).plot(label=r"$g(-t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

            TimePlot(
                fig_matched, grid_matched, (1, 0),
                t,
                It_prime,
                labels=[r"$I'(t)$"],
                title=r"Sinal filtrado - Componente $I$",
                xlim=(40, 200),
                colors="darkgreen"
            ).plot()

            TimePlot(
                fig_matched, grid_matched, (1, 1),
                t,
                Qt_prime,
                labels=[r"$Q'(t)$"],
                title=r"Sinal filtrado - Componente $Q$",
                xlim=(40, 200),
                colors="navy"
            ).plot()

            fig_matched.tight_layout()
            save_figure(fig_matched, "receiver_mf_time.pdf")

            fig_matched_freq, grid_matched_freq = create_figure(3, 2, figsize=(16, 9))

            ImpulseResponsePlot(
                fig_matched_freq, grid_matched_freq, (0, 0),
                matched_filter_I.t_rc, matched_filter_I.g_inverted,
                t_unit="ms",
                colors="darkorange",
            ).plot(label=r"$g(-t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

            ImpulseResponsePlot(
                fig_matched_freq, grid_matched_freq, (0, 1),
                matched_filter_Q.t_rc, matched_filter_Q.g_inverted,
                t_unit="ms",
                colors="darkorange",
            ).plot(label=r"$g(-t)$", xlabel=r"Tempo ($ms$)", ylabel="Amplitude", xlim=(-15, 15))

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

        Exemplo:
            Tempo: ![pageplot](assets/receiver_sync_time.svg)
            Módulo Correlação: ![pageplot](assets/receiver_sync_corr.svg)
        """
        synchronizer = Synchronizer(fs=self.fs, Rb=self.Rb)

        delayQ_min, delayQ_max, delayQ, corr_vec = synchronizer.correlation(Qt_prime, "Q")

        # Configurado para sincronização dos canais.
        delayI_min, delayI_max, delayI = delayQ_min, delayQ_max, delayQ

        if self.output_print:
            print("\n ==== SINCRONIZADOR ==== \n")
            print("Delay Min  :", delayQ_min)
            print("Delay Max  :", delayQ_max)
            print("Delay Corr :", delayQ)
        
        if self.output_plot:

            fig_corr, grid_corr = create_figure(1, 1, figsize=(16, 9))
            CorrelationPlot(
                fig_corr, grid_corr, (0, 0),
                corr_vec=corr_vec,  
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

    def sampler(self, It_prime, Qt_prime, t, delayI, delayQ):
        r"""
        Realiza a decisão (amostragem e quantização) dos sinais $I'(t)$ e $Q'(t)$, retornando os vetores de simbolos $X'_{NRZ}[n]$ e $Y'_{MAN}[n]$.

        Args:
            It_prime (np.ndarray): Sinal $I'(t)$ a ser amostrado e quantizado.
            Qt_prime (np.ndarray): Sinal $Q'(t)$ a ser amostrado e quantizado.
            t (np.ndarray): Vetor de tempo.

        Returns:
            Xnrz_prime (np.ndarray): Sinal $X'_{NRZ}[n]$ amostrado e quantizado.
            Yman_prime (np.ndarray): Sinal $Y'_{MAN}[n]$ amostrado e quantizado.
        
        Exemplo:
            - Tempo: ![pageplot](assets/receiver_sampler_time.svg)
            - Constelação: ![pageplot](assets/receiver_sampler_const.svg)  
            - Fase: ![pageplot](assets/receiver_sampler_phase.svg)  
        """ 
        sampler = Sampler(fs=self.fs, Rb=self.Rb, t=t, delay=delayQ)
        i_signal_sampled = sampler.sample(It_prime)
        q_signal_sampled = sampler.sample(Qt_prime)
        t_sampled = sampler.sample(t)

        Xnrz_prime = sampler.quantize(i_signal_sampled)
        Yman_prime = sampler.quantize(q_signal_sampled)

        if self.output_print:
            print("\n ==== DECISOR ==== \n")
            print("X'nrz:", ' '.join(f"{x:+d}" for x in Xnrz_prime[:20]),"...")
            print("Y'man:", ' '.join(f"{y:+d}" for y in Yman_prime[:20]),"...")

        if self.output_plot:
            fig_sampler, grid_sampler = create_figure(2, 1, figsize=(16, 9))

            SampledSignalPlot(
                fig_sampler, grid_sampler, (0, 0),
                t,
                It_prime,
                t_sampled,
                i_signal_sampled,
                colors='darkgreen'
            ).plot(label_signal="Sinal original", label_samples="Amostras", xlim=(80, 240), title="Componente $I$ amostrado")

            SampledSignalPlot(
                fig_sampler, grid_sampler, (1, 0),
                t,
                Qt_prime,
                t_sampled,
                q_signal_sampled,
                colors='navy'
            ).plot(label_signal="Sinal original", label_samples="Amostras", xlim=(80, 240), title="Componente $Q$ amostrado")

            fig_sampler.tight_layout()
            save_figure(fig_sampler, "receiver_sampler_time.pdf")            


            fig_const, grid_const = create_figure(1, 2, figsize=(16, 9))

            ConstellationPlot(
                fig_const, grid_const, (0, 0),
                dI=It_prime[:40000:5],
                dQ=Qt_prime[:40000:5],
                xlim=(-1.1, 1.1),
                ylim=(-1.1, 1.1),
                title="Constelação $IQ$",
                colors=["darkred"],
                style={"line": {"linewidth": 2, "alpha": 1}, "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}}
            ).plot() 

            ConstellationPlot(
                fig_const, grid_const, (0, 1),
                dI=i_signal_sampled,
                dQ=q_signal_sampled,
                xlim=(-1.1, 1.1),
                ylim=(-1.1, 1.1),
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
                xlim=(80, 240),
                ylim=(-np.pi, np.pi),
                colors=["darkred"],
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            PhasePlot(
                fig_phase, grid_phase, (0, 1),
                t=t_sampled,
                signals=[np.array(Xnrz_prime), np.array(Yman_prime)],
                labels=["Fase $I + jQ$"],
                title="Fase $I + jQ$ - Decidido",
                xlim=(80, 240),
                ylim=(-np.pi, np.pi),
                colors=["darkred"],
                style={
                    "line": {"linewidth": 2, "alpha": 1},
                    "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
                }
            ).plot()

            fig_phase.tight_layout()
            save_figure(fig_phase, "receiver_sampler_phase.pdf")

        return Xnrz_prime, Yman_prime

    def decode(self, Xnrz_prime, Yman_prime):
        r"""
        Decodifica os vetores de simbolos codificados $X'_{NRZ}[n]$ e $Y'_{MAN}[n]$, retornando os vetores de bits $X'n$ e $Y'n$.

        Args:
            Xnrz_prime (np.ndarray): Sinal $X'_{NRZ}[n]$ quantizado.
            Yman_prime (np.ndarray): Sinal $Y'_{MAN}[n]$ quantizado.

        Returns:
            Xn_prime (np.ndarray): Sinal $X'n$ decodificado.
            Yn_prime (np.ndarray): Sinal $Y'n$ decodificado.
        
        Exemplo:
            - Tempo: ![pageplot](assets/receiver_decoder_time.svg)
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
        
        if self.output_plot:
            fig_decoder, grid_decoder = create_figure(4, 1, figsize=(16, 9))

            EncodedBitsPlot(
                fig_decoder, grid_decoder, (0, 0),
                bits=Xnrz_prime,
                color='darkgreen',
            ).plot(xlabel="Index de Simbolo", ylabel="$X_{NRZ}[n]$", label="$X_{NRZ}[n]$", xlim=(0, len(Xnrz_prime)/2))

            BitsPlot(
                fig_decoder, grid_decoder, (1, 0),
                bits_list=[Xn_prime],
                sections=[("$X_n$", len(Xn_prime))],
                colors=["darkgreen"]
            ).plot(xlabel="Index de Bit", ylabel="$X_n$", xlim=(0, len(Xn_prime)/2))

            EncodedBitsPlot(
                fig_decoder, grid_decoder, (2, 0),
                bits=Yman_prime,
                color="navy",
            ).plot(xlabel="Index de Simbolo", ylabel="$Y_{MAN}[n]$", label="$Y_{MAN}[n]$", xlim=(0, len(Yman_prime)/2))

            BitsPlot(
                fig_decoder, grid_decoder, (3, 0),
                bits_list=[Yn_prime],
                sections=[("$Y_n$", len(Yn_prime))],
                colors=["navy"]
            ).plot(xlabel="Index de Bit", ylabel="$Y_n$", xlim=(0, len(Yn_prime)/2))

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

        Exemplo:
            - Tempo: ![pageplot](assets/receiver_descrambler_time.svg)
        """
        descrambler = Scrambler()
        vt0, vt1 = descrambler.descramble(Xn_prime, Yn_prime)

        if self.output_print:
            print("\n ==== DESEMBARALHADOR ==== \n")
            print("vt0':", ''.join(map(str, vt0)))
            print("vt1':", ''.join(map(str, vt1)))
        
        if self.output_plot:
            fig_descrambler, grid_descrambler = create_figure(2, 2, figsize=(16, 9))

            BitsPlot(
                fig_descrambler, grid_descrambler, (0, 0),
                bits_list=[Xn_prime],
                sections=[("$X_n$", len(Xn_prime))],
                colors=["darkgreen"]
            ).plot(ylabel="Embaralhado")

            BitsPlot(
                fig_descrambler, grid_descrambler, (0, 1),
                bits_list=[Yn_prime],
                sections=[("$Y_n$", len(Yn_prime))],
                colors=["navy"]
            ).plot()

            BitsPlot(
                fig_descrambler, grid_descrambler, (1, 0),
                bits_list=[vt0],
                sections=[("$v_t^{0}$", len(vt0))],
                colors=["darkgreen"]
            ).plot(ylabel="Restaurado", xlabel="Index de Bit")

            BitsPlot(
                fig_descrambler, grid_descrambler, (1, 1),
                bits_list=[vt1],
                sections=[("$v_t^{1}$", len(vt1))],
                colors=["navy"]
            ).plot(xlabel="Index de Bit")

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
        
        Exemplo:
            - Tempo: ![pageplot](assets/receiver_conv_time.svg)
        """
        conv_decoder = DecoderViterbi()
        ut = conv_decoder.decode(vt0, vt1)

        if self.output_print:
            print("\n ==== DECODIFICADOR VITERBI ==== \n")
            print("u't:", ''.join(map(str, ut)))
        
        if self.output_plot:
            fig_conv_decoder, grid_conv_decoder = create_figure(3, 1, figsize=(16, 9))

            BitsPlot(
                fig_conv_decoder, grid_conv_decoder, (0, 0),
                bits_list=[vt0],
                sections=[("$v_t^{0}$", len(vt0))],
                colors=["darkgreen"]
            ).plot(ylabel="Canal $I$")

            BitsPlot(
                fig_conv_decoder, grid_conv_decoder, (1, 0),
                bits_list=[vt1],
                sections=[("$v_t^{1}$", len(vt1))],
                colors=["navy"]
            ).plot(ylabel="Canal $Q$")

            BitsPlot(
                fig_conv_decoder, grid_conv_decoder, (2, 0),
                bits_list=[ut],
                sections=[("$u_t'$", len(ut))],
                colors=["darkred"]
            ).plot(ylabel="Decodificado", xlabel="Index de Bit")

            fig_conv_decoder.tight_layout()
            save_figure(fig_conv_decoder, "receiver_conv_time.pdf")     

        return ut
    
    def run(self, s, t):
        r"""
        Executa o processo de recepção, retornando o resultado da recepção.

        Args:
            s (np.ndarray): Sinal $s(t)$ recebido.
            t (np.ndarray): Vetor de tempo.
            fc (float): Frequência de portadora.

        Returns:
            ut (np.ndarray): Vetor de bits $u_{t}'$ decodificado.

        Exemplo:
            - Tempo: ![pageplot](assets/transmitter_datagram_time.svg)
        """
        xI_prime, yQ_prime = self.demodulate(s, t)
        dI_prime, dQ_prime= self.lowpassfilter(600, xI_prime, yQ_prime, t)
        It_prime, Qt_prime = self.matchedfilter(dI_prime, dQ_prime, t)
        delayI, delayQ = self.synchronizer(It_prime, Qt_prime)
        Xnrz_prime, Yman_prime = self.sampler(It_prime, Qt_prime, t, delayI, delayQ)
        Xn_prime, Yn_prime = self.decode(Xnrz_prime, Yman_prime)
        vt0, vt1 = self.descrambler(Xn_prime, Yn_prime)
        ut = self.conv_decoder(vt0, vt1)
        return ut 
    


if __name__ == "__main__":

    fc = 4000

    datagramTX = Datagram(pcdnum=1234, numblocks=1)
    bitsTX = datagramTX.streambits  
    transmitter = Transmitter(fc=fc, datagram=datagramTX, output_print=True, output_plot=True)
    t, s = transmitter.run()

    ebn0_db = 20
    add_noise = NoiseEBN0(ebn0_db=ebn0_db, seed=11)
    s_noisy = add_noise.add_noise(s)
    
    # s_noisy = s

    print("\n ==== CANAL ==== \n")
    print("s(t):", ''.join(map(str, s_noisy[:5])), "...")
    print("t:   ", ''.join(map(str, t[:5])), "...")

    receiver = Receiver(fc=fc, output_print=True)
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

