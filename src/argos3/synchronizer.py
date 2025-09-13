"""
Implementa um formatador de pulso para transmissão de sinais digitais. 

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
import matplotlib.pyplot as plt

from .preamble import Preamble
from .formatter import Formatter
from .encoder import Encoder
from .plotter import create_figure, save_figure, TimePlot, SincronizationPlot, CorrelationPlot
from .multiplexer import Multiplexer
from .matchedfilter import MatchedFilter

class Synchronizer:
    def __init__(self, fs=128_000, Rb=400, sync_word="2BEEEEBF"):
        r"""
         Inicializa o sincronizador de simbolos para identificar o momento de maior correlação entre o sinal recebido e o sinal de sincronismo.

        Args:
            fs (int): Frequência de amostragem do sinal recebido.
            Rb (int): Taxa de transmissão do sinal recebido.
            sync_word (str): Palavra de sincronismo.

        Exemplo: 
            ![pageplot](assets/example_synchronizer_sync.svg)
        """
        self.fs = fs
        self.Rb = Rb
        self.Tb = 1 / Rb
        self.sps = int(fs / Rb)
        self.encoder_I = Encoder(method="NRZ")
        self.encoder_Q = Encoder(method="nrz2")
        self.formatterI = Formatter(alpha=0.8, fs=self.fs, Rb=self.Rb, span=6, type="RRC", channel="I", bits_per_symbol=1)
        self.formatterQ = Formatter(alpha=0.8, fs=self.fs, Rb=self.Rb, span=6, type="Manchester", channel="Q", bits_per_symbol=2)
        self.matched_filter_I = MatchedFilter(alpha=0.8, fs=self.fs, Rb=self.Rb, span=6, type="RRC-Inverted", channel="I", bits_per_symbol=1)
        self.matched_filter_Q = MatchedFilter(alpha=0.8, fs=self.fs, Rb=self.Rb, span=6, type="Manchester-Inverted", channel="Q", bits_per_symbol=2)
        self.create_sincronized_word(sync_word)

    def create_sincronized_word(self, sync_word):
        r"""
        Monta os vetores de simbolo $S_I(t)$ e $S_Q(t)$, correspondente a palavra de sincronismo do canal $I$ e $Q$, respectivamente. O comprimento da palavra de sincronismo é dado por $\Delta \tau$, conforme a expressão abaixo.

        $$
        \Delta \tau = L_{sync} \cdot \frac{f_s}{R_b}
        $$

        Sendo: 
            - $\Delta \tau$ é o comprimento da palavra de sincronismo.
            - $L_{sync}$ é o comprimento da palavra de sincronismo de $S_I(t)$ e $S_Q(t)$.
            - $R_b$ é a taxa de bits.
            - $f_s$ é a frequência de amostragem.

        Args:
            sync_word (str): Palavra de sincronismo.

        Exemplo: 
            ![pageplot](assets/example_synchronizer_word.svg)
        """

        self.preamble = Preamble(sync_word)
        self.preamble_sI = self.preamble.preamble_sI
        self.preamble_sQ = self.preamble.preamble_sQ

        self.sincronized_word_I = self.formatterI.apply_format(self.encoder_I.encode(self.preamble_sI), add_prefix=False)
        self.sincronized_word_Q = self.formatterQ.apply_format(self.encoder_Q.encode(self.preamble_sQ), add_prefix=False)

        self.sincronized_word_I = self.matched_filter_I.apply_filter(self.sincronized_word_I)
        self.sincronized_word_Q = self.matched_filter_Q.apply_filter(self.sincronized_word_Q)

    def correlation(self, signal, channel):
        r"""

        Realiza a correlação cruzada entre o sinal recebido $s(t)$ e a palavra de sincronismo $d(t)$, para cada index de tempo $t$.

        $$
        c[k] = \sum_{t=0} s[t] d[t - k]
        $$ 

        Sendo: 
            - $s(t)$ e $d(t)$ são os vetores de simbolos do sinal recebido e da palavra de sincronismo, respectivamente.
            - $k$ é o index de tempo no vetor de correlação cruzada.
            - $c[k]$ é o valor da correlação cruzada para o index $k$.

        Em seguida localiza-se o indice de $c[k]$ com maior valor, resultando em $k_{max}$, este é o indice de amostra com a maior correlação entre o sinal recebido e a palavra de sincronismo, por fim, calcula-se o delay $\tau$. 

        $$
            \tau = \frac{k_{max}}{f_s}
        $$

        Sendo: 
            - $\tau$: Delay entre o sinal recebido e a palavra de sincronismo.
            - $f_s$: Frequência de amostragem do sinal recebido.
            - $k_{max}$: Indice de amostra com a maior correlação entre o sinal recebido e a palavra de sincronismo.

        Args:
            signal (np.ndarray): Sinal recebido.
            channel (str): Canal de recebimento, $I$ ou $Q$.

        Returns:
           delay (tuple): Tupla contendo o delay $\tau$, o delay $\tau_{min}$ e o delay $\tau_{max}$.
        
        Exemplo: 
            ![pageplot](assets/example_synchronizer_corr.svg)
        """
        if channel == "I":
            correlation_vec = np.correlate(signal, self.sincronized_word_I, mode="same")
            max_correlation_index = correlation_vec.argmax()
        elif channel == "Q":
            correlation_vec = np.correlate(signal, self.sincronized_word_Q, mode="same")
            max_correlation_index = correlation_vec.argmax()
        else:
            raise ValueError("Canal inválido. Use 'I' ou 'Q'.")
        
        # normaliza o vetor
        correlation_vec = (correlation_vec - correlation_vec.min()) / (correlation_vec.max() - correlation_vec.min())
        
        # calcula o index do início e fim da palavra de sincronismo
        low_index = max_correlation_index - len(self.sincronized_word_I) // 2
        high_index = max_correlation_index + len(self.sincronized_word_I) // 2

        # calcula o delay com base no index: 
        low_delay = low_index / self.fs
        high_delay = high_index / self.fs
        delay = max_correlation_index / self.fs
        return low_delay, high_delay, delay, correlation_vec

if __name__ == "__main__":
    
    synchronizer = Synchronizer()

    fig_format, grid_format = create_figure(2,1, figsize=(16, 9))

    TimePlot(
        fig_format, grid_format, (0,0),
        t= np.arange(len(synchronizer.sincronized_word_I)) / synchronizer.formatterI.fs,
        signals=[synchronizer.sincronized_word_I],
        labels=[r"$S_I(t)$"],
        title=r"Canal $I$",
        colors="darkgreen",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    TimePlot(
        fig_format, grid_format, (1,0),
        t= np.arange(len(synchronizer.sincronized_word_Q)) / synchronizer.formatterQ.fs,
        signals=[synchronizer.sincronized_word_Q],
        labels=[r"$S_Q(t)$"],
        title=r"Canal $Q$",
        colors="darkblue",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    fig_format.tight_layout()
    save_figure(fig_format, "example_synchronizer_word.pdf")

    # TESTE
    preamble = Preamble()  
    SI = preamble.preamble_sI
    SQ = preamble.preamble_sQ
    X = np.random.randint(0, 2, 60)
    Y = np.random.randint(0, 2, 60)
    
    mux = Multiplexer()
    Xn, Yn = mux.concatenate(SI, SQ, X, Y)

    encoder_nrz = Encoder(method="NRZ")
    encoder_man = Encoder(method="NRZ2")
    
    Xnrz = encoder_nrz.encode(Xn)
    Yman = encoder_man.encode(Yn)
    
    formatterI = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC", channel="I", bits_per_symbol=1)
    formatterQ = Formatter(alpha=0.8, fs=128_000, Rb=400, span=6, type="Manchester", channel="Q", bits_per_symbol=2)
    
    dI = formatterI.apply_format(Xnrz)
    dQ = formatterQ.apply_format(Yman)
    
    # Faz a sincronização apenas no canal Q, pois o canal I é apenas uns.
    delayQ_min, delayQ_max, delayQ, corr_vec = synchronizer.correlation(dQ, "Q")
    delayI_min, delayI_max, delayI = delayQ_min, delayQ_max, delayQ

    print("Delay I (ms):", delayI_min)
    print("Delay Q (ms):", delayQ_min)

    fig_sync, grid_sync = create_figure(2,1, figsize=(16, 9))
    
    SincronizationPlot(
        fig_sync, grid_sync, (0,0),
        t= np.arange(len(dI)) / formatterI.fs,
        signal=dI,
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
        t=np.arange(len(dQ)) / formatterQ.fs,
        signal=dQ,
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
    save_figure(fig_sync, "example_synchronizer_sync.pdf")

    fig_corr, grid_corr = create_figure(1, 1, figsize=(16, 9))
    CorrelationPlot(
        fig_corr, grid_corr, (0, 0),
        corr_vec=corr_vec,  
        fs=formatterQ.fs,
        xlim_ms=(40, 200),
        colors="darkblue",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        },
    ).plot()
    fig_corr.tight_layout()
    save_figure(fig_corr, "example_synchronizer_corr.pdf")
    
    
    
        
