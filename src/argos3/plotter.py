# """
# Implementação das operações de plot

# Autor: Arthur Cadore
# Data: 16-08-2025
# """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots 
import os
from typing import Optional, List, Union, Tuple, Dict, Any
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.signal import freqz

# Parâmetros gerais de plotagem
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.transparent"] = True

# Estilização science para os plots
plt.style.use("science")

# Cores e estilos
mpl.rcParams["text.color"] = "black"
mpl.rcParams["axes.labelcolor"] = "black"
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
plt.rcParams["figure.figsize"] = (16, 9)

# Fontes e legendas
plt.rc("font", size=16)
plt.rc("axes", titlesize=22, labelsize=22)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("legend", fontsize=12, frameon=True)
plt.rc("figure", titlesize=22)


def mag2db(signal: np.ndarray) -> np.ndarray:
    r"""
    Converte a magnitude do sinal para escala logarítmica ($dB$). O processo de conversão é dado pela expressão abaixo.

    $$
     dB(x) = 20 \log_{10}\left(\frac{|x|}{x_{peak} + 10^{-12}}\right)
    $$

    Sendo:
        - $x$: Sinal a ser convertido para $dB$.
        - $x_{peak}$: Pico de maior magnitude do sinal.
        - $10^{-12}$: Constante para evitar divisão por zero.
    
    Args:
        signal: Array com os dados do sinal
        
    Returns:
        Array com o sinal convertido para $dB$
    """
    mag = np.abs(signal)
    peak = np.max(mag) if np.max(mag) != 0 else 1.0
    mag = mag / peak
    return 20 * np.log10(mag + 1e-12)

def create_figure(rows: int, cols: int, figsize: Tuple[int, int] = (16, 9)) -> Tuple[plt.Figure, gridspec.GridSpec]:
    r"""
    Cria uma figura com `GridSpec`, retornando o objeto `fig` e `grid` para desenhar os plots.
    
    Args:
        rows (int): Número de linhas do GridSpec
        cols (int): Número de colunas do GridSpec
        figsize (Tuple[int, int]): Tamanho da figura
        
    Returns:
        Tuple[plt.Figure, gridspec.GridSpec]: Tupla com a figura e o GridSpec
    """
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(rows, cols, figure=fig)
    return fig, grid

def save_figure(fig: plt.Figure, filename: str, out_dir: str = "../../out") -> None:
    r"""
    Salva a figura em `<out_dir>/<filename>` a partir do diretório raiz do script. 
    
    Args:
        fig (plt.Figure): Objeto `Figure` do matplotlib
        filename (str): Nome do arquivo de saída
        out_dir (str): Diretório de saída
    
    Raises:
        ValueError: Se o diretório de saída for inválido
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(script_dir, out_dir))
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

class BasePlot:
    r"""
    Classe base para plotagem de gráficos, implementando funcionalidades comuns a todos os plots.
    
    Args:
        ax (plt.Axes): Objeto `Axes` do matplotlib. 
        title (str): Título do plot. 
        labels (Optional[List[str]]): Lista de rótulos para os eixos. 
        xlim (Optional[Tuple[float, float]]): Limites do eixo x `x = [xlim[0], xlim[1]]`. 
        ylim (Optional[Tuple[float, float]]): Limites do eixo y `y = [ylim[0], ylim[1]]`. 
        colors (Optional[Union[str, List[str]]]): Cores do plot. 
        style (Optional[Dict[str, Any]]): Estilo do plot.
    """
    def __init__(self,
                 ax: plt.Axes,
                 title: str = "",
                 labels: Optional[List[str]] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 colors: Optional[Union[str, List[str]]] = None,
                 style: Optional[Dict[str, Any]] = None) -> None:
        self.ax = ax
        self.title = title
        self.labels = labels
        self.xlim = xlim
        self.ylim = ylim
        self.colors = colors
        self.style = style or {}

    # Aplica estilos gerais ao eixo
    def apply_ax_style(self) -> None:
        grid_kwargs = self.style.get("grid", {"alpha": 0.6, "linestyle": "--", "linewidth": 0.5})
        self.ax.grid(True, **grid_kwargs)
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        if self.title:
            self.ax.set_title(self.title)
        self.apply_legend()

    # Aplica legendas
    def apply_legend(self) -> None:
        handles, labels = self.ax.get_legend_handles_labels()
        if not handles:
            return
        leg = self.ax.legend(
            loc="upper right",
            frameon=True,
            edgecolor="black",
            fancybox=True,
            fontsize=self.style.get("legend_fontsize", 12),
        )
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")
        frame.set_alpha(1)

    # Aplica cores
    def apply_color(self, idx: int) -> Optional[str]:
        if self.colors is None:
            return None
        if isinstance(self.colors, str):
            return self.colors
        if isinstance(self.colors, (list, tuple)):
            return self.colors[idx % len(self.colors)]
        return None

class TimePlot(BasePlot):
    r"""
    Classe para plotar sinais no domínio do tempo, recebendo um vetor de tempo $t$, e uma lista de sinais $s(t)$.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        t (np.ndarray): Vetor de tempo
        signals (Union[np.ndarray, List[np.ndarray]]): Sinal ou lista de sinais $s(t)$.
        time_unit (str): Unidade de tempo para plotagem ("ms" por padrão, pode ser "s").
        amp_norm (bool): Normalização do sinal para amplitude máxima

    Example:
        - Modulador: ![pageplot](assets/example_modulator_time.svg)
        - Demodulador: ![pageplot](assets/example_demodulator_time.svg)
        - Adição de AWGN ![pageplot](assets/example_noise_time.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 t: np.ndarray,
                 signals: Union[np.ndarray, List[np.ndarray]],
                 time_unit: str = "ms",
                 amp_norm: bool = False,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)

        self.amp_norm = amp_norm

        # Copia os sinais de entrada para evitar modificações no sinal original
        original_signals = signals if isinstance(signals, (list, tuple)) else [signals]
        self.signals = [sig.copy() for sig in original_signals]

        # Unidade de tempo
        self.time_unit = time_unit.lower()
        if self.time_unit == "ms":
            self.t = t * 1e3
        else:
            self.t = t

        # Sinal ou lista de sinais
        if self.labels is None:
            self.labels = [f"Signal {i+1}" for i in range(len(self.signals))]

    def plot(self) -> None:
        # Normalização
        if self.amp_norm:
            max_val = np.max(np.abs(np.concatenate(self.signals)))
            if max_val > 0:
                f = 1 / max_val
                for i, sig in enumerate(self.signals):
                    self.signals[i] *= f

        # Plotagem
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        for i, sig in enumerate(self.signals):
            color = self.apply_color(i)
            if color is not None:
                self.ax.plot(self.t, sig, label=self.labels[i], color=color, **line_kwargs)
            else:
                self.ax.plot(self.t, sig, label=self.labels[i], **line_kwargs)

        # Labels
        xlabel = r"Tempo ($ms$)" if self.time_unit == "ms" else r"Tempo ($s$)"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(r"Amplitude")
        self.apply_ax_style()

class FrequencyPlot(BasePlot):
    r"""
    Classe para plotar sinais no domínio da frequência, recebendo uma frequência de amostragem $f_s$ e um sinal $s(t)$ e realizando a transformada de Fourier do sinal, conforme a expressão abaixo. 

    $$
    \begin{equation}
        S(f) = \mathcal{F}\{s(t)\}
    \end{equation}
    $$

    Sendo:
        - $S(f)$: Sinal no domínio da frequência.
        - $s(t)$: Sinal no domínio do tempo.
        - $\mathcal{F}$: Transformada de Fourier.
    
    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        fs (float): Frequência de amostragem
        signal (np.ndarray): Sinal a ser plotado
        fc (float): Frequência central

    Example:
        - Modulador: ![pageplot](assets/example_modulator_freq.svg)
        - Demodulador: ![pageplot](assets/example_demodulator_freq.svg)
        - Adição de AWGN ![pageplot](assets/example_noise_freq.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 fs: float,
                 signal: np.ndarray,
                 fc: float = 0.0,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.fs = fs
        self.fc = fc
        self.signal = signal

    def plot(self) -> None:
        # Transformada de Fourier
        freqs = np.fft.fftshift(np.fft.fftfreq(len(self.signal), d=1 / self.fs))
        fft_signal = np.fft.fftshift(np.fft.fft(self.signal))
        y = mag2db(fft_signal)

        # Escala de frequência
        if self.fc > 1000:
            freqs = freqs / 1000
            self.ax.set_xlabel(r"Frequência ($kHz$)")
        else:
            self.ax.set_xlabel(r"Frequência ($Hz$)")

        # Plotagem
        line_kwargs = {"linewidth": 1, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0)
        label = self.labels[0] if self.labels else None
        if color is not None:
            self.ax.plot(freqs, y, label=label, color=color, **line_kwargs)
        else:
            self.ax.plot(freqs, y, label=label, **line_kwargs)

        # Labels
        self.ax.set_ylabel(r"Magnitude ($dB$)")
        if self.ylim is None:
            self.ax.set_ylim(-60, 5)

        self.apply_ax_style()

class ConstellationPlot(BasePlot):
    r"""
    Classe para plotar sinais no domínio da constelação, recebendo os sinais $d_I$ e $d_Q$, realizando o plot em fase $I$ e quadratura $Q$, conforme a expressão abaixo.

    $$
    s(t) = d_I(t) + j d_Q(t)
    $$

    Sendo:
        - $s(t)$: Sinal complexo.
        - $d_I(t)$: Sinal em fase.
        - $d_Q(t)$: Sinal em quadratura.


    O plot de constelação pode ser normalizado por um fator de normalização dado por: 

    $$
    \varphi = \frac{\text{A}}{
          \sqrt{
            \displaystyle \frac{1}{N} 
            \sum_{n=0}^{N-1} \Big( I(n)^2 + Q(n)^2 \Big)
          }
        }
    $$

    Sendo:
        - $\text{A}$: Amplitude desejada, definido como `1`. 
        - $\varphi$: Fator de normalização.
        - $N$: Número de amostras.
        - $I(n)$ e $Q(n)$: Sinais em fase e quadratura.
    
    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        dI (np.ndarray): Sinal I
        dQ (np.ndarray): Sinal Q

    Example:
        - Fase e Constelação: ![pageplot](assets/example_modulator_constellation.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 dI: np.ndarray,
                 dQ: np.ndarray,
                 show_ideal_points: bool = True, 
                 rms_norm: bool = False,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.dI = dI
        self.dQ = dQ
        self.amp = 1
        self.show_ideal_points = show_ideal_points
        self.rms_norm = rms_norm

    def plot(self) -> None:
        # Centraliza os dados em torno do zero
        dI_c, dQ_c = self.dI.copy(), self.dQ.copy()
    
        # Se amp_norm for True, normaliza os sinais usando 1/RMS
        if self.rms_norm:
            max_val = np.sqrt(np.mean(dI_c**2 + dQ_c**2))
            if max_val > 0:
                f = self.amp / max_val
                dI_c *= f
                dQ_c *= f
            lim = 1.2 * self.amp
        else:
            lim = 1.2 * np.max(np.abs(np.concatenate([dI_c, dQ_c])))
    
        # Amostras IQ
        scatter_kwargs = {"s": 20, "alpha": 0.6}
        scatter_kwargs.update(self.style.get("scatter", {}))
        color = self.apply_color(0) or "darkgreen"
        self.ax.scatter(dI_c, dQ_c, label="Amostras IQ", color=color, **scatter_kwargs)
    
        # Pontos ideais QPSK
        qpsk_points = np.array([
            [self.amp, self.amp],
            [self.amp, -self.amp],
            [-self.amp, self.amp],
            [-self.amp, -self.amp]
        ])
        if self.show_ideal_points:
            self.ax.scatter(qpsk_points[:, 0], qpsk_points[:, 1],
                            color="blue", s=160, marker="o",
                            label="Pontos Ideais", linewidth=2)
    
        # Linhas auxiliares
        self.ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        self.ax.axvline(0, color="gray", linestyle="--", alpha=0.5)    
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
    
        # Labels
        self.ax.set_xlabel("In Phase $I$")
        self.ax.set_ylabel("Quadrature $Q$")
        self.apply_ax_style()

class BitsPlot(BasePlot):
    r"""
    Classe para plotar bits, recebendo uma lista de bits $b_t$ e realizando o plot em função do tempo $t$.
    
    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        bits_list (List[np.ndarray]): Lista de bits
        sections (Optional[List[Tuple[str, int]]]): Seções do plot
        colors (Optional[List[str]]): Cores do plot
        show_bit_values (bool): Se `True`, exibe os valores dos bits.
        xlabel (Optional[str]): Label do eixo x.
        ylabel (Optional[str]): Label do eixo y.
        label (Optional[str]): Label do plot.
        xlim (Optional[Tuple[float, float]]): Limites do eixo x.

    Example:
        - Datagrama: ![pageplot](assets/example_datagram_time.svg)
        - Codificador Convolucional: ![pageplot](assets/example_conv_time.svg)
        - Embaralhador: ![pageplot](assets/example_scrambler_time.svg)
        - Preâmbulo: ![pageplot](assets/example_preamble.svg)
        - Multiplexador: ![pageplot](assets/example_mux.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 bits_list: List[np.ndarray],
                 sections: Optional[List[Tuple[str, int]]] = None,
                 colors: Optional[List[str]] = None,
                 show_bit_values: bool = True,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 label: Optional[str] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.bits_list = bits_list
        self.sections = sections
        self.colors = colors
        self.show_bit_values = show_bit_values
        self.xlim = xlim
        self.bit_value_offset = 0.15
        self.bit_value_size = 13
        self.bit_value_weight = 'bold'
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label = label

    def plot(self) -> None:
        # Concatena e superamostra os vetores de bit.
        all_bits = np.concatenate(self.bits_list)
        bits_up = np.repeat(all_bits, 2)
        x = np.arange(len(bits_up))

        # Ajustes de eixo
        y_upper = 1.4 if self.show_bit_values else 1.2
        if self.xlim is not None:
            # Ajusta o limite superior do xlim para xlim*2
            self.xlim = (self.xlim[0], self.xlim[1]*2)
            self.ax.set_xlim(self.xlim)
        else:
            self.ax.set_xlim(0, len(bits_up))
        self.ax.set_ylim(-0.2, y_upper)
        self.ax.grid(False)
        self.ax.set_yticks([0, 1])

        # Linhas auxiliares
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: int(val/2)))
        bit_edges = np.arange(0, len(bits_up) + 1, 2)
        for pos in bit_edges:
            self.ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        # Para cada vetor de bits, desenha uma seção do plot.
        if self.sections:
            start_bit = 0
            for i, (sec_name, sec_len) in enumerate(self.sections):
                bit_start = start_bit * 2
                bit_end = (start_bit + sec_len) * 2
                color = self.colors[i] if self.colors and i < len(self.colors) else 'black'
                if i > 0:
                    bit_start -= 1

                # Desenha a seção do plot.
                self.ax.step(
                    x[bit_start:bit_end],
                    bits_up[bit_start:bit_end],
                    where='post',
                    color=color,
                    linewidth=2.0,
                    label=sec_name if self.label is None else self.label
                )
                
                # Exibe os valores dos bits acima da linha do plot.
                if self.show_bit_values:
                    xmin, xmax = self.ax.get_xlim()
                    section_bits = all_bits[start_bit:start_bit + sec_len]
                    for j, bit in enumerate(section_bits):
                        xpos = (start_bit + j) * 2 + 1
                        if xpos < xmin or xpos > xmax:
                            continue
                        self.ax.text(
                            xpos,
                            1.0 + self.bit_value_offset,
                            str(int(bit)),
                            ha='center',
                            va='bottom',
                            fontsize=self.bit_value_size,
                            fontweight=self.bit_value_weight,
                            color='black'
                        )
                start_bit += sec_len
        else:
            # Desenha a seção do plot.
            self.ax.step(x, bits_up, where='post',
                         color='black', linewidth=2.0,
                         label=self.label if self.label else None)

            # Exibe os valores dos bits acima da linha do plot.
            if self.show_bit_values:
                xmin, xmax = self.ax.get_xlim()
                for i, bit in enumerate(all_bits):
                    xpos = i * 2 + 1
                    if xpos < xmin or xpos > xmax:
                        continue
                    self.ax.text(
                        xpos,
                        1.0 + self.bit_value_offset,
                        str(int(bit)),
                        ha='center',
                        va='bottom',
                        fontsize=self.bit_value_size,
                        fontweight=self.bit_value_weight
                    )

        # Labels
        if self.xlabel:
            self.ax.set_xlabel(self.xlabel)
        if self.ylabel:
            self.ax.set_ylabel(self.ylabel)
        self.apply_ax_style()

class SymbolsPlot(BasePlot):
    r"""
    Classe para plotar simbolos codificados com codificação de linha, recebendo um vetor de simbolos $s[i]$ e realizando o plot em função do index de simbolo $i$.
    
    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        symbols_list (List[np.ndarray]): Lista de símbolos
        samples_per_symbol (int): Número de amostras por símbolo
        sections (Optional[List[Tuple[str, int]]]): Seções do plot
        colors (Optional[List[str]]): Cores do plot
        show_symbol_values (bool): Se `True`, exibe os valores dos símbolos.
        xlabel (Optional[str]): Label do eixo x.
        ylabel (Optional[str]): Label do eixo y.
        label (Optional[str]): Label do plot.
        xlim (Optional[Tuple[float, float]]): Limites do eixo x.

    Example:
        - Codificação de Canal: ![pageplot](assets/example_encoder_time.svg)
    """

    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 symbols_list: List[np.ndarray],
                 samples_per_symbol: int = 1,
                 sections: Optional[List[Tuple[str, int]]] = None,
                 colors: Optional[List[str]] = None,
                 show_symbol_values: bool = True,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 label: Optional[str] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.symbols_list = symbols_list
        self.samples_per_symbol = samples_per_symbol
        self.sections = sections
        self.colors = colors
        self.show_symbol_values = show_symbol_values
        self.xlim = xlim
        self.symbol_value_offset = 0.15
        self.symbol_value_size = 13
        self.symbol_value_weight = 'bold'
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label = label

    def plot(self) -> None:

        # Concatena e superamostra os vetores de símbolos
        all_symbols = np.concatenate(self.symbols_list)
        symbols_up = np.repeat(all_symbols, self.samples_per_symbol)
        x = np.arange(len(symbols_up))

        # Ajustes de eixo
        y_upper = 1.8 if self.show_symbol_values else 1.5
        if self.xlim is not None:
            self.xlim = (self.xlim[0] * self.samples_per_symbol,
                         self.xlim[1] * self.samples_per_symbol)
            self.ax.set_xlim(self.xlim)
        else:
            self.ax.set_xlim(0, len(symbols_up))
        self.ax.set_ylim(-1.5, y_upper)
        self.ax.set_yticks([-1, 0, 1])
        self.ax.grid(False)

        # Linhas verticais marcando início de cada símbolo
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: int(val / self.samples_per_symbol)))
        symbol_edges = np.arange(0, len(symbols_up) + 1, self.samples_per_symbol)
        for pos in symbol_edges:
            self.ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        # Para cada vetor de símbolos, desenha uma seção do plot
        if self.sections:
            start_symbol = 0
            for i, (sec_name, sec_len) in enumerate(self.sections):
                sym_start = start_symbol * self.samples_per_symbol
                sym_end = (start_symbol + sec_len) * self.samples_per_symbol
                color = self.colors[i] if self.colors and i < len(self.colors) else 'black'
                if i > 0:
                    sym_start -= 1

                # Desenha a seção do plot.
                self.ax.step(
                    x[sym_start:sym_end],
                    symbols_up[sym_start:sym_end],
                    where='post',
                    color=color,
                    linewidth=2.0,
                    label=sec_name if self.label is None else self.label
                )

                # Exibe os valores dos símbolos acima da linha do plot.
                if self.show_symbol_values:
                    xmin, xmax = self.ax.get_xlim()
                    section_symbols = all_symbols[start_symbol:start_symbol + sec_len]
                    for j, sym in enumerate(section_symbols):
                        xpos = (start_symbol + j) * self.samples_per_symbol + 0.5 * self.samples_per_symbol
                        if xpos < xmin or xpos > xmax:
                            continue
                        self.ax.text(
                            xpos,
                            1.0 + self.symbol_value_offset,
                            str(int(sym)),
                            ha='center',
                            va='bottom',
                            fontsize=self.symbol_value_size,
                            fontweight=self.symbol_value_weight,
                            color='black'
                        )
                start_symbol += sec_len
        else:

            # Desenha a seção do plot.
            color = self.colors[0] if self.colors else 'black'
            self.ax.step(
                x, symbols_up, where='post',
                color=color, linewidth=2.0,
                label=self.label if self.label else None
            )
            # Exibe os valores dos símbolos acima da linha do plot.
            if self.show_symbol_values:
                xmin, xmax = self.ax.get_xlim()
                for i, sym in enumerate(all_symbols):
                    xpos = i * self.samples_per_symbol + 0.5 * self.samples_per_symbol
                    if xpos < xmin or xpos > xmax:
                        continue
                    self.ax.text(
                        xpos,
                        1.0 + self.symbol_value_offset,
                        str(int(sym)),
                        ha='center',
                        va='bottom',
                        fontsize=self.symbol_value_size,
                        fontweight=self.symbol_value_weight
                    )

        # Labels
        if self.xlabel:
            self.ax.set_xlabel(self.xlabel)
        if self.ylabel:
            self.ax.set_ylabel(self.ylabel)
        self.apply_ax_style()

class ImpulseResponsePlot(BasePlot):
    r"""
    Classe para plotar a resposta ao impulso de um filtro, recebendo um vetor de tempo $t_{imp}$ e realizando o plot em função do tempo $t$.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot no GridSpec
        t_imp (np.ndarray): Vetor de tempo da resposta ao impulso
        impulse_response (np.ndarray): Amostras da resposta ao impulso
        t_unit (str, optional): Unidade de tempo no eixo X ("ms" ou "s"). Default é "ms"
        label (Optional[Union[str, List[str]]]): Label do plot
        xlabel (Optional[str]): Label do eixo x
        ylabel (Optional[str]): Label do eixo y
        xlim (Optional[Tuple[float, float]]): Limites do eixo x
        amp_norm (Optional[bool]): Normaliza a resposta ao impulso para ter amplitude unitária. 

    Example:
        - Resposta ao Impulso RRC: ![pageplot](assets/example_formatter_impulse.svg)
        - Resposta ao Impulso Filtro Passa baixa: ![pageplot](assets/example_lpf_impulse.svg)
        - Resposta ao Impulso RRC Invertido: ![pageplot](assets/example_mf_impulse.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 t_imp: np.ndarray,
                 impulse_response: Union[np.ndarray, List[np.ndarray]],
                 t_unit: str = "ms",
                 label: Optional[Union[str, List[str]]] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 amp_norm: Optional[bool] = False,
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.t_imp = t_imp
        self.label = label
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.amp_norm = amp_norm
        self.t_unit = t_unit

        # Cria lista de respostas ao impulso
        if isinstance(impulse_response, np.ndarray):
            self.impulse_response = [impulse_response]
        else:
            self.impulse_response = impulse_response


    def plot(self) -> None:
        # Unidade de tempo
        if self.t_unit == "ms":
            t_plot = self.t_imp * 1000
            default_xlabel = r"Tempo ($ms$)"
        else:
            t_plot = self.t_imp
            default_xlabel = r"Tempo ($s$)"

        # Label
        if isinstance(self.label, str) or self.label is None:
            labels = [self.label] * len(self.impulse_response)
        else:
            labels = self.label
        self.ax.set_xlabel(self.xlabel if self.xlabel is not None else default_xlabel)
        self.ax.set_ylabel(self.ylabel if self.ylabel is not None else "Amplitude")

        # Plotagem
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        for i, resp in enumerate(self.impulse_response):
            color = self.apply_color(i) or None
            lbl = labels[i] if labels and i < len(labels) else None
            if self.amp_norm:
                resp = resp / np.max(resp)
            self.ax.plot(t_plot, resp, color=color, label=lbl, **line_kwargs)

        # Limites
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        self.apply_ax_style()

class SampledSignalPlot(BasePlot):
    r"""
    Classe para plotar um sinal $s(t)$ amostrado em $t_s$.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int ou tuple): Posição no GridSpec
        t_signal (np.ndarray): Vetor de tempo do sinal filtrado
        signal (np.ndarray): Sinal filtrado
        t_samples (np.ndarray): Instantes de amostragem
        samples (np.ndarray): Amostras correspondentes
        time_unit (str): Unidade de tempo. 
        label_signal (str): Label do sinal filtrado.
        label_samples (str): Label das amostras.
        xlabel (str): Label do eixo x.
        ylabel (str): Label do eixo y.
        title (str): Título do plot.
        xlim (tuple): Limites do eixo x.

    Example:
        ![pageplot](assets/example_sampler_time.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 t_signal: np.ndarray,
                 signal: np.ndarray,
                 t_samples: np.ndarray,
                 samples: np.ndarray,
                 time_unit: str = "ms",
                 label_signal: Optional[str] = None,
                 label_samples: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = "Amplitude",
                 title: Optional[str] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)

        self.time_unit = time_unit.lower()
        self.label_signal = label_signal
        self.label_samples = label_samples
        if xlabel is None:
            xlabel = r"Tempo ($ms$)" if self.time_unit == "ms" else r"Tempo ($s$)"
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xlim = xlim
        self.signal = signal
        self.samples = samples

        # Ajusta unidade de tempo
        if self.time_unit == "ms":
            self.t_signal = t_signal * 1e3
            self.t_samples = t_samples * 1e3
        else:
            self.t_signal = t_signal
            self.t_samples = t_samples

    def plot(self) -> None:
        # Plotagem
        signal_color = self.colors if isinstance(self.colors, str) else "blue"
        self.ax.plot(self.t_signal, self.signal,color=signal_color, label=self.label_signal, linewidth=2)
        self.ax.stem(self.t_samples, self.samples,linefmt="k-", markerfmt="ko", basefmt=" ",label=self.label_samples)

        # Ajuste dos eixos
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlim(self.xlim)
        self.ax.set_title(self.title)
        self.apply_ax_style()
        
        # Legenda
        if self.label_signal or self.label_samples:
            leg = self.ax.legend(loc='upper right', frameon=True, fontsize=12)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_alpha(1.0)

class PhasePlot(BasePlot):
    r"""
    Classe para plotar a fase dos sinais $d_I(t)$ e $d_Q(t)$ no domínio do tempo.

    $$
        s(t) = \arctan\left(\frac{d_Q(t)}{d_I(t)}\right)
    $$

    Sendo: 
        - $s(t)$: Vetor de fases por intervalo de tempo.
        - $d_I(t)$: Componente sinal $d_I(t)$, em fase. 
        - $d_Q(t)$: Componente sinal $d_Q(t)$, em quadratura.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        t (np.ndarray): Vetor de tempo
        signals (Union[np.ndarray, List[np.ndarray]]): Sinais IQ (I e Q)
        time_unit (str): Unidade de tempo para plotagem ("ms" por padrão, pode ser "s").
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos: int,
                 t: np.ndarray,
                 signals: Union[np.ndarray, List[np.ndarray]],
                 time_unit: str = "ms",
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)

        # Unidade de tempo
        self.time_unit = time_unit.lower()
        self.t = t
        if self.time_unit == "ms":
            self.t *= 1e3

        if self.labels is None:
            self.labels = ["Fase IQ"]

        # Garantir que os sinais estão em uma tupla
        if isinstance(signals, (list, tuple)):
            assert len(signals) == 2, "Os sinais devem ser passados como tupla com dois componentes (I, Q)."
            self.I = signals[0]
            self.Q = signals[1]
        else:
            raise ValueError("Os sinais devem ser passados como tupla com dois componentes (I, Q).")

    def plot(self) -> None:
        # Calcula a fase usando atan2
        fase = np.angle(self.I + 1j * self.Q)

        # Plot da fase ao longo do tempo
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0)
        self.ax.plot(self.t, fase, label=self.labels[0], color=color, **line_kwargs)

        # Limite de fase entre pi e -pi
        self.ax.set_ylim([-np.pi*1.1, np.pi*1.1])
        ticks = [0, np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4, -np.pi, np.pi]
        labels = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{4}$", r"$-\frac{\pi}{4}$", r"$-\frac{3\pi}{4}$", r"$-\pi$", r"$\pi$"]
        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(labels)

        # Ajuste dos eixos
        self.ax.set_xlabel(r"Tempo ($ms$)" if self.time_unit == "ms" else r"Tempo ($s$)")
        self.ax.set_ylabel(r"Fase ($rad$)")
        self.ax.legend()
        self.apply_ax_style()

class GaussianNoisePlot(BasePlot):
    r"""
    Classe para plotar a densidade de probabilidade $p(x)$ de uma dada variância $\sigma^2$, seguindo a expressão abaixo. 

    $$
    p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)
    $$

    Sendo: 
        - $p(x)$: Densidade de probabilidade do ruído.
        - $\sigma^2$: Variância do ruído.
        - $x$: Amplitude do ruído.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot no GridSpec
        variance (float): Variância do ruído
        num_points (int): Número de pontos para a curva da gaussiana
        legend (str): Legenda do plot
        xlabel (str): Label do eixo x
        ylabel (str): Label do eixo y
        xlim (Optional[Tuple[float, float]]): Limites do eixo x
        span (int): Span do plot

    Example:
        ![pageplot](assets/example_noise_gaussian_ebn0.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 variance: float,
                 num_points: int = 5000,
                 legend: str = "Ruído AWGN",
                 xlabel: str = "Amplitude",
                 ylabel: str = "Densidade de Probabilidade",
                 xlim: Optional[Tuple[float, float]] = None,
                 span: int = 100,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.variance = variance
        self.num_points = num_points
        self.legend = legend
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.span = span

    def plot(self) -> None:
        # Calculo da pdf
        sigma = np.sqrt(self.variance)
        x = np.linspace(-self.span*sigma, self.span*sigma, self.num_points)
        pdf = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-x**2 / (2*self.variance))

        # Plotagem
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or "darkgreen"
        self.ax.plot(x, pdf, label=self.legend, color=color, **line_kwargs)
       
        # Ajuste dos eixos
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        self.apply_ax_style()

class PoleZeroPlot(BasePlot):
    r"""
    Classe para plotar o diagrama de polos e zeros de uma função de transferência discreta no plano-z.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição no GridSpec
        b (np.ndarray): Coeficientes do numerador da função de transferência
        a (np.ndarray): Coeficientes do denominador da função de transferência

    Example:
        ![pageplot](assets/example_lpf_pz.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 b: np.ndarray,
                 a: np.ndarray,
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.b = b
        self.a = a

    def plot(self) -> None:
        # Calcula zeros e polos
        zeros = np.roots(self.b)
        poles = np.roots(self.a)

        # Plot da circuferência, polos e zeros
        theta = np.linspace(0, 2*np.pi, 512)
        self.ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.6)
        self.ax.scatter(np.real(zeros), np.imag(zeros),marker='o', facecolors='none', edgecolors='blue', s=120, label='Zeros')
        self.ax.scatter(np.real(poles), np.imag(poles),marker='x', color='red',s=120, label='Polos')

        # Eixos
        self.ax.axhline(0, color='black', linewidth=0.8)
        self.ax.axvline(0, color='black', linewidth=0.8)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])

        # Labels
        self.ax.set_xlabel("Parte Real")
        self.ax.set_ylabel("Parte Imaginária")
        self.apply_ax_style()

class FrequencyResponsePlot(BasePlot):
    r"""
    Classe para plotar a resposta em frequência de um filtro a partir de seus coeficientes (b, a). 
    Calcula a transformada de Fourier discreta da resposta ao impulso usando `scipy.signal.freqz`.

    $$
        H(f) = \sum_{n=0}^{N} b_n e^{-j 2 \pi f n} \Big/ \sum_{m=0}^{M} a_m e^{-j 2 \pi f m}
    $$

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot no GridSpec
        b (np.ndarray): Coeficientes do numerador do filtro
        a (np.ndarray): Coeficientes do denominador do filtro
        fs (float): Frequência de amostragem
        f_cut (Optional[float]): Frequência de corte do filtro (Hz)
        xlim (Optional[Tuple[float, float]]): Limites do eixo X (Hz)
        worN (int): Número de pontos para a transformada de Fourier
        show_phase (bool): Se `True`, plota a fase da resposta em frequência
        xlabel (str): Label do eixo X
        ylabel (str): Label do eixo Y

    Example:
        ![pageplot](assets/example_lpf_freq_response.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 b: np.ndarray,
                 a: np.ndarray,
                 fs: float,
                 f_cut: float = None,
                 xlim: tuple = None,
                 worN: int = 1024,
                 show_phase: bool = False,
                 xlabel: str = r"Frequência ($Hz$)",
                 ylabel: str = r"Magnitude ($dB$)",
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.b = b
        self.a = a
        self.fs = fs
        self.f_cut = f_cut
        self.xlim = xlim
        self.worN = worN
        self.show_phase = show_phase
        self.xlabel = xlabel
        self.ylabel = ylabel    

    def plot(self) -> None:
        # calcula resposta em frequência
        w, h = freqz(self.b, self.a, worN=self.worN, fs=self.fs)
        magnitude = mag2db(h)

        # Plotagem
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or "darkorange"
        label = self.labels[0] if self.labels else "$H(f)$"
        self.ax.plot(w, magnitude, color=color, label=label, **line_kwargs)

        # Plotagem da fase
        if self.show_phase:
            ax2 = self.ax.twinx()
            phase = np.unwrap(np.angle(h))
            ax2.plot(w, phase, color="darkorange", linestyle="--", linewidth=1.5, label="Fase")
            ax2.set_ylabel("Fase (rad)")

        # adiciona a barra vertical na frequência de corte
        if self.f_cut is not None:
            self.ax.axvline(self.f_cut, color="red", linestyle="--", linewidth=2, label=f"$f_c$ = {self.f_cut} Hz")

        # Eixos
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(-60, 5)

        # Labels
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.apply_ax_style()

class DetectionFrequencyPlot(BasePlot):
    r"""
    Classe para plotar o espectro de uma sinal recebido, com threshold e frequências detectadas. Recebendo uma frequência de amostragem $f_s$ e um sinal $s(t)$ e realizando a transformada de Fourier do sinal, conforme a expressão abaixo. 

    $$
    \begin{equation}
        S(f) = \mathcal{F}\{s(t)\}
    \end{equation}
    $$

    Sendo:
        - $S(f)$: Sinal no domínio da frequência.
        - $s(t)$: Sinal no domínio do tempo.
        - $\mathcal{F}$: Transformada de Fourier.
    
    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        fs (float): Frequência de amostragem
        signal (np.ndarray): Sinal a ser plotado
        fc (float): Frequência central

    Example: 
        ![pageplot](assets/example_detector_freq.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 fs: float,
                 signal: np.ndarray,
                 threshold: float,
                 fc: float = 0.0,
                 title: str = "",
                 labels: Optional[List[str]] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 colors: Optional[Union[str, List[str]]] = None,
                 style: Optional[Dict[str, Any]] = None,
                 freqs_detected: Optional[List[float]] = None
                 ) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax,
                         title=title,
                         labels=labels,
                         xlim=xlim,
                         ylim=ylim,
                         colors=colors,
                         style=style)

        self.fs = fs
        self.fc = fc
        self.signal = signal
        self.threshold = threshold
        self.freqs_detected = freqs_detected
        self.N = len(self.signal)
        self.U = 1.0
        self.xw = self.signal

    def plot(self) -> None:
        # Transformada de Fourier
        X = np.fft.rfft(self.xw, n=self.N)
        P_bin = (np.abs(X) ** 2) / (self.N * self.U + 1e-20)
        P_db = 10.0 * np.log10(P_bin + 1e-20)
        freqs = np.fft.rfftfreq(self.N, d=1 / self.fs)

        # Ajusta escala para KHz
        freqs_plot = freqs / 1000.0

        # Plotagem
        line_kwargs = {"linewidth": 1.5, "alpha": 0.9}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or "blue"
        label = self.labels[0] if self.labels else "Espectro (P_bin)"
        self.ax.plot(freqs_plot, P_db, label=label, color=color, **line_kwargs)

        # Threshold
        thr_line = self.threshold
        thr_label = f"Threshold = {self.threshold:.2f} dB"
        self.ax.axhline(thr_line, color="blue", linestyle="--", linewidth=2, label=thr_label)

        # Linhas verticais
        if self.freqs_detected is not None:
            for idx, f in enumerate(self.freqs_detected, start=1):
                f_plot = f / 1000.0

                # Calcula o índice do bin mais próximo
                i = np.argmin(np.abs(freqs_plot - f_plot))
                P_bin = P_db[i]

                # Ponto e linha vertical sobre P_bin
                self.ax.plot(f_plot, P_bin, 'o', color='k', markersize=6,
                             label=f"$f_{{{idx}}} = {f_plot:.2f}$ kHz")
                self.ax.axvline(f_plot, color="k", linestyle=":", linewidth=2)

        # Limites
        if self.ylim is None:
            self.ax.set_ylim(np.max(P_db) - 50, np.max(P_db) + 5)

        # Legenda
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

        # Labels
        self.ax.set_xlabel(r"Frequência ($kHz$)")
        self.ax.set_ylabel(r"Magnitude ($dB$)")
        self.apply_ax_style()

class BersnrPlot(BasePlot):
    r"""
    Classe para plotar curvas de BER em função de Eb/N0.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição no GridSpec
        EbN0 (np.ndarray): Vetor de valores Eb/N0 (dB)
        ber_curves (List[np.ndarray]): Lista de curvas BER correspondentes
        labels (List[str]): Rótulos de cada curva
        linestyles (List[str], opcional): Lista com estilos de linha. 
        markers (List[str], opcional): Lista com formatos de marcadores. 
        xlabel (str, opcional): Rótulo do eixo x
        ylabel (str, opcional): Rótulo do eixo y
        logy (bool, opcional): Se deve usar escala logarítmica no eixo y

    Example: 
        - ![pageplot](assets/ber_vs_ebn0.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos: int,
                 EbN0: np.ndarray,
                 ber_curves: List[np.ndarray],
                 linestyles: List[str] = None,
                 markers: List[str] = None,
                 xlabel: str = r"$E_b/N_0$ (dB)",
                 ylabel: str = r"Taxa de Erro de Bit (BER)",
                 logy: bool = True,
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.logy = logy
        self.EbN0 = EbN0

        # Cria as curvas BER
        self.ber_curves = ber_curves if isinstance(ber_curves, (list, tuple)) else [ber_curves]

        # Cria os rótulos
        if self.labels is None:
            self.labels = [f"Curva {i+1}" for i in range(len(self.ber_curves))]
        self.linestyles = linestyles if linestyles is not None else ["-"] * len(self.ber_curves)
        self.markers = markers if markers is not None else ["o"] * len(self.ber_curves)

    def plot(self) -> None:
        # Plotagem
        for i, curve in enumerate(self.ber_curves):
            color = self.apply_color(i)
            label = self.labels[i]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]

            plot_kwargs = {"linewidth": 2, "alpha": 1.0,
                           "linestyle": linestyle,
                           "marker": marker}

            self.ax.plot(self.EbN0, curve, label=label, color=color, **plot_kwargs)

        # Usa escala logarítmica por padrão
        if self.logy:
            self.ax.set_yscale("log")
            self.ax.grid(True, which="both", axis="y", linestyle="--", color="gray", alpha=0.6)

        # Labels
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.apply_ax_style()

class SincronizationPlot(BasePlot):
    r"""
    Classe para plotar um sinal no domínio do tempo com marcações de sincronismo.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int ou tuple): Posição no GridSpec
        t (np.ndarray): Vetor de tempo
        signal (np.ndarray): Sinal no tempo
        sync_start (float): Instante de início da palavra de sincronismo
        sync_end (float): Instante de fim da palavra de sincronismo
        max_corr (float): Instante do pico de correlação
        time_unit (str): Unidade de tempo para plotagem ("ms" por padrão, pode ser "s").

    Example: 
        ![pageplot](assets/example_synchronizer_sync.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 t: np.ndarray,
                 signal: np.ndarray,
                 sync_start: float,
                 sync_end: float,
                 max_corr: float,
                 time_unit: str = "ms",
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)

        # Ajuste de unidade de tempo
        self.time_unit = time_unit.lower()
        if self.time_unit == "ms":
            self.t = t * 1e3
            self.sync_start = sync_start * 1e3
            self.sync_end = sync_end * 1e3
            self.max_corr = max_corr * 1e3
        else:
            self.t = t
            self.sync_start = sync_start
            self.sync_end = sync_end
            self.max_corr = max_corr

        self.signals = [signal]
        if self.labels is None:
            self.labels = ["Signal"]

    def plot(self) -> None:
        # Plotagem
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        for i, sig in enumerate(self.signals):
            color = self.apply_color(i)
            if color is not None:
                self.ax.plot(self.t, sig, label=self.labels[i], color=color, **line_kwargs)
            else:
                self.ax.plot(self.t, sig, label=self.labels[i], **line_kwargs)

        # Periodo de recepção do preambulo
        self.ax.axvspan(self.sync_start, self.sync_end,
                        color="gray", alpha=0.2, label=r"$\Delta \tau$")

        # Linhas verticais de sincronismo
        self.ax.axvline(self.max_corr, color="darkorange", linestyle="--", linewidth=2, label=r"$\tau$")
        self.ax.axvline(self.sync_start, color="red", linestyle="--", linewidth=2, label=r"$\tau +/- (\Delta \tau)/2$")
        self.ax.axvline(self.sync_end, color="red", linestyle="--", linewidth=2)
    
        # Labels
        xlabel = r"Tempo ($ms$)" if self.time_unit == "ms" else r"Tempo ($s$)"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(r"Amplitude")
        self.apply_ax_style()


class CorrelationPlot(BasePlot):
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 corr_vec: np.ndarray,
                 fs: float,
                 xlim_ms: Tuple[float, float],
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        
        # Encontrar o índice de maior correlação
        self.corr_vec = corr_vec
        self.sample_indices = np.arange(len(corr_vec))
        self.fs = fs
        self.max_corr_index = np.argmax(corr_vec)
        self.max_corr_value = corr_vec[self.max_corr_index]

        # Limites em índices de amostra
        self.index_low = int(xlim_ms[0] * 1e-3 * fs)
        self.index_high = int(xlim_ms[1] * 1e-3 * fs)

        # Definir o título e a legenda
        if self.labels is None:
            self.labels = [r"$c[k]$"]

    def plot(self) -> None:
        # Plotagem
        color = self.apply_color(0)
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        self.ax.plot(self.sample_indices, self.corr_vec, label=self.labels[0], color=color, **line_kwargs)

        # Máxima correlação
        self.ax.axvline(self.max_corr_index, color='red', linestyle='--', label=f"$k_{{max}}$ = {self.max_corr_index}")
        self.ax.scatter(self.max_corr_index, self.max_corr_value, color='red', zorder=5)

        # limites
        self.ax.set_xlim(self.index_low, self.index_high)

        # Labels
        self.ax.set_xlabel(r"Índice de Amostra $k$")
        self.ax.set_ylabel(r"Fator de Correlação Normalizado $c[k]$")
        self.ax.legend()
        self.apply_ax_style()
