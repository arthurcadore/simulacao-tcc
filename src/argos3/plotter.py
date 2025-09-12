import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots 
import os
from typing import Optional, List, Union, Tuple, Dict, Any
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.signal import freqz

plt.style.use("science")
plt.rcParams["figure.figsize"] = (16, 9)
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
    fig.savefig(save_path, bbox_inches="tight")
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

    def apply_legend(self) -> None:
        handles, labels = self.ax.get_legend_handles_labels()
        if not handles:
            return
        leg = self.ax.legend(
            loc="upper right",
            frameon=True,
            edgecolor="black",
            facecolor="white",
            fancybox=True,
            fontsize=self.style.get("legend_fontsize", 12),
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_alpha(1.0)

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

    Exemplos:
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
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)

        self.time_unit = time_unit.lower()
        if self.time_unit == "ms":
            self.t = t * 1e3
        else:
            self.t = t

        self.signals = signals if isinstance(signals, (list, tuple)) else [signals]
        if self.labels is None:
            self.labels = [f"Signal {i+1}" for i in range(len(self.signals))]

    def plot(self) -> None:
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))

        for i, sig in enumerate(self.signals):
            color = self.apply_color(i)
            if color is not None:
                self.ax.plot(self.t, sig, label=self.labels[i], color=color, **line_kwargs)
            else:
                self.ax.plot(self.t, sig, label=self.labels[i], **line_kwargs)

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

    Exemplos:
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
        freqs = np.fft.fftshift(np.fft.fftfreq(len(self.signal), d=1 / self.fs))
        fft_signal = np.fft.fftshift(np.fft.fft(self.signal))
        y = mag2db(fft_signal)

        if self.fc > 1000:
            freqs = freqs / 1000
            self.ax.set_xlabel(r"Frequência ($kHz$)")
        else:
            self.ax.set_xlabel(r"Frequência ($Hz$)")

        line_kwargs = {"linewidth": 1, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))

        color = self.apply_color(0)
        label = self.labels[0] if self.labels else None
        if color is not None:
            self.ax.plot(freqs, y, label=label, color=color, **line_kwargs)
        else:
            self.ax.plot(freqs, y, label=label, **line_kwargs)

        self.ax.set_ylabel(r"Magnitude ($dB$)")
        if self.ylim is None:
            self.ax.set_ylim(-80, 5)

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
    
    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        dI (np.ndarray): Sinal I
        dQ (np.ndarray): Sinal Q
        amplitude (Optional[float]): Amplitude alvo para pontos ideais

    Exemplos:
        - Fase e Constelação: ![pageplot](assets/example_modulator_constellation.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 dI: np.ndarray,
                 dQ: np.ndarray,
                 amplitude: Optional[float] = None,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.dI = dI
        self.dQ = dQ
        self.amplitude = amplitude

    def plot(self, show_ideal_points: bool = True) -> None:
        # Centraliza os dados em torno do zero
        dI_c, dQ_c = self.dI, self.dQ

        # Define amplitude alvo para pontos ideais
        if self.amplitude is None:
            power = np.mean(dI_c**2 + dQ_c**2)
            amp = np.sqrt(power) / np.sqrt(2)
        else:
            amp = self.amplitude
            # Normaliza as amostras para a amplitude definida
            max_val = np.max(np.sqrt(dI_c**2 + dQ_c**2))
            if max_val > 0:
                scale = amp / max_val
                dI_c *= scale
                dQ_c *= scale

        scatter_kwargs = {"s": 10, "alpha": 0.6}
        scatter_kwargs.update(self.style.get("scatter", {}))
        color = self.apply_color(0) or "darkgreen"

        # Amostras IQ
        self.ax.scatter(dI_c, dQ_c, label="Amostras IQ", color=color, **scatter_kwargs)

        # Pontos ideais QPSK
        qpsk_points = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)]])
        if show_ideal_points:
            self.ax.scatter(qpsk_points[:, 0], qpsk_points[:, 1],
                            color="blue", s=160, marker="o", label="Pontos Ideais", linewidth=2)

        # Linhas auxiliares
        self.ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        self.ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

        # Ajusta limites para manter centro
        lim = 1.2 * amp
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)

        self.ax.set_xlabel("Componente em Fase $I$")
        self.ax.set_ylabel("Componente em Quadratura $Q$")
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

    Exemplos:
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
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.bits_list = bits_list
        self.sections = sections
        self.colors = colors

    def plot(self,
             show_bit_values: bool = True,
             bit_value_offset: float = 0.15,
             bit_value_size: int = 12,
             bit_value_weight: str = 'bold',
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             label: Optional[str] = None,
             xlim: Optional[Tuple[float, float]] = None) -> None:

        all_bits = np.concatenate(self.bits_list)
        bits_up = np.repeat(all_bits, 2)
        x = np.arange(len(bits_up))

        # Ajustes de eixo
        y_upper = 1.4 if show_bit_values else 1.2
        if xlim is not None:
            self.ax.set_xlim(xlim)
        else:
            self.ax.set_xlim(0, len(bits_up))
        self.ax.set_ylim(-0.2, y_upper)
        self.ax.grid(False)
        self.ax.set_yticks([0, 1])

        self.ax.xaxis.set_major_locator(MultipleLocator(8))
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: int(val/2)))

        bit_edges = np.arange(0, len(bits_up) + 1, 2)
        for pos in bit_edges:
            self.ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        if self.sections:
            start_bit = 0
            for i, (sec_name, sec_len) in enumerate(self.sections):
                bit_start = start_bit * 2
                bit_end = (start_bit + sec_len) * 2
                color = self.colors[i] if self.colors and i < len(self.colors) else 'black'
                if i > 0:
                    bit_start -= 1

                self.ax.step(
                    x[bit_start:bit_end],
                    bits_up[bit_start:bit_end],
                    where='post',
                    color=color,
                    linewidth=2.0,
                    label=sec_name if label is None else label
                )
                
                if show_bit_values:
                    xmin, xmax = self.ax.get_xlim()
                    section_bits = all_bits[start_bit:start_bit + sec_len]
                    for j, bit in enumerate(section_bits):
                        xpos = (start_bit + j) * 2 + 1
                        if xpos < xmin or xpos > xmax:
                            continue
                        self.ax.text(
                            xpos,
                            1.0 + bit_value_offset,
                            str(int(bit)),
                            ha='center',
                            va='bottom',
                            fontsize=bit_value_size,
                            fontweight=bit_value_weight,
                            color='black'
                        )
                start_bit += sec_len
        else:
            self.ax.step(x, bits_up, where='post',
                         color='black', linewidth=2.0,
                         label=label if label else None)
            if show_bit_values:
                xmin, xmax = self.ax.get_xlim()
                for i, bit in enumerate(all_bits):
                    xpos = i * 2 + 1
                    if xpos < xmin or xpos > xmax:
                        continue
                    self.ax.text(
                        xpos,
                        1.0 + bit_value_offset,
                        str(int(bit)),
                        ha='center',
                        va='bottom',
                        fontsize=bit_value_size,
                        fontweight=bit_value_weight
                    )

        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)

        plt.tight_layout()
        self.apply_ax_style()

class EncodedBitsPlot(BasePlot):
    r"""
    Classe para plotar sinais codificados com codificação de linha, recebendo um vetor de simbolos $s$ e realizando o plot em função do tempo $t$.
    
    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        symbols (np.ndarray): Vetor de simbolos $s$
        color (str): Cor do plot

    Exemplos:
        - Codificação de Linha: ![pageplot](assets/example_encoder_time.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 bits: np.ndarray,
                 color: str = "black",
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.bits = np.array(bits).astype(int)
        self.color = color

    def plot(self, 
             show_pairs: bool = True,
             pair_value_offset: float = 0.15,
             pair_value_size: int = 12,
             pair_value_weight: str = "bold",
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             label: Optional[str] = None,
             xlim: Optional[Tuple[float, float]] = None) -> None:

        if len(self.bits) % 2 != 0:
            raise ValueError("O número de bits deve ser par para codificação em pares.")

        bits_up = np.repeat(self.bits, 2)
        x = np.arange(len(bits_up))

        if xlim is not None:
            self.ax.set_xlim(xlim)
        else:
            self.ax.set_xlim(0, len(bits_up))

        self.ax.set_ylim(-1.2, 1.6 if show_pairs else 1.2)
        self.ax.grid(False)
        self.ax.set_yticks([-1, 1])
        self.ax.set_yticklabels([r"$-1$", r"$+1$"])

        self.ax.xaxis.set_major_locator(MultipleLocator(8))
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: int(val/2)))

        pair_edges = np.arange(0, len(bits_up) + 1, 2)
        for pos in pair_edges:
            self.ax.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)

        self.ax.step(x, bits_up, where="post",
                     color=self.color, linewidth=2.0,
                     label=label if label else None)

        if show_pairs:
            xmin, xmax = self.ax.get_xlim()
            for i in range(0, len(self.bits), 2):
                xpos = i * 2 + 2
                if xpos < xmin or xpos > xmax:
                    continue
                pair = f"{self.bits[i]:+d}{self.bits[i+1]:+d}"
                self.ax.text(
                    xpos,
                    1.0 + pair_value_offset,
                    pair,
                    ha="center",
                    va="bottom",
                    fontsize=pair_value_size,
                    fontweight=pair_value_weight,
                    color="black"
                )

        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)

        plt.tight_layout()
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

    Exemplos:
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
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.t_imp = t_imp

        if isinstance(impulse_response, np.ndarray):
            self.impulse_response = [impulse_response]
        else:
            self.impulse_response = impulse_response

        self.t_unit = t_unit

    def plot(self,
             label: Optional[Union[str, List[str]]] = None,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             xlim: Optional[Tuple[float, float]] = None) -> None:

        if self.t_unit == "ms":
            t_plot = self.t_imp * 1000
            default_xlabel = r"Tempo ($ms$)"
        else:
            t_plot = self.t_imp
            default_xlabel = r"Tempo ($s$)"

        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))

        if isinstance(label, str) or label is None:
            labels = [label] * len(self.impulse_response)
        else:
            labels = label

        for i, resp in enumerate(self.impulse_response):
            color = self.apply_color(i) or None
            lbl = labels[i] if labels and i < len(labels) else None
            self.ax.plot(t_plot, resp, color=color, label=lbl, **line_kwargs)

        self.ax.set_xlabel(xlabel if xlabel is not None else default_xlabel)
        self.ax.set_ylabel(ylabel if ylabel is not None else "Amplitude")

        if xlim is not None:
            self.ax.set_xlim(xlim)

        self.apply_ax_style()


class TrellisPlot(BasePlot):
    r"""
    Classe para plotar o diagrama de treliça de um decodificador viterbi, recebendo um dicionário de treliça e realizando o plot em função do tempo $t$.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot no GridSpec
        trellis (dict): Dicionário do treliça.
        num_steps (int): Número de passos no tempo
        initial_state (int): Estado inicial

    Exemplos:
        - Treliça Decodificador Viterbi: ![pageplot](assets/example_conv_trellis.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 trellis: dict,
                 num_steps: int = 5,
                 initial_state: int = 0,
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.trellis = trellis
        self.num_steps = num_steps
        self.initial_state = initial_state

    def plot(self,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             show_legend: bool = True) -> None:
        states_per_time = defaultdict(set)
        states_per_time[0].add(self.initial_state)
        branches = []

        for t in range(self.num_steps):
            for state in states_per_time[t]:
                for bit in [0, 1]:
                    next_state, out = self.trellis[state][bit]
                    module = sum(np.abs(out))
                    branches.append((t, state, bit, next_state, module, out))
                    states_per_time[t+1].add(next_state)

        all_states = sorted(set(s for states in states_per_time.values() for s in states))
        state_to_x = {s: i for i, s in enumerate(all_states)}
        num_states = len(all_states)

        # Ajusta tamanho da figura dinamicamente
        self.ax.set_xlim(-0.5, num_states - 0.5)
        self.ax.set_ylim(-0.5, self.num_steps + 0.5)
        self.ax.set_xticks(range(num_states))
        self.ax.set_xticklabels([f"{hex(s)[2:].upper():0>2}" for s in all_states])
        self.ax.set_yticks(range(self.num_steps + 1))

        if xlabel:
            self.ax.set_xlabel(xlabel)
        else:
            self.ax.set_xlabel("Estado")

        if ylabel:
            self.ax.set_ylabel(ylabel)
        else:
            self.ax.set_ylabel("Intervalo de tempo")

        self.ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        self.ax.grid(True, axis='y', linestyle=':', alpha=0.2)
        self.ax.invert_yaxis()

        # Desenha os ramos (transições)
        for t, state, bit, next_state, module, out in branches:
            x = [state_to_x[state], state_to_x[next_state]]
            y = [t, t+1]
            color = 'C0' if bit == 0 else 'C1'
            self.ax.plot(x, y, color=color, lw=2, alpha=0.8)

        # Desenha os nós (estados)
        for t in range(self.num_steps+1):
            for state in states_per_time[t]:
                self.ax.plot(state_to_x[state], t, 'o', color='k', markersize=8)

        # Legenda
        if show_legend:
            legend_elements = [
                Line2D([0], [0], color='C0', lw=2, label='Bit de entrada 0'),
                Line2D([0], [0], color='C1', lw=2, label='Bit de entrada 1')
            ]
            self.ax.legend(handles=legend_elements,
                           loc='upper right', frameon=True, fontsize=12)
        
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
        time_unit (str): Unidade de tempo ("ms" por padrão, pode ser "s").
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
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)

        self.time_unit = time_unit.lower()

        if self.time_unit == "ms":
            self.t_signal = t_signal * 1e3
            self.t_samples = t_samples * 1e3
        else:
            self.t_signal = t_signal
            self.t_samples = t_samples

        self.signal = signal
        self.samples = samples

    def plot(self,
             label_signal: Optional[str] = None,
             label_samples: Optional[str] = None,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = "Amplitude",
             title: Optional[str] = None,
             xlim: Optional[Tuple[float, float]] = None) -> None:

        # Sinal filtrado - usa a cor fornecida ou azul como padrão
        signal_color = self.colors if isinstance(self.colors, str) else "blue"
        self.ax.plot(self.t_signal, self.signal,
                     color=signal_color, label=label_signal, linewidth=2)

        # Amostras - usa preto como padrão para manter contraste
        self.ax.stem(self.t_samples, self.samples,
                     linefmt="k-", markerfmt="ko", basefmt=" ",
                     label=label_samples)

        if title:
            self.title = title
            self.ax.set_title(title)

        # Define o eixo X de acordo com a unidade
        if xlabel is None:
            xlabel = r"Tempo ($ms$)" if self.time_unit == "ms" else r"Tempo ($s$)"
        self.ax.set_xlabel(xlabel)

        if ylabel:
            self.ax.set_ylabel(ylabel)
        if xlim:
            self.ax.set_xlim(xlim)

        # Aplica os estilos da classe base, incluindo a legenda
        self.apply_ax_style()
        
        # Garante que a legenda seja exibida
        if label_signal or label_samples:
            leg = self.ax.legend(loc='upper right', frameon=True, fontsize=12)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_alpha(1.0)


class PhasePlot(BasePlot):
    r"""
    Classe para plotar a fase dos sinais $d_I(t)$ e $d_Q(t)$ no domínio do tempo, conforme a expressão abaixo.

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
        labels (List[str], opcional): Rótulos para os sinais. Se não fornecido, será gerado automaticamente.

    Exemplos:
        - Fase e Constelação: ![pageplot](assets/example_modulator_constellation.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos: int,
                 t: np.ndarray,
                 signals: Union[np.ndarray, List[np.ndarray]],
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.t = t
        
        # Garantir que os sinais estão em uma lista
        if isinstance(signals, (list, tuple)):
            assert len(signals) == 2, "Os sinais devem conter exatamente dois componentes: I e Q."
            self.I = signals[0]
            self.Q = signals[1]
        else:
            raise ValueError("Os sinais devem ser passados como uma lista ou tupla com dois componentes (I, Q).")
        
        if self.labels is None:
            self.labels = ["Fase IQ"]

    def plot(self) -> None:
        # Calcula a fase usando a função atan2
        fase = np.angle(self.I + 1j * self.Q)

        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))

        # Plot da fase ao longo do tempo
        color = self.apply_color(0)
        if color is not None:
            self.ax.plot(self.t, fase, label=self.labels[0], color=color, **line_kwargs)
        else:
            self.ax.plot(self.t, fase, label=self.labels[0], **line_kwargs)

        # Ajuste dos eixos
        self.ax.set_xlabel(r"Tempo ($s$")
        self.ax.set_ylabel(r"Fase ($rad$")

        # Limite de fase entre -π e π
        self.ax.set_ylim([-np.pi, np.pi])

        # Definir ticks em radianos e labels em frações de pi
        ticks = [0, np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4]
        labels = [r"$0\pi$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{4}$", r"$-\frac{\pi}{4}$", r"$-\frac{3\pi}{4}$"]

        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(labels)

        self.ax.legend()
        self.apply_ax_style()


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
        labels (List[str], opcional): Rótulos para os sinais. Se não fornecido, será gerado automaticamente.
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

        self.time_unit = time_unit.lower()
        if self.time_unit == "ms":
            self.t = t * 1e3
        else:
            self.t = t

        # Garantir que os sinais estão em uma lista/tupla de tamanho 2
        if isinstance(signals, (list, tuple)):
            assert len(signals) == 2, "Os sinais devem conter exatamente dois componentes: I e Q."
            self.I = signals[0]
            self.Q = signals[1]
        else:
            raise ValueError("Os sinais devem ser passados como uma lista ou tupla com dois componentes (I, Q).")
        
        if self.labels is None:
            self.labels = ["Fase IQ"]

    def plot(self) -> None:
        # Calcula a fase usando atan2
        fase = np.angle(self.I + 1j * self.Q)

        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))

        # Plot da fase ao longo do tempo
        color = self.apply_color(0)
        if color is not None:
            self.ax.plot(self.t, fase, label=self.labels[0], color=color, **line_kwargs)
        else:
            self.ax.plot(self.t, fase, label=self.labels[0], **line_kwargs)

        # Ajuste dos eixos
        xlabel = r"Tempo ($ms$)" if self.time_unit == "ms" else r"Tempo ($s$)"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(r"Fase ($rad$)")

        # Limite de fase entre -π e π
        self.ax.set_ylim([-np.pi, np.pi])

        # Definir ticks em radianos e labels em frações de pi
        ticks = [0, np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4]
        labels = [r"$0\pi$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{4}$", r"$-\frac{\pi}{4}$", r"$-\frac{3\pi}{4}$"]

        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(labels)

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

    Exemplos:
        ![pageplot](assets/example_noise_gaussian_ebn0.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 variance: float,
                 num_points: int = 1000,
                 legend: str = "Ruído AWGN",
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.variance = variance
        self.num_points = num_points
        self.legend = legend

    def plot(self,
             xlabel: str = "Amplitude",
             ylabel: str = "Densidade de Probabilidade",
             xlim: Optional[Tuple[float, float]] = None) -> None:
        sigma = np.sqrt(self.variance)

        x = np.linspace(-50*sigma, 50*sigma, self.num_points)
        pdf = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-x**2 / (2*self.variance))

        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or "darkgreen"

        self.ax.plot(x, pdf, label=self.legend, color=color, **line_kwargs)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # Aplica xlim customizado, se passado
        if xlim is not None:
            self.ax.set_xlim(xlim)

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

    Exemplos:
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

        # Circunferência unitária
        theta = np.linspace(0, 2*np.pi, 512)
        self.ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.6)

        # Plota zeros (bolinhas) e polos (x)
        self.ax.scatter(np.real(zeros), np.imag(zeros),
                        marker='o', facecolors='none', edgecolors='blue',
                        s=120, label='Zeros')
        self.ax.scatter(np.real(poles), np.imag(poles),
                        marker='x', color='red',
                        s=120, label='Polos')

        # Eixos
        self.ax.axhline(0, color='black', linewidth=0.8)
        self.ax.axvline(0, color='black', linewidth=0.8)

        # Labels e limites
        self.ax.set_xlabel("Parte Real")
        self.ax.set_ylabel("Parte Imaginária")
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])

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

    Exemplos:
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
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.b = b
        self.a = a
        self.fs = fs
        self.f_cut = f_cut
        self.xlim = xlim

    def plot(self,
             worN: int = 1024,
             show_phase: bool = False,
             xlabel: str = r"Frequência ($Hz$)",
             ylabel: str = r"Magnitude ($dB$)") -> None:

        # calcula resposta em frequência
        w, h = freqz(self.b, self.a, worN=worN, fs=self.fs)
        magnitude = mag2db(h)

        # plota magnitude em dB
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or "darkorange"
        label = self.labels[0] if self.labels else "$H(f)$"

        self.ax.plot(w, magnitude, color=color, label=label, **line_kwargs)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_ylim(-80, 5)

        # define limites de frequência se fornecido
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)

        # adiciona a barra vertical na frequência de corte
        if self.f_cut is not None:
            self.ax.axvline(self.f_cut, color="red", linestyle="--", linewidth=2, label=f"$f_c$ = {self.f_cut} Hz")

        if show_phase:
            ax2 = self.ax.twinx()
            phase = np.unwrap(np.angle(h))
            ax2.plot(w, phase, color="darkorange", linestyle="--", linewidth=1.5, label="Fase")
            ax2.set_ylabel("Fase (rad)")

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

    Exemplo: 
        ![pageplot](assets/example_detector_freq.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 fs: float,
                 signal: np.ndarray,
                 threshold: float,
                 threshold_unit: str = "db",
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
        self.threshold_unit = threshold_unit.lower()
        self.freqs_detected = freqs_detected  # Frequências detectadas

    def plot(self) -> None:
        N = len(self.signal)    
        U = 1.0
        xw = self.signal

        X = np.fft.rfft(xw, n=N)
        P_bin = (np.abs(X) ** 2) / (N * U + 1e-20)
        P_db = 10.0 * np.log10(P_bin + 1e-20)
        freqs = np.fft.rfftfreq(N, d=1 / self.fs)

        # Sempre em kHz
        freqs_plot = freqs / 1000.0
        self.ax.set_xlabel(r"Frequência ($kHz$)")
        self.ax.set_ylabel(r"Magnitude ($dB$)")

        line_kwargs = {"linewidth": 1.5, "alpha": 0.9}
        line_kwargs.update(self.style.get("line", {}))

        color = self.apply_color(0) or "blue"
        label = self.labels[0] if self.labels else "Espectro (P_bin)"
        self.ax.plot(freqs_plot, P_db, label=label, color=color, **line_kwargs)

        # Threshold
        if self.threshold_unit == "db":
            thr_line = self.threshold
            thr_label = f"Threshold = {self.threshold:.2f} dB"
        elif self.threshold_unit == "linear":
            thr_line = 10.0 * np.log10(self.threshold + 1e-20)
            thr_label = f"Threshold = {self.threshold:.3g} (→ {thr_line:.2f} dB)"
        else:
            raise ValueError("threshold_unit deve ser 'db' ou 'linear'.")
        self.ax.axhline(thr_line, color="blue", linestyle="--", linewidth=2, label=thr_label)

        # Plotar frequências detectadas em kHz com ponto sobre S(f)
        if self.freqs_detected is not None:
            for idx, f in enumerate(self.freqs_detected, start=1):
                f_plot = f / 1000.0  # kHz
                # índice do bin mais próximo
                i = np.argmin(np.abs(freqs_plot - f_plot))
                P_at_f = P_db[i]

                # ponto sobre S(f)
                self.ax.plot(f_plot, P_at_f, 'o', color='k', markersize=6,
                             label=f"$f_{{{idx}}} = {f_plot:.2f}$ kHz")

                # linha vertical
                self.ax.axvline(f_plot, color="k", linestyle=":", linewidth=2)

        if self.ylim is None:
            self.ax.set_ylim(np.max(P_db) - 50, np.max(P_db) + 5)

        # Evitar repetição de legendas
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

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
        linestyles (List[str], opcional): Lista com estilos de linha (e.g., ["-", "--", "-."])
        markers (List[str], opcional): Lista com formatos de marcadores (e.g., ["o", "s", "d"])
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos: int,
                 EbN0: np.ndarray,
                 ber_curves: List[np.ndarray],
                 linestyles: List[str] = None,
                 markers: List[str] = None,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.EbN0 = EbN0
        self.ber_curves = ber_curves if isinstance(ber_curves, (list, tuple)) else [ber_curves]

        if self.labels is None:
            self.labels = [f"Curva {i+1}" for i in range(len(self.ber_curves))]

        # Estilos personalizados
        self.linestyles = linestyles if linestyles is not None else ["-"] * len(self.ber_curves)
        self.markers = markers if markers is not None else ["o"] * len(self.ber_curves)

    def plot(self,
             xlabel: str = r"$E_b/N_0$ (dB)",
             ylabel: str = "Taxa de Erro de Bit (BER)",
             logy: bool = True) -> None:

        for i, curve in enumerate(self.ber_curves):
            color = self.apply_color(i)
            label = self.labels[i]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]

            plot_kwargs = {"linewidth": 2, "alpha": 1.0,
                           "linestyle": linestyle,
                           "marker": marker}

            if color is not None:
                self.ax.plot(self.EbN0, curve, label=label, color=color, **plot_kwargs)
            else:
                self.ax.plot(self.EbN0, curve, label=label, **plot_kwargs)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        if logy:
            self.ax.set_yscale("log")
            self.ax.grid(True, which="both", axis="y", linestyle="--", color="gray", alpha=0.6)

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

    Exemplo: 
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

        # Sempre armazenar o sinal como lista para seguir padrão do TimePlot
        self.signals = [signal]
        if self.labels is None:
            self.labels = ["Signal"]

    def plot(self) -> None:
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))

        # Plot do sinal
        for i, sig in enumerate(self.signals):
            color = self.apply_color(i)
            if color is not None:
                self.ax.plot(self.t, sig, label=self.labels[i], color=color, **line_kwargs)
            else:
                self.ax.plot(self.t, sig, label=self.labels[i], **line_kwargs)

        # Área de fundo entre início e fim de sincronismo
        self.ax.axvspan(self.sync_start, self.sync_end,
                        color="yellow", alpha=0.2, label=r"$\Delta \tau$")

        # Linhas verticais de sincronismo
        self.ax.axvline(self.max_corr, color="k", linestyle="--", linewidth=2, label=r"$\tau$")
        self.ax.axvline(self.sync_start, color="red", linestyle="--", linewidth=2, label=r"$\tau +/- (\Delta \tau)/2$")
        self.ax.axvline(self.sync_end, color="red", linestyle="--", linewidth=2)
    
        # Labels de eixo
        xlabel = r"Tempo ($ms$)" if self.time_unit == "ms" else r"Tempo ($s$)"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(r"Amplitude")

        # Aplica estilo, limites e legenda
        self.apply_ax_style()


class CorrelationPlot(BasePlot):
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 corr_vec: np.ndarray,
                 fs: float,  # Frequência de amostragem em Hz
                 xlim_ms: Tuple[float, float],  # Limites em milissegundos
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        
        self.corr_vec = corr_vec
        self.sample_indices = np.arange(len(corr_vec))  # Índices das amostras
        self.fs = fs

        # Calcular os índices de amostra para os limites em milissegundos
        self.index_40ms = int(xlim_ms[0] * 1e-3 * fs)  # Convertendo para índice
        self.index_140ms = int(xlim_ms[1] * 1e-3 * fs)  # Convertendo para índice

        # Encontrar o índice de maior correlação
        self.max_corr_index = np.argmax(corr_vec)
        self.max_corr_value = corr_vec[self.max_corr_index]

        # Definir o título e a legenda
        if self.labels is None:
            self.labels = [r"$c[k]$"]

    def plot(self) -> None:
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))

        # Plot do vetor de correlação em função do índice k
        color = self.apply_color(0)
        self.ax.plot(self.sample_indices, self.corr_vec, label=self.labels[0], color=color, **line_kwargs)

        # Adicionar marcação para o valor de maior correlação
        self.ax.axvline(self.max_corr_index, color='red', linestyle='--', label=f"$k_{{max}}$ = {self.max_corr_index}")
        self.ax.scatter(self.max_corr_index, self.max_corr_value, color='red', zorder=5)

        # Ajustes dos eixos
        self.ax.set_xlabel(r"Índice de Amostra $k$")
        self.ax.set_ylabel(r"Fator de Correlação Normalizado $c[k]$")
        
        # Definir os limites de x com base nos índices calculados
        self.ax.set_xlim(self.index_40ms, self.index_140ms)
        
        # Legenda
        self.ax.legend()

        # Aplica os estilos da classe base
        self.apply_ax_style()
