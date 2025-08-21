import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots 
import os
from typing import Optional, List, Union, Tuple, Dict, Any
from collections import defaultdict
from matplotlib.lines import Line2D

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
    Converte a magnitude do sinal para escala logarítmica (dB), normalizada.
    
    Args:
        signal: Array com os dados do sinal
        
    Returns:
        Array com o sinal convertido para dB
    """
    mag = np.abs(signal)
    peak = np.max(mag) if np.max(mag) != 0 else 1.0
    mag = mag / peak
    return 20 * np.log10(mag + 1e-12)


def create_figure(rows: int, cols: int, figsize: Tuple[int, int] = (16, 9)) -> Tuple[plt.Figure, gridspec.GridSpec]:
    r"""
    Cria uma figura com GridSpec e retorna (fig, grid).
    
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

def save_figure(fig: plt.Figure, filename: str, out_dir: str = "../out") -> None:
    r"""
    Salva a figura em <out_dir>/<filename> a partir do diretório onde está este script.
    
    Args:
        fig (plt.Figure): Figura a ser salva
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
    Inicializa o plot.
    
    Args:
        ax (plt.Axes): Eixo do plot
        title (str): Título do plot
        labels (Optional[List[str]]): Lista de rótulos
        xlim (Optional[Tuple[float, float]]): Limites do eixo x
        ylim (Optional[Tuple[float, float]]): Limites do eixo y
        colors (Optional[Union[str, List[str]]]): Cores do plot
        style (Optional[Dict[str, Any]]): Estilo do plot
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
    Classe para plotar sinais no domínio do tempo.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        t (np.ndarray): Vetor de tempo
        signals (Union[np.ndarray, List[np.ndarray]]): Sinal ou lista de sinais

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
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
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

        self.ax.set_xlabel("Tempo (s)")
        self.ax.set_ylabel("Amplitude")
        self.apply_ax_style()


class FrequencyPlot(BasePlot):
    r"""
    Classe para plotar sinais no domínio da frequência.
    
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
            self.ax.set_xlabel("Frequência (kHz)")
        else:
            self.ax.set_xlabel("Frequência (Hz)")

        line_kwargs = {"linewidth": 1, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))

        color = self.apply_color(0)
        label = self.labels[0] if self.labels else None
        if color is not None:
            self.ax.plot(freqs, y, label=label, color=color, **line_kwargs)
        else:
            self.ax.plot(freqs, y, label=label, **line_kwargs)

        self.ax.set_ylabel("Magnitude (dB)")
        if self.ylim is None:
            self.ax.set_ylim(-80, 5)

        self.apply_ax_style()


class ConstellationPlot(BasePlot):
    r"""
    Classe para plotar sinais no domínio da constelação.
    
    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        dI (np.ndarray): Sinal I
        dQ (np.ndarray): Sinal Q
        amplitude (Optional[float]): Amplitude alvo para pontos ideais

    Exemplos:
        - Modulador: ![pageplot](assets/example_modulator_constellation.svg)
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

    def plot(self) -> None:
        # Centraliza os dados em torno do zero
        dI_c = self.dI - np.mean(self.dI)
        dQ_c = self.dQ - np.mean(self.dQ)

        # Define amplitude alvo para pontos ideais
        if self.amplitude is None:
            amp = np.mean(np.abs(np.concatenate([dI_c * 1.1, dQ_c * 1.1])))
        else:
            amp = self.amplitude

        scatter_kwargs = {"s": 10, "alpha": 0.6}
        scatter_kwargs.update(self.style.get("scatter", {}))
        color = self.apply_color(0) or "darkgreen"

        # Amostras IQ
        self.ax.scatter(dI_c, dQ_c, label="Amostras IQ", color=color, **scatter_kwargs)

        # Pontos ideais QPSK
        qpsk_points = np.array([[amp, amp], [amp, -amp], [-amp, amp], [-amp, -amp]])
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
    Classe para plotar bits.
    
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
             label: Optional[str] = None) -> None:

        all_bits = np.concatenate(self.bits_list)
        bits_up = np.repeat(all_bits, 2)
        x = np.arange(len(bits_up))

        # Ajustes de eixo
        y_upper = 1.4 if show_bit_values else 1.2
        self.ax.set_xlim(0, len(bits_up))
        self.ax.set_ylim(-0.2, y_upper)
        self.ax.grid(False)
        self.ax.set_yticks([0, 1])

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
                    section_bits = all_bits[start_bit:start_bit + sec_len]
                    for j, bit in enumerate(section_bits):
                        self.ax.text(
                            (start_bit + j) * 2 + 1,
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
                for i, bit in enumerate(all_bits):
                    self.ax.text(
                        i * 2 + 1,
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
    Classe para plotar bits com codificação de linha.
    
    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        bits (np.ndarray): Vetor de bits
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
             label: Optional[str] = None) -> None:

        if len(self.bits) % 2 != 0:
            raise ValueError("O número de bits deve ser par para codificação em pares.")

        bits_up = np.repeat(self.bits, 2)
        x = np.arange(len(bits_up))

        self.ax.set_xlim(0, len(bits_up))
        self.ax.set_ylim(-0.2, 1.4 if show_pairs else 1.2)
        self.ax.grid(False)
        self.ax.set_yticks([0, 1])

        pair_edges = np.arange(0, len(bits_up) + 1, 4)
        for pos in pair_edges:
            self.ax.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)

        self.ax.step(x, bits_up, where="post",
                     color=self.color, linewidth=2.0,
                     label=label if label else None)

        if show_pairs:
            for i in range(0, len(self.bits), 2):
                pair = f"{self.bits[i]}{self.bits[i+1]}"
                self.ax.text(
                    i * 2 + 2,
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
    Classe para plotar a resposta ao impulso de um filtro.

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
                 impulse_response: np.ndarray,
                 t_unit: str = "ms",
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.t_imp = t_imp
        self.impulse_response = impulse_response
        self.t_unit = t_unit

    def plot(self,
             label: Optional[str] = None,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             xlim: Optional[Tuple[float, float]] = None) -> None:

        if self.t_unit == "ms":
            t_plot = self.t_imp * 1000
            default_xlabel = "Tempo (ms)"
        else:
            t_plot = self.t_imp
            default_xlabel = "Tempo (s)"

        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))

        color = self.apply_color(0) or "red"
        lbl = label if label else (self.labels[0] if self.labels else None)

        self.ax.plot(t_plot, self.impulse_response,
                     color=color, label=lbl, **line_kwargs)

        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        else:
            self.ax.set_xlabel(default_xlabel)

        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        else:
            self.ax.set_ylabel("Amplitude")

        if xlim is not None:
            self.ax.set_xlim(xlim)

        self.apply_ax_style()


class TrellisPlot(BasePlot):
    r"""
    Classe para plotar o diagrama de treliça de um decodificador viterbi.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot no GridSpec
        trellis (dict): Dicionário do treliça. 
                        Formato: {estado: {0: (prox_estado, saída), 1: (prox_estado, saída)}}
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
    Classe para plotar um sinal no domínio do tempo com seus pontos de amostragem.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int ou tuple): Posição no GridSpec
        t_signal (np.ndarray): Vetor de tempo do sinal filtrado
        signal (np.ndarray): Sinal filtrado
        t_samples (np.ndarray): Instantes de amostragem
        samples (np.ndarray): Amostras correspondentes

    Exemplos:
        - Sinal Amostrado: ![pageplot](assets/example_sampler_time.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 t_signal: np.ndarray,
                 signal: np.ndarray,
                 t_samples: np.ndarray,
                 samples: np.ndarray,
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.t_signal = t_signal
        self.signal = signal
        self.t_samples = t_samples
        self.samples = samples

    def plot(self,
             label_signal: Optional[str] = None,
             label_samples: Optional[str] = None,
             xlabel: Optional[str] = "Tempo (s)",
             ylabel: Optional[str] = "Amplitude",
             title: Optional[str] = None,
             x_lim: Optional[float] = None) -> None:

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
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)
        if x_lim:
            self.ax.set_xlim(0, x_lim)

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
    Classe para plotar a fase dos sinais IQ no domínio do tempo.

    Args:
        fig (plt.Figure): Figura do plot
        grid (gridspec.GridSpec): GridSpec do plot
        pos (int): Posição do plot
        t (np.ndarray): Vetor de tempo
        signals (Union[np.ndarray, List[np.ndarray]]): Sinais IQ (I e Q)
        labels (List[str], opcional): Rótulos para os sinais. Se não fornecido, será gerado automaticamente.

    Exemplos:
        - Fase dos sinais IQ
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
        self.ax.set_xlabel("Tempo (s)")
        
        # Usar \pi no rótulo do eixo Y para o LaTeX
        self.ax.set_ylabel(r"Fase ($\pi$ rad)")
        
        # Definir limites de fase entre -π e π
        self.ax.set_ylim([-np.pi, np.pi])
        
        # Formatando os ticks do eixo Y para múltiplos de π
        self.ax.set_yticks(np.linspace(-np.pi, np.pi, 5))
        self.ax.set_yticklabels([f'{-i} $\pi$' if i < 0 else f'{i} $\pi$' for i in np.linspace(-1, 1, 5)])

        self.ax.legend()
        self.apply_ax_style()