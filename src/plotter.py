import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots 
from typing import Optional, List, Union, Tuple, Dict, Any
import os

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
    def __init__(self,
                 ax: plt.Axes,
                 title: str = "",
                 labels: Optional[List[str]] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 colors: Optional[Union[str, List[str]]] = None,
                 style: Optional[Dict[str, Any]] = None) -> None:
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
        self.ax = ax
        self.title = title
        self.labels = labels
        self.xlim = xlim
        self.ylim = ylim
        self.colors = colors
        self.style = style or {}

    def apply_ax_style(self) -> None:
        r"""
        Aplica o estilo do eixo.
        """
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
        r"""
        Aplica a legenda do plot.
        """
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
        r"""
        Aplica a cor do vetor de dados.
        
        Args:
            idx (int): Índice do vetor de dados
        
        Returns:
            Optional[str]: Cor do vetor de dados
        """
        if self.colors is None:
            return None
        if isinstance(self.colors, str):
            return self.colors
        if isinstance(self.colors, (list, tuple)):
            return self.colors[idx % len(self.colors)]
        return None


class TimePlot(BasePlot):
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 t: np.ndarray,
                 signals: Union[np.ndarray, List[np.ndarray]],
                 **kwargs) -> None:
        r"""
        Classe para plotar sinais no domínio do tempo.

        Args:
            fig (plt.Figure): Figura do plot
            grid (gridspec.GridSpec): GridSpec do plot
            pos (int): Posição do plot
            t (np.ndarray): Vetor de tempo
            signals (Union[np.ndarray, List[np.ndarray]]): Sinal ou lista de sinais
        """
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
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 fs: float,
                 signal: np.ndarray,
                 fc: float = 0.0,
                 **kwargs) -> None:
        r"""
        Classe para plotar sinais no domínio da frequência.
        
        Args:
            fig (plt.Figure): Figura do plot
            grid (gridspec.GridSpec): GridSpec do plot
            pos (int): Posição do plot
            fs (float): Frequência de amostragem
            signal (np.ndarray): Sinal a ser plotado
            fc (float): Frequência central
        """
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
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 dI: np.ndarray,
                 dQ: np.ndarray,
                 amplitude: Optional[float] = None,
                 **kwargs) -> None:
        r"""
        Classe para plotar sinais no domínio da constelação.
        
        Args:
            fig (plt.Figure): Figura do plot
            grid (gridspec.GridSpec): GridSpec do plot
            pos (int): Posição do plot
            dI (np.ndarray): Sinal I
            dQ (np.ndarray): Sinal Q
            amplitude (Optional[float]): Amplitude alvo para pontos ideais
        """
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
                        color="red", s=160, marker="x", label="Pontos Ideais", linewidth=2)

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

    def plot(self, show_bit_values: bool = True, bit_value_offset: float = 0.15, 
             bit_value_size: int = 12, bit_value_weight: str = 'bold') -> None:
        """
        Plota os bits com opção para exibir os valores dos bits acima do plot.
        
        Args:
            show_bit_values (bool): Se True, exibe os valores dos bits acima do plot
            bit_value_offset (float): Deslocamento vertical dos valores dos bits
            bit_value_size (int): Tamanho da fonte dos valores dos bits
            bit_value_weight (str): Peso da fonte dos valores dos bits (ex: 'normal', 'bold')
        """
        all_bits = np.concatenate(self.bits_list)
        bits_up = np.repeat(all_bits, 2)
        x = np.arange(len(bits_up))

        # Ajusta os limites para acomodar os valores dos bits
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

                # Plota a linha dos bits
                self.ax.step(
                    x[bit_start:bit_end],
                    bits_up[bit_start:bit_end],
                    where='post',
                    color=color,
                    linewidth=2.0,
                    label=sec_name if i == 0 or sec_name not in [s[0] for s in self.sections[:i]] else None
                )
                
                # Adiciona os valores dos bits
                if show_bit_values:
                    section_bits = all_bits[start_bit:start_bit + sec_len]
                    for j, bit in enumerate(section_bits):
                        bit_pos = start_bit + j
                        self.ax.text(
                            bit_pos * 2 + 1,  # Posição central do bit
                            1.0 + bit_value_offset,  # Posição vertical acima da linha
                            str(int(bit)),  # Converte para 0 ou 1
                            ha='center',
                            va='bottom',
                            fontsize=bit_value_size,
                            fontweight=bit_value_weight,
                            color='black'
                        )
                
                start_bit += sec_len
        else:
            # Para o caso sem seções
            self.ax.step(x, bits_up, where='post', color='black', linewidth=2.0, label='Bits')
            
            # Adiciona os valores dos bits
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

        self.ax.set_xlabel('Índice do Bit')
        self.ax.set_ylabel('Valor')
        
        # Ajusta o layout para garantir que os valores dos bits não sejam cortados
        plt.tight_layout()
        self.apply_ax_style()