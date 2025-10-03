# """
# Implementation of plot operations.
# 
# Author: Arthur Cadore
# Date: 16-08-2025
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
from mpl_toolkits.mplot3d import Axes3D 
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.signal import freqz

from .env_vars import *

# General plot parameters
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.transparent"] = True

# Science plot style
plt.style.use("science")

# Colors and styles
mpl.rcParams["text.color"] = "black"
mpl.rcParams["axes.labelcolor"] = "black"
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
plt.rcParams["figure.figsize"] = (16, 9)

# Fonts
plt.rc("font", size=16)
plt.rc("axes", titlesize=22, labelsize=22)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("legend", fontsize=12, frameon=True)
plt.rc("figure", titlesize=22)

def mag2db(signal: np.ndarray) -> np.ndarray:
    r"""
    Converts the signal magnitude to a logarithmic scale ($dB$). The conversion process is given by the expression below.

    $$
     dB(x) = 20 \log_{10}\left(\frac{|x|}{x_{peak} + 10^{-12}}\right)
    $$

    Where:
        - $x$: Signal to be converted to $dB$.
        - $x_{peak}$: Peak magnitude of the signal.
        - $10^{-12}$: Constant to avoid division by zero.
    
    Args:
        signal: Array with signal data
        
    Returns:
        Array with signal converted to $dB$
    """
    mag = np.abs(signal)
    peak = np.max(mag) if np.max(mag) != 0 else 1.0
    mag = mag / peak
    return 20 * np.log10(mag + 1e-12)

def create_figure(rows: int, cols: int, figsize: Tuple[int, int] = (16, 9)) -> Tuple[plt.Figure, gridspec.GridSpec]:
    r"""
    Creates a figure with `GridSpec`, returning the `fig` and `grid` objects for plotting.
    
    Args:
        rows (int): Number of rows in the GridSpec
        cols (int): Number of columns in the GridSpec
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        Tuple[plt.Figure, gridspec.GridSpec]: Tuple with the figure and GridSpec objects
    """
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(rows, cols, figure=fig)
    return fig, grid

def save_figure(fig: plt.Figure, filename: str, out_dir: str = "../../out") -> None:
    r"""
    Saves the figure in `<out_dir>/<filename>` from the script root directory. 
    
    Args:
        fig (plt.Figure): Matplotlib `Figure` object
        filename (str): Output file name
        out_dir (str): Output directory
    
    Raises:
        ValueError: If the output directory is invalid
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
    Base class for plotting graphs, implementing common functionality for all plots.
    
    Args:
        ax (plt.Axes): Matplotlib `Axes` object. 
        title (str): Plot title. 
        labels (Optional[List[str]]): List of axis labels. 
        xlim (Optional[Tuple[float, float]]): Limits of the x-axis `x = [xlim[0], xlim[1]]`. 
        ylim (Optional[Tuple[float, float]]): Limits of the y-axis `y = [ylim[0], ylim[1]]`. 
        colors (Optional[Union[str, List[str]]]): Plot colors. 
        style (Optional[Dict[str, Any]]): Plot style.
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

    # Apply general styles to the axis
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

    # Apply legends
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

    # Apply colors
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
    Class for plotting signals in the time domain, receiving a time vector $t$, and a list of signals $s(t)$.

    Args:
        fig (plt.Figure): Figure object
        grid (gridspec.GridSpec): GridSpec object
        pos (int): Plot position
        t (np.ndarray): Time vector
        signals (Union[np.ndarray, List[np.ndarray]]): Signal or list of signals $s(t)$.
        time_unit (str): Time unit for plotting ("ms" by default, can be "s").
        amp_norm (bool): Signal normalization for maximum amplitude

    Examples:
        - Modulator Time Domain Example: ![pageplot](assets/example_modulator_time.svg)
        - AWGN addition Time Domain Example: ![pageplot](assets/example_noise_time_ebn0.svg)
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

        # Copy the input signals to avoid modifying the original signal
        original_signals = signals if isinstance(signals, (list, tuple)) else [signals]
        self.signals = [sig.copy() for sig in original_signals]

        # Time unit
        self.time_unit = time_unit.lower()
        if self.time_unit == "ms":
            self.t = t * 1e3
        else:
            self.t = t

        # Signal or list of signals
        if self.labels is None:
            self.labels = [f"Signal {i+1}" for i in range(len(self.signals))]

    def plot(self) -> None:
        # Normalization
        if self.amp_norm:
            max_val = np.max(np.abs(np.concatenate(self.signals)))
            if max_val > 0:
                f = 1 / max_val
                for i, sig in enumerate(self.signals):
                    self.signals[i] *= f

        # Plot
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        for i, sig in enumerate(self.signals):
            color = self.apply_color(i)
            if color is not None:
                self.ax.plot(self.t, sig, label=self.labels[i], color=color, **line_kwargs)
            else:
                self.ax.plot(self.t, sig, label=self.labels[i], **line_kwargs)

        # Labels
        xlabel = r"Time ($ms$)" if self.time_unit == "ms" else r"Time ($s$)"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(r"Amplitude")
        self.apply_ax_style()

class FrequencyPlot(BasePlot):
    r"""
    Class for plotting signals in the frequency domain, receiving a sampling frequency $f_s$ and a signal $s(t)$ and performing the Fourier transform of the signal, according to the expression below. 

    $$
    \begin{equation}
        S(f) = \mathcal{F}\{s(t)\}
    \end{equation}
    $$

    Where:
        - $S(f)$: Signal in the frequency domain.
        - $s(t)$: Signal in the time domain.
        - $\mathcal{F}$: Fourier transform.
    
    Args:
        fig (plt.Figure): Figure object
        grid (gridspec.GridSpec): GridSpec object
        pos (int): Plot position
        fs (float): Sampling frequency
        signal (np.ndarray): Signal to be plotted
        fc (float): Central frequency

    Examples:
        - Modulator Frequency Domain Example: ![pageplot](assets/example_modulator_freq.svg)
        - AWGN addition Frequency Domain Example: ![pageplot](assets/example_noise_freq_ebn0.svg)
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
        # Fourier transform
        freqs = np.fft.fftshift(np.fft.fftfreq(len(self.signal), d=1 / self.fs))
        fft_signal = np.fft.fftshift(np.fft.fft(self.signal))
        y = mag2db(fft_signal)

        # Frequency scale
        if self.fc > 1000:
            freqs = freqs / 1000
            self.ax.set_xlabel(r"Frequency ($kHz$)")
        else:
            self.ax.set_xlabel(r"Frequency ($Hz$)")

        # Plot
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
    Class for plotting signals in the constellation domain, receiving the signals $d_I$ and $d_Q$, performing the plot in phase $I$ and quadrature $Q$, according to the expression below.

    $$
    s(t) = d_I(t) + j d_Q(t)
    $$

    Where:
        - $s(t)$: Complex signal.
        - $d_I(t)$: In-phase signal.
        - $d_Q(t)$: Quadrature signal.


    The constellation plot can be normalized by a normalization factor given by: 

    $$
    \varphi = \frac{\text{A}}{
          \sqrt{
            \displaystyle \frac{1}{N} 
            \sum_{n=0}^{N-1} \Big( I(n)^2 + Q(n)^2 \Big)
          }
        }
    $$

    Where:
        - $\text{A}$: Desired amplitude, defined as `1`. 
        - $\varphi$: Normalization factor.
        - $N$: Number of samples.
        - $I(n)$ and $Q(n)$: In-phase and quadrature signals.
    
    Args:
        fig (plt.Figure): Figure object
        grid (gridspec.GridSpec): GridSpec object
        pos (int): Plot position
        dI (np.ndarray): In-phase signal
        dQ (np.ndarray): Quadrature signal

    Examples:
        - Modulator Constellation/Phase Example: ![pageplot](assets/example_modulator_constellation.svg)
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
        # Centralize the data around zero
        dI_c, dQ_c = self.dI.copy(), self.dQ.copy()
    
        # If amp_norm is True, normalize the signals using 1/RMS
        if self.rms_norm:
            max_val = np.sqrt(np.mean(dI_c**2 + dQ_c**2))
            if max_val > 0:
                f = self.amp / max_val
                dI_c *= f
                dQ_c *= f
            lim = 1.2 * self.amp
        else:
            lim = 1.2 * np.max(np.abs(np.concatenate([dI_c, dQ_c])))
    
        # IQ samples
        scatter_kwargs = {"s": 20, "alpha": 0.6}
        scatter_kwargs.update(self.style.get("scatter", {}))
        color = self.apply_color(0) or "darkgreen"
        self.ax.scatter(dI_c, dQ_c, label="$IQ$ samples", color=color, **scatter_kwargs)
    
        # QPSK ideal points
        qpsk_points = np.array([
            [self.amp, self.amp],
            [self.amp, -self.amp],
            [-self.amp, self.amp],
            [-self.amp, -self.amp]
        ])
        if self.show_ideal_points:
            self.ax.scatter(qpsk_points[:, 0], qpsk_points[:, 1],
                            color=QPSK_IDEAL_COLOR, s=160, marker="D",
                            label="$QPSK$ Ideal", linewidth=2)
    
        # Auxiliary lines
        self.ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        self.ax.axvline(0, color="gray", linestyle="--", alpha=0.5)    
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
    
        # Labels
        self.ax.set_xlabel("In Phase ($I$)")
        self.ax.set_ylabel("Quadrature ($Q$)")
        self.apply_ax_style()

class BitsPlot(BasePlot):
    r"""
    Class for plotting bits, receiving a list of bits $b_t$ and performing the plot in function of time $t$.
    
    Args:
        fig (plt.Figure): Figure object
        grid (gridspec.GridSpec): GridSpec object
        pos (int): Plot position
        bits_list (List[np.ndarray]): List of bits
        sections (Optional[List[Tuple[str, int]]]): Plot sections
        colors (Optional[List[str]]): Plot colors
        show_bit_values (bool): If `True`, shows the bit values.
        xlabel (Optional[str]): X-axis label.
        ylabel (Optional[str]): Y-axis label.
        label (Optional[str]): Label of the plot.
        xlim (Optional[Tuple[float, float]]): X-axis limits.

    Examples:
        - Datagram Bitstream Example: ![pageplot](assets/example_datagram_time.svg)
        - Convolutional Bitstream Example: ![pageplot](assets/example_conv_time.svg)
        - Scrambler Bitstream Example: ![pageplot](assets/example_scrambler_time.svg)
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
        # upsample bits
        all_bits = np.concatenate(self.bits_list)
        bits_up = np.repeat(all_bits, 2)
        x = np.arange(len(bits_up))

        # set y limits
        y_upper = 1.4 if self.show_bit_values else 1.2
        if self.xlim is not None:
            # double the xlim
            self.xlim = (self.xlim[0], self.xlim[1]*2)
            self.ax.set_xlim(self.xlim)
        else:
            self.ax.set_xlim(0, len(bits_up))
        self.ax.set_ylim(-0.2, y_upper)
        self.ax.grid(False)
        self.ax.set_yticks([0, 1])

        # auxiliary lines
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: int(val/2)))
        bit_edges = np.arange(0, len(bits_up) + 1, 2)
        for pos in bit_edges:
            self.ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        # for each bit vector, draw a section of the plot
        if self.sections:
            start_bit = 0
            for i, (sec_name, sec_len) in enumerate(self.sections):
                bit_start = start_bit * 2
                bit_end = (start_bit + sec_len) * 2
                color = self.colors[i] if self.colors and i < len(self.colors) else 'black'
                if i > 0:
                    bit_start -= 1

                # draw the section of the plot
                self.ax.step(
                    x[bit_start:bit_end],
                    bits_up[bit_start:bit_end],
                    where='post',
                    color=color,
                    linewidth=2.0,
                    label=sec_name if self.label is None else self.label
                )
                
                # show bit values above the plot line
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
            # draw the plot section
            self.ax.step(x, bits_up, where='post',
                         color='black', linewidth=2.0,
                         label=self.label if self.label else None)

            # show bit values above the plot line
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

        # labels
        if self.xlabel:
            self.ax.set_xlabel(self.xlabel)
        if self.ylabel:
            self.ax.set_ylabel(self.ylabel)
        self.apply_ax_style()

class SymbolsPlot(BasePlot):
    r"""
    Class for plotting symbols encoded with line coding, receiving a vector of symbols $s[i]$ and performing the plot in function of the symbol index $i$.
    
    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position of the plot
        symbols_list (List[np.ndarray]): List of symbols
        samples_per_symbol (int): Number of samples per symbol
        sections (Optional[List[Tuple[str, int]]]): Plot sections
        colors (Optional[List[str]]): Plot colors
        show_symbol_values (bool): If `True`, shows the symbol values.
        xlabel (Optional[str]): X-axis label.
        ylabel (Optional[str]): Y-axis label.
        label (Optional[str]): Plot label.
        xlim (Optional[Tuple[float, float]]): X-axis limits.

    Examples:
        - Symbols Plot Example: ![pageplot](assets/example_encoder_time.svg)
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
                 x_axis_label: Optional[Tuple[int, int]] = (-1, 1),
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
        self.x_axis_label = x_axis_label

    def plot(self) -> None:

        # concatenate and oversample the symbol vectors
        all_symbols = np.concatenate(self.symbols_list)
        symbols_up = np.repeat(all_symbols, self.samples_per_symbol)
        x = np.arange(len(symbols_up))

        # axis adjustments
        y_upper = 1.8 if self.show_symbol_values else 1.5
        if self.xlim is not None:
            self.xlim = (self.xlim[0] * self.samples_per_symbol,
                         self.xlim[1] * self.samples_per_symbol)
            self.ax.set_xlim(self.xlim)
        else:
            self.ax.set_xlim(0, len(symbols_up))
        self.ax.set_ylim(-1.5, y_upper)
        self.ax.set_yticks([self.x_axis_label[0], 0, self.x_axis_label[1]])
        self.ax.grid(False)

        # vertical lines marking the start of each symbol
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: int(val / self.samples_per_symbol)))
        symbol_edges = np.arange(0, len(symbols_up) + 1, self.samples_per_symbol)
        for pos in symbol_edges:
            self.ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        # for each symbol vector, draw a section of the plot
        if self.sections:
            start_symbol = 0
            for i, (sec_name, sec_len) in enumerate(self.sections):
                sym_start = start_symbol * self.samples_per_symbol
                sym_end = (start_symbol + sec_len) * self.samples_per_symbol
                color = self.colors[i] if self.colors and i < len(self.colors) else 'black'
                if i > 0:
                    sym_start -= 1

                # draw the plot section
                self.ax.step(
                    x[sym_start:sym_end],
                    symbols_up[sym_start:sym_end],
                    where='post',
                    color=color,
                    linewidth=2.0,
                    label=sec_name if self.label is None else self.label
                )

                # show symbol values above the plot line
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

            # draw the plot section
            color = self.colors[0] if self.colors else 'black'
            self.ax.step(
                x, symbols_up, where='post',
                color=color, linewidth=2.0,
                label=self.label if self.label else None
            )
            # show symbol values above the plot line
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

        # labels
        if self.xlabel:
            self.ax.set_xlabel(self.xlabel)
        if self.ylabel:
            self.ax.set_ylabel(self.ylabel)
        self.apply_ax_style()

class ImpulseResponsePlot(BasePlot):
    r"""
    Class for plotting the impulse response of a filter, receiving a vector of time $t_{imp}$ and performing the plot in function of time $t$.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position of the plot in the GridSpec
        t_imp (np.ndarray): Vector of time $t_{imp}$
        impulse_response (np.ndarray): Impulse response
        t_unit (str, optional): Unit of time on the x-axis ("ms" or "s"). Default is "ms"
        label (Optional[Union[str, List[str]]]): Plot label
        xlabel (Optional[str]): x-axis label
        ylabel (Optional[str]): y-axis label
        xlim (Optional[Tuple[float, float]]): x-axis limits
        amp_norm (Optional[bool]): Normalizes the impulse response to have unitary amplitude. 

    Examples:
        - Impulse Response RRC: ![pageplot](assets/example_formatter_impulse.svg)
        - Impulse Response Manchester: ![pageplot](assets/example_formatter_impulse_man.svg)
        - Impulse Response LPF: ![pageplot](assets/example_lpf_impulse.svg)
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

        # create list of impulse responses
        if isinstance(impulse_response, np.ndarray):
            self.impulse_response = [impulse_response]
        else:
            self.impulse_response = impulse_response


    def plot(self) -> None:
        # time unit
        if self.t_unit == "ms":
            t_plot = self.t_imp * 1000
            default_xlabel = r"Time ($ms$)"
        else:
            t_plot = self.t_imp
            default_xlabel = r"Time ($s$)"

        # Label
        if isinstance(self.label, str) or self.label is None:
            labels = [self.label] * len(self.impulse_response)
        else:
            labels = self.label
        self.ax.set_xlabel(self.xlabel if self.xlabel is not None else default_xlabel)
        self.ax.set_ylabel(self.ylabel if self.ylabel is not None else "Amplitude")

        # plot
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        for i, resp in enumerate(self.impulse_response):
            color = self.apply_color(i) or None
            lbl = labels[i] if labels and i < len(labels) else None
            if self.amp_norm:
                resp = resp / np.max(resp)
            self.ax.plot(t_plot, resp, color=color, label=lbl, **line_kwargs)

        # limits
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        self.apply_ax_style()

class SampledSignalPlot(BasePlot):
    r"""
    Class to plot a sampled signal $s(t)$.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int ou tuple): Position in the GridSpec
        t_signal (np.ndarray): Vector of time $t_{signal}$
        signal (np.ndarray): Filtered signal
        t_samples (np.ndarray): Vector of time $t_{samples}$
        samples (np.ndarray): Samples
        time_unit (str): Time unit. 
        label_signal (str): Label of the filtered signal.
        label_samples (str): Label of the samples.
        xlabel (str): Label of the x-axis.
        ylabel (str): Label of the y-axis.
        title (str): Title of the plot.
        xlim (tuple): Limits of the x-axis.

    Examples:
        - Time Domain Plot Example: ![pageplot](assets/example_sampler_time.svg)
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
            xlabel = r"Time ($ms$)" if self.time_unit == "ms" else r"Time ($s$)"
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xlim = xlim
        self.signal = signal
        self.samples = samples

        # Adjust time unit
        if self.time_unit == "ms":
            self.t_signal = t_signal * 1e3
            self.t_samples = t_samples * 1e3
        else:
            self.t_signal = t_signal
            self.t_samples = t_samples

    def plot(self) -> None:
        # plot
        signal_color = self.colors if isinstance(self.colors, str) else "blue"
        self.ax.plot(self.t_signal, self.signal,color=signal_color, label=self.label_signal, linewidth=2)
        self.ax.stem(self.t_samples, self.samples,linefmt="k-", markerfmt="ko", basefmt=" ",label=self.label_samples)

        # Adjust axis
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlim(self.xlim)
        self.ax.set_title(self.title)
        self.apply_ax_style()
        
        # Legend
        if self.label_signal or self.label_samples:
            leg = self.ax.legend(loc='upper right', frameon=True, fontsize=12)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_alpha(1.0)

class PhasePlot(BasePlot):
    r"""
    Class to plot the phase of the signals $d_I(t)$ and $d_Q(t)$ in the time domain.

    $$
        s(t) = \arctan\left(\frac{d_Q(t)}{d_I(t)}\right)
    $$

    Where: 
        - $s(t)$: Phase vector $s(t)$.
        - $d_I(t)$: In-phase signal component $d_I(t)$.
        - $d_Q(t)$: Quadrature signal component $d_Q(t)$.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position of the plot
        t (np.ndarray): Vector of time
        signals (Union[np.ndarray, List[np.ndarray]]): IQ signals (I and Q)
        time_unit (str): Time unit for plot ("ms" by default, can be "s").

    Examples: 
        - Modulator Constellation/Phase Example: ![pageplot](assets/example_modulator_constellation.svg)

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

        # Time unit
        self.time_unit = time_unit.lower()
        self.t = t
        if self.time_unit == "ms":
            self.t *= 1e3

        if self.labels is None:
            self.labels = ["Phase $I + jQ$"]

        # Ensure signals are in a tuple
        if isinstance(signals, (list, tuple)):
            assert len(signals) == 2, "Signals must be passed as a tuple with two components (I, Q)."
            self.I = signals[0]
            self.Q = signals[1]
        else:
            raise ValueError("Signals must be passed as a tuple with two components (I, Q).")

    def plot(self) -> None:
        # Calculate phase using atan2
        phase = np.angle(self.I + 1j * self.Q)

        # Plot the phase over time
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0)
        self.ax.plot(self.t, phase, label=self.labels[0], color=color, **line_kwargs)

        # Limit of phase between pi and -pi
        self.ax.set_ylim([-np.pi*1.1, np.pi*1.1])
        ticks = [0, np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4, -np.pi, np.pi]
        labels = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{4}$", r"$-\frac{\pi}{4}$", r"$-\frac{3\pi}{4}$", r"$-\pi$", r"$\pi$"]
        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(labels)

        # Adjust axis
        self.ax.set_xlabel(r"Time ($ms$)" if self.time_unit == "ms" else r"Time ($s$)")
        self.ax.set_ylabel(r"Phase ($rad$)")
        self.ax.legend()
        self.apply_ax_style()

class GaussianNoisePlot(BasePlot):
    r"""
    Class to plot the probability density $p(x)$ of a given variance $\sigma^2$, following the expression below. 

    $$
    p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)
    $$

    Where: 
        - $p(x)$: Probability density of the noise.
        - $\sigma^2$: Variance of the noise.
        - $x$: Amplitude of the noise.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position of the plot in the GridSpec
        variance (float): Variance of the noise
        num_points (int): Number of points for the gaussian curve
        legend (str): Legend of the plot
        xlabel (str): Label of the x-axis
        ylabel (str): Label of the y-axis
        xlim (Optional[Tuple[float, float]]): Limit of the x-axis
        span (int): Span of the plot

    Examples:
        - Noise Density Plot Example: ![pageplot](assets/example_noise_gaussian_ebn0.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 variance: float,
                 num_points: int = 5000,
                 legend: str = "$p(x)$",
                 xlabel: str = "Amplitude ($x$)",
                 ylabel: str = "Probability Density $p(x)$",
                 ylim: Optional[Tuple[float, float]] = None,
                 span: int = 100,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.variance = variance
        self.num_points = num_points
        self.legend = legend
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ylim = ylim
        self.span = span

    def plot(self) -> None:
        # Calculate the pdf
        sigma = np.sqrt(self.variance)
        x = np.linspace(-self.span*sigma, self.span*sigma, self.num_points)
        pdf = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-x**2 / (2*self.variance))

        # Plot
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or "darkgreen"

        # plot the pdf
        label = r"$p(x)$" + "\n" + r"$\sigma^2 = " + f"{self.variance:.4f}" + "$"
        self.ax.plot(pdf, x, label=label, color=color, **line_kwargs)
 

        # Adjust axis
        self.ax.set_xlabel(self.ylabel)  
        self.ax.set_ylabel(self.xlabel) 
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        else:
            self.ax.set_ylim([-1, 1])
        self.apply_ax_style()

class PoleZeroPlot(BasePlot):
    r"""
    Plot the diagram of poles and zeros of a discrete transfer function in the z-plane.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position in the GridSpec
        b (np.ndarray): Coefficients of the numerator of the transfer function
        a (np.ndarray): Coefficients of the denominator of the transfer function

    Examples:
        - Pole-Zero Diagram Example: ![pageplot](assets/example_lpf_pz.svg)
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
        # Calculate zeros and poles
        zeros = np.roots(self.b)
        poles = np.roots(self.a)

        # Plot the circle, poles and zeros
        theta = np.linspace(0, 2*np.pi, 512)
        self.ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.6)
        self.ax.scatter(np.real(zeros), np.imag(zeros),marker='o', facecolors='none', edgecolors='blue', s=120, label='Zeros')
        self.ax.scatter(np.real(poles), np.imag(poles),marker='x', color='red',s=120, label='Poles')

        # Adjust axis
        self.ax.axhline(0, color='black', linewidth=0.8)
        self.ax.axvline(0, color='black', linewidth=0.8)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])

        # Labels
        self.ax.set_xlabel("Real")
        self.ax.set_ylabel("Imaginary")
        self.apply_ax_style()

class FrequencyResponsePlot(BasePlot):
    r"""
    Plot the frequency response of a filter from its coefficients (b, a). 
    Calculates the Discrete Fourier Transform of the impulse response using `scipy.signal.freqz`.

    $$
        H(f) = \sum_{n=0}^{N} b_n e^{-j 2 \pi f n} \Big/ \sum_{m=0}^{M} a_m e^{-j 2 \pi f m}
    $$

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position in the GridSpec
        b (np.ndarray): Coefficients of the numerator of the filter
        a (np.ndarray): Coefficients of the denominator of the filter
        fs (float): Sampling frequency
        f_cut (Optional[float]): Cut-off frequency of the filter (Hz)
        xlim (Optional[Tuple[float, float]]): Limit of the x-axis (Hz)
        worN (int): Number of points for the Discrete Fourier Transform
        show_phase (bool): If `True`, plots the phase of the frequency response
        xlabel (str): Label of the x-axis
        ylabel (str): Label of the y-axis

    Examples:
        - Frequency Domain Plot Example: ![pageplot](assets/example_lpf_freq_response.svg)
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
                 xlabel: str = r"Frequency ($Hz$)",
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
        # Calculate frequency response
        w, h = freqz(self.b, self.a, worN=self.worN, fs=self.fs)
        magnitude = mag2db(h)

        # Plot
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or COLOR_IMPULSE
        label = self.labels[0] if self.labels else "$H(f)$"
        self.ax.plot(w, magnitude, color=color, label=label, **line_kwargs)

        # Plot the phase
        if self.show_phase:
            ax2 = self.ax.twinx()
            phase = np.unwrap(np.angle(h))
            ax2.plot(w, phase, color=LPF_PHASE_COLOR, linestyle="--", linewidth=1.5, label="Phase ($rad$)")
            ax2.set_ylabel("Phase ($rad$)")

        # Add vertical bar at cut-off frequency
        if self.f_cut is not None:
            self.ax.axvline(self.f_cut, color=LPF_CUT_OFF_COLOR, linestyle="--", linewidth=2, label=f"$f_c$ = {self.f_cut} Hz")

        # Adjust axis
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(-60, 5)

        # Adjust labels
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.apply_ax_style()

class DetectionFrequencyPlot(BasePlot):
    r"""
    Plot the spectrum of a received signal, with threshold and detected frequencies. Receiving a sampling frequency $f_s$ and a signal $s(t)$ and performing the Fourier transform of the signal, according to the expression below. 

    $$
    \begin{equation}
        S(f) = \mathcal{F}\{s(t)\}
    \end{equation}
    $$

    Where:
        - $S(f)$: Signal in the frequency domain.
        - $s(t)$: Signal in the time domain.
        - $\mathcal{F}$: Fourier transform.
    
    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position of the plot
        fs (float): Sampling frequency
        signal (np.ndarray): Signal to be plotted
        threshold (float): Threshold of the signal
        fc (float): Central frequency

    Examples: 
        - Frequency Domain Plot Example: ![pageplot](assets/example_detector_freq.svg)
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
                 freqs_detected: Optional[Union[List[float], np.ndarray]] = None
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
        self.signal = np.asarray(signal)
        self.threshold = threshold
        self.freqs_detected = freqs_detected
        self.U = 1.0
        self.style = self.style or {}

    def plot(self) -> None:
        P_db = self.signal
        if P_db.ndim != 1:
            raise ValueError("DetectionFrequencyPlot expects a power_matrix vector.")

        n_bins = len(P_db)
        n_fft = 2 * (n_bins - 1)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.fs)

        # Plot in KHz
        freqs_plot = freqs / 1000.0
        line_kwargs = {"linewidth": 1.5, "alpha": 0.9}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or DETECTION_THRESHOLD_COLOR
        label = self.labels[0] if self.labels else "Magnitude (dB)"
        self.ax.plot(freqs_plot, P_db, label=label, color=color, **line_kwargs)

        # Threshold
        thr_line = self.threshold
        thr_label = f"Threshold = {thr_line:.2f} dB"
        self.ax.axhline(thr_line, color=DETECTION_THRESHOLD_COLOR, linestyle="--", linewidth=2, label=thr_label)

        # Plot vertical lines
        detected_bins = np.where(np.asarray(self.freqs_detected) > 0)[0]
        for idx, k in enumerate(detected_bins, start=1):
            f_plot = freqs[k] / 1000.0
            Pk = P_db[k]
            self.ax.plot(f_plot, Pk, 'o', color='k', markersize=6, label=f"$f_{{{idx}}} = {f_plot:.2f}$ kHz")
            self.ax.axvline(f_plot, color='k', linestyle=':', linewidth=2)

        # Adjust axis
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)

        # Labels
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        self.ax.set_xlabel(r"Frequency ($kHz$)")
        self.ax.set_ylabel(r"Magnitude ($dB$)")
        self.apply_ax_style()

class BersnrPlot(BasePlot):
    r"""
    Plot BER curves as a function of Eb/N0.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position in the GridSpec
        EbN0 (np.ndarray): Array of Eb/N0 values (dB)
        ber_curves (List[np.ndarray]): List of BER curves
        labels (List[str]): Labels of each curve
        linestyles (List[str], optional): List of line styles
        markers (List[str], optional): List of marker styles
        xlabel (str, optional): Label of the x-axis
        ylabel (str, optional): Label of the y-axis
        logy (bool, optional): Whether to use a log scale for the y-axis

    Examples: 
        - BER vs Eb/N0 Plot Example: ![pageplot](assets/ber_vs_ebn0.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos: int,
                 EbN0: np.ndarray,
                 ber_curves: List[np.ndarray],
                 linestyles: List[str] = None,
                 markers: List[str] = None,
                 xlabel: str = r"$E_b/N_0$ ($dB$)",
                 ylabel: str = r"Bit Error Rate ($BER$)",
                 logy: bool = True,
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.logy = logy
        self.EbN0 = EbN0

        # Create BER curves
        self.ber_curves = ber_curves if isinstance(ber_curves, (list, tuple)) else [ber_curves]

        # Create labels
        if self.labels is None:
            self.labels = [f"Curve {i+1}" for i in range(len(self.ber_curves))]
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

        # Use log scale by default
        if self.logy:
            self.ax.set_yscale("log")
            self.ax.grid(True, which="both", axis="y", linestyle="--", color="gray", alpha=0.6)

        # Labels
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.apply_ax_style()

class SincronizationPlot(BasePlot):
    r"""
    Plot a signal in the time domain with synchronization marks.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int ou tuple): Position in the GridSpec
        t (np.ndarray): Array of time
        signal (np.ndarray): Signal in the time domain
        sync_start (float): Start time of the synchronization word
        sync_end (float): End time of the synchronization word
        max_corr (float): Time of the peak of correlation
        time_unit (str): Time unit for plotting ("ms" by default, can be "s").

    Examples: 
        - Time Domain Synchronization Plot Example: ![pageplot](assets/example_synchronizer_sync.svg)
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

        # Adjust time unit
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

        # Synchronization word period
        self.ax.axvspan(self.sync_start, self.sync_end,
                        color=SYNC_PLOT_BACKGROUND_COLOR, alpha=0.2, label=r"$\Delta \tau$")

        # Vertical lines of synchronization
        self.ax.axvline(self.max_corr, color=SYNC_PLOT_V_CENTRAL_COLOR, linestyle="--", linewidth=2, label=r"$\tau$")
        self.ax.axvline(self.sync_start, color=SYNC_PLOT_V_LIMIT_COLOR, linestyle="--", linewidth=2, label=r"$\tau +/- (\Delta \tau)/2$")
        self.ax.axvline(self.sync_end, color=SYNC_PLOT_V_LIMIT_COLOR, linestyle="--", linewidth=2)
    
        # Labels
        xlabel = r"Time ($ms$)" if self.time_unit == "ms" else r"Time ($s$)"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(r"Amplitude")
        self.apply_ax_style()

class CorrelationPlot(BasePlot):
    r"""
    Plot correlation vector $c[k]$ as a function of the index $k$.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int ou tuple): Position in the GridSpec
        corr_vec (np.ndarray): Correlation vector $c[k]$
        fs (float): Signal sampling rate in $Hz$
        xlim (Tuple[float, float]): Time limits in $ms$

    Examples: 
        - Correlation Plot Example: ![pageplot](assets/example_synchronizer_corr.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 corr_vec: np.ndarray,
                 fs: float,
                 xlim: Tuple[float, float],
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        
        # Find the index of maximum correlation
        self.corr_vec = corr_vec
        self.sample_indices = np.arange(len(corr_vec))
        self.fs = fs
        self.max_corr_index = np.argmax(corr_vec)
        self.max_corr_value = corr_vec[self.max_corr_index]

        # Sample index limits
        self.index_low = int(xlim[0] * 1e-3 * fs)
        self.index_high = int(xlim[1] * 1e-3 * fs)

        # Define the title and legend
        if self.labels is None:
            self.labels = [r"$c[k]$"]

    def plot(self) -> None:
        # Plot
        color = self.apply_color(0)
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        self.ax.plot(self.sample_indices, self.corr_vec, label=self.labels[0], color=color, **line_kwargs)

        # Maximum correlation
        self.ax.axvline(self.max_corr_index, color=CORR_PLOT_V_LIMIT_COLOR, linestyle='--', label=f"$k_{{max}}$ = {self.max_corr_index}")
        self.ax.scatter(self.max_corr_index, self.max_corr_value, color=CORR_PLOT_V_LIMIT_COLOR, zorder=5)

        # Set limits
        self.ax.set_xlim(self.index_low, self.index_high)

        # Labels
        self.ax.set_xlabel(r"Sample Index $k$")
        self.ax.set_ylabel(r"Normalized Correlation Factor $c[k]$")
        self.ax.legend()
        self.apply_ax_style()

class WaterfallPlot(BasePlot):
    r"""
    Waterfall plot of the power matrix.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int ou tuple): Position in the GridSpec
        power_matrix (np.ndarray): Power matrix
        fs (float): Signal sampling rate in $Hz$
        N (int): Number of samples
        xlim (Tuple[float, float]): Time limits in $ms$

    Examples: 
        - Waterfall Plot Example: ![pageplot](assets/example_detector_waterfall.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 power_matrix: np.ndarray,
                 fs: float,
                 N: int,
                 xlim: Tuple[float, float] = (0, 10),
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.power_matrix = power_matrix
        self.fs = fs
        self.N = N
        self.xlim = xlim

    def plot(self) -> None:
        n_segments, n_freqs = self.power_matrix.shape

        # Frequencies in kHz
        freqs = np.fft.rfftfreq(self.N, d=1/self.fs)
        freqs_khz = freqs / 1000.0
        x = np.linspace(freqs_khz[0], freqs_khz[-1], n_freqs + 1)

        # Segments on the Y axis
        y = np.arange(n_segments + 1)

        # Plot
        im = self.ax.pcolormesh(
            x, y, self.power_matrix,
            cmap="inferno", shading="auto"
        )
        self.ax.invert_yaxis()

        # Colorbar    
        cbar = self.ax.figure.colorbar(im, ax=self.ax)
        cbar.set_label("Magnitude ($dB$)")

        # Set limits
        self.ax.set_xlim(self.xlim[0], self.xlim[1])

        # Labels
        self.ax.set_xlabel("Frequency ($kHz$)")
        self.ax.set_ylabel("Segment Index ($10 ms$)")
        self.ax.grid(False)
        self.apply_ax_style()



class Waterfall3DPlot(BasePlot):
    r"""
    3D waterfall plot of the power matrix.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int ou tuple): Position in the GridSpec
        power_matrix (np.ndarray): Power matrix
        fs (float): Signal sampling rate in $Hz$
        N (int): Number of samples
        freq_window (tuple[float, float]): Frequency limits in $kHz$
        threshold (float): Threshold value
        smooth (bool): Whether to smooth the power matrix
        sigma (float): Standard deviation for the Gaussian filter
        elev (float): Elevation angle in degrees
        azim (float): Azimuth angle in degrees

    Examples: 
        - Waterfall 3D Plot Example: ![pageplot](assets/example_detector_waterfall_3d.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 power_matrix: np.ndarray,
                 fs: float,
                 N: int,
                 freq_window: tuple[float, float] = (0, 10),
                 threshold: float = None,
                 smooth: bool = True,
                 sigma: float = 1.0,
                 elev: float = 5.0,
                 azim: float = -60.0,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos], projection="3d")
        super().__init__(ax, **kwargs)
        self.power_matrix = power_matrix
        self.fs = fs
        self.N = N
        self.freq_window = freq_window
        self.threshold = threshold
        self.smooth = smooth
        self.sigma = sigma
        self.elev = elev
        self.azim = azim

    def plot(self) -> None:
        freqs = np.fft.rfftfreq(self.N, d=1/self.fs)

        # apply frequency window
        if self.freq_window is not None:
            fmin, fmax = self.freq_window
            mask = (freqs >= fmin) & (freqs <= fmax)
            freqs = freqs[mask]
            Z = self.power_matrix[:, mask]
        else:
            Z = self.power_matrix

        # apply smoothing (only to make it more readable)
        if self.smooth:
            Z = gaussian_filter(Z, sigma=self.sigma)

        X = np.arange(Z.shape[0])
        Y = freqs / 1000.0
        X, Y = np.meshgrid(X, Y, indexing="ij")

        # plot surface
        surf = self.ax.plot_surface(
            X, Y, Z,
            cmap="inferno",
            linewidth=0,
            antialiased=True,
            alpha=0.95
        )

        # plot threshold plane
        if self.threshold is not None:
            Z_thr = np.full_like(Z, self.threshold)
            self.ax.plot_surface(
                X, Y, Z_thr,
                color="blue", alpha=0.5, rstride=1, cstride=1, linewidth=0
            )

        self.ax.set_xlabel("Segment Index ($10 ms$)", labelpad=15)
        self.ax.set_ylabel("Frequency ($kHz$)", labelpad=15)
        self.ax.set_zlabel("Magnitude ($dB$)", labelpad=15)

        # reduce the number of ticks on the x axis
        self.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))

        if self.freq_window is not None:
            self.ax.set_ylim(self.freq_window[0]/1000, self.freq_window[1]/1000)

        # apply camera angle
        self.ax.view_init(elev=self.elev, azim=self.azim)

class WaterfallDecisionPlot(BasePlot):
    r"""
    Decision waterfall plot.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int ou tuple): Position in the GridSpec
        matrix (np.ndarray): Decision matrix
        fs (float): Signal sampling rate in $Hz$
        N (int): Number of samples
        xlim (Tuple[float, float]): Time limits in $ms$
        legend_list (List[str]): List of legend labels

    Examples: 
        - Waterfall Detection Plot Example: ![pageplot](assets/example_detector_waterfall_detection.svg)
        - Waterfall Decision Plot Example: ![pageplot](assets/example_detector_waterfall_decision.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 matrix: np.ndarray,
                 fs: float,
                 N: int,
                 xlim: Tuple[float, float] = (0, 10),
                 legend_list: List[str] = None,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.matrix = matrix
        self.fs = fs
        self.N = N
        self.xlim = xlim
        self.legend_list = legend_list or ["Detected", "Confirmed", "Span", "Demodulation"]

        self.cmap = mpl.colors.ListedColormap([
            (1, 1, 1, 0),
            DETECTOR_COLOR1,
            DETECTOR_COLOR2,
            DETECTOR_COLOR3,
            DETECTOR_COLOR4,
        ])
        self.bounds = [0, 1, 2, 3, 4, 5]
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)

    def plot(self) -> None:
        n_segments, n_freqs = self.matrix.shape

        # X axis = frequencies (kHz) -> should have length n_freqs + 1
        freqs = np.fft.rfftfreq(self.N, d=1/self.fs)
        freqs_khz = freqs / 1000.0
        x = np.linspace(freqs_khz[0], freqs_khz[-1], n_freqs + 1)

        # Y axis = segments -> should have length n_segments + 1
        y = np.arange(n_segments + 1)

        im = self.ax.pcolormesh(
            x, y, self.matrix,
            cmap=self.cmap,
            norm=self.norm,
            shading="auto"
        )

        # Legend
        legend_map = {
            "Detected": DETECTOR_COLOR1,
            "Confirmed": DETECTOR_COLOR2,
            "Span": DETECTOR_COLOR3,
            "Demodulation": DETECTOR_COLOR4,
        }

        legend_elements = [
            Line2D([0], [0],
                   marker='s',
                   color='w',
                   markerfacecolor=color,
                   markersize=12,
                   label=label)
            for label, color in legend_map.items()
            if label in self.legend_list
        ]

        if legend_elements:
            leg = self.ax.legend(handles=legend_elements, loc="upper right")
            frame = leg.get_frame()
            frame.set_edgecolor("black")
            frame.set_alpha(1)

        # Labels and limits
        self.ax.set_xlabel("Frequency ($kHz$)")
        self.ax.set_ylabel("Segment Index ($10 ms$)")
        self.ax.grid(False)

        # Limit frequency on X axis (already in kHz)
        self.ax.set_xlim(self.xlim[0], self.xlim[1])

        # Invert Y axis (segment 0 on top)
        self.ax.invert_yaxis()

        self.apply_ax_style()



