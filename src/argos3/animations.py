# """
# Implementation of animated plot operations.
# 
# Author: Arthur Cadore
# Date: 05-10-2025
# """

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import scienceplots

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



def create_animation(rows=1, cols=1, figsize=(16, 9)):
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


def save_animation(fig: plt.Figure, filename: str, out_dir: str = "../../media", fps: int = 30):
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

    animated_entries = getattr(fig, "_animated_plots", None)
    if not animated_entries:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[OK] Figura salva em {save_path}")
        return save_path

    n_frames = max(len(entry["frames"]) for entry in animated_entries)
    print(f"[INFO] Salvando animação ({n_frames} frames) em {save_path} ...")

    writer = animation.PillowWriter(fps=fps)

    old_bbox = mpl.rcParams.get("savefig.bbox", None)
    mpl.rcParams["savefig.bbox"] = None

    try:
        with writer.saving(fig, save_path, dpi=100):
            for i in range(n_frames):
                for entry in animated_entries:
                    plot_obj = entry["plot"]
                    frames_arr = entry["frames"]
                    idx = i if i < len(frames_arr) else len(frames_arr) - 1
                    frame_val = frames_arr[idx]
                    plot_obj._update(frame_val)
                writer.grab_frame(facecolor="none", edgecolor="none")

        print(f"[OK] Animação salva em {save_path}")
        return save_path

    except Exception as e:
        print(f"[ERRO] Falha ao salvar animação: {e}")
        raise

    finally:
        if old_bbox is not None:
            mpl.rcParams["savefig.bbox"] = old_bbox

class BaseAnimatedPlot:
    """
    Base class for animated plots that can share a figure and GridSpec.

    Args:
        fig (plt.Figure): Figura principal.
        grid (GridSpec): Layout de subplots.
        pos (tuple | int): Posição no GridSpec.
        fps (int): Frames por segundo.
        duration (float): Duração da animação.
        filename (str): Nome padrão do arquivo (caso deseje salvar).
        out_dir (str): Diretório padrão de saída.
    """

    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 fps: int = 30,
                 duration: float = 3.0,
                 filename: str = "animation.gif",
                 out_dir: str = "../../out",
                 title: str = "",
                 xlim=None,
                 ylim=None,
                 colors=None,
                 **kwargs):

        self.fps = fps
        self.duration = duration
        self.filename = filename
        self.out_dir = out_dir
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.colors = colors
        self.fig = fig

        self.ax = fig.add_subplot(grid[pos])
        self._ani = None

    def setup(self):
        raise NotImplementedError

    def _update(self, frame):
        raise NotImplementedError

    def animate(self, frames):
        """Cria o objeto de animação sem salvar e registra o plot na figura."""
        self._ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=frames,
            interval=1000 / self.fps,
            blit=True,
            repeat=False
        )

        if not hasattr(self.fig, "_animated_plots"):
            self.fig._animated_plots = []
        self.fig._animated_plots.append({
            "plot": self,
            "ani": self._ani,
            "frames": list(frames)
        })

        return self._ani

    def save_gif(self, filename=None):
        """Salva individualmente o GIF desse plot (caso queira)."""
        if self._ani is None:
            raise RuntimeError("Chame `.build()` antes de salvar.")
        name = filename or self.filename
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.abspath(os.path.join(script_dir, self.out_dir))
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, name)
        self._ani.save(save_path, writer="pillow", fps=self.fps)
        print(f"[OK] GIF salvo em {save_path}")


class ConstellationAnimatedPlot(BaseAnimatedPlot):
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
        - Modulator Constellation/Phase Example: ![pageplot](../media/transmitter_modulator_constellation_animated.gif)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 dI: np.ndarray,
                 dQ: np.ndarray,
                 rms_norm: bool = True,
                 show_ideal_points: bool = True,
                 amp: float = 1.0,
                 colors: str = "darkgreen",
                 **kwargs):

        super().__init__(fig, grid, pos, **kwargs)

        self.dI = np.asarray(dI)
        self.dQ = np.asarray(dQ)
        self.rms_norm = rms_norm
        self.show_ideal_points = show_ideal_points
        self.amp = amp
        self.color = colors

        if self.rms_norm:
            rms = np.sqrt(np.mean(self.dI**2 + self.dQ**2))
            if rms > 0:
                self.dI *= self.amp / rms
                self.dQ *= self.amp / rms

        self.lim = 1.2 * np.max(np.abs(np.concatenate([self.dI, self.dQ]))) if len(self.dI) > 0 else 1.0
        self.setup()

    def setup(self):
        ax = self.ax
        ax.set_xlim(-self.lim, self.lim)
        ax.set_ylim(-self.lim, self.lim)
        ax.set_xlabel("In Phase ($I$)")
        ax.set_ylabel("Quadrature ($Q$)")
        ax.set_title(self.title or "Animated Constellation")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.6)

        # Pontos QPSK ideais
        if self.show_ideal_points:
            qpsk_points = np.array([
                [self.amp, self.amp],
                [self.amp, -self.amp],
                [-self.amp, self.amp],
                [-self.amp, -self.amp]
            ])
            ax.scatter(qpsk_points[:, 0], qpsk_points[:, 1],
                       color=QPSK_IDEAL_COLOR,
                       s=160, marker="D",
                       label="$QPSK$ Ideal")

        # Scatter animado com label
        self.scatter = ax.scatter([], [], s=20, color=self.color, alpha=0.7, label="$IQ$ samples")

        # Limites opcionais
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        # Cria legenda **uma vez**, removendo duplicatas
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    def _update(self, frame):
        self.scatter.set_offsets(np.column_stack((self.dI[:frame], self.dQ[:frame])))
        return [self.scatter]

    def build(self):
        n_frames = int(self.fps * self.duration)
        if len(self.dI) > 0:
            frames = np.linspace(1, len(self.dI), n_frames, dtype=int)
        else:
            frames = [1]
        self.animate(frames)
        return self.fig

