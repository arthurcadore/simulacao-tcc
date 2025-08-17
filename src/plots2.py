import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any

# Configurações padrão do matplotlib
plt.style.use('science')
plt.rcParams["figure.figsize"] = (16, 9)
plt.rc('font', size=16)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=22)
plt.rc('legend', frameon=True, edgecolor='black', facecolor='white', fancybox=True, fontsize=12)

def mag2db(signal: np.ndarray) -> np.ndarray:
    """
    Converte a magnitude do sinal para escala logarítmica (dB).
    
    Args:
        signal: Array com os dados do sinal
        
    Returns:
        Array com o sinal convertido para dB
    """
    mag = np.abs(signal)
    mag /= np.max(mag)
    return 20 * np.log10(mag + 1e-12)

class BasePlot(ABC):
    """Classe base abstrata para todos os tipos de plot."""
    
    def __init__(self, 
                 signals: Union[List[Union[np.ndarray, List[np.ndarray]]], np.ndarray],
                 labels: Optional[Union[List[Union[str, List[str]]], str]] = None,
                 titles: Optional[Union[str, List[Union[str, List[str]]]]] = None,
                 layout: Tuple[int, int] = (1, 1),
                 position: Optional[List[Union[Tuple[int, int], Tuple[slice, slice]]]] = None,
                 figsize: Tuple[int, int] = (16, 9),
                 style: Optional[Dict[str, Any]] = None,
                 xlim: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
                 ylim: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
                 colors: Optional[Union[str, List[str], List[List[str]]]] = None):
        """
        Inicializa o plot base.
        
        Args:
            signals: Um ou mais sinais para plotar. Pode ser uma lista de arrays ou uma lista aninhada
                    de arrays para múltiplos sinais no mesmo subplot.
            labels: Rótulos para cada sinal ou lista de listas de rótulos para sinais agrupados
            titles: Títulos para cada subplot ou lista de títulos para subplots
            layout: Layout dos subplots (linhas, colunas) - usado apenas se position=None
            position: Lista de posições dos subplots no grid (ex: [(0,0), (0,1), (1,:)])
            figsize: Tamanho da figura (largura, altura)
            style: Dicionário com estilos personalizados
        """
        # Normaliza os sinais para uma lista de listas de arrays
        self.signals = self._normalize_signals(signals)
        num_plots = len(self.signals)
        
        # Normaliza labels e titles
        self.labels = self._normalize_labels(labels, num_plots)
        self.titles = self._normalize_titles(titles, num_plots)
        
        self.layout = layout
        self.figsize = figsize
        self.style = style or {}
        self.position = position
        
        # Normaliza xlim, ylim e cores
        self.xlim = self._normalize_limits(xlim, num_plots)
        self.ylim = self._normalize_limits(ylim, num_plots)
        self.colors = self._normalize_colors(colors, self.signals)
        
        # Inicializa a figura
        self.fig = plt.figure(figsize=figsize)
        
        if position is not None:
            # Modo de posicionamento personalizado
            if len(position) != num_plots:
                raise ValueError(f"Número de posições ({len(position)}) não corresponde ao número de subplots ({num_plots})")
                
            # Cria um GridSpec com o layout fornecido
            self.gs = gridspec.GridSpec(*layout)
            
            # Cria os eixos para cada posição
            self.axes = np.empty((layout[0], layout[1]), dtype=object)
            for idx, pos in enumerate(position):
                if isinstance(pos[0], (int, np.integer)) and isinstance(pos[1], (int, np.integer)):
                    # Posição simples (i,j)
                    self.axes[pos] = self.fig.add_subplot(self.gs[pos])
                elif isinstance(pos[0], slice) or isinstance(pos[1], slice):
                    # Posição com slice (ex: [0, :])
                    self.axes[pos] = self.fig.add_subplot(self.gs[pos])
                else:
                    raise ValueError(f"Formato de posição inválido: {pos}. Use (int, int) ou (slice, slice)")
        else:
            # Modo de layout padrão
            self.fig, self.axes = plt.subplots(*layout, figsize=figsize)
            if not isinstance(self.axes, np.ndarray):
                self.axes = np.array([self.axes])
            
            # Garante que self.axes seja sempre um array 2D
            if len(self.axes.shape) == 1:
                self.axes = self.axes.reshape(-1, 1)
            
            # Cria um GridSpec para referência
            self.gs = gridspec.GridSpec(*layout)
    
    def _normalize_signals(self, signals):
        """Normaliza os sinais para uma lista de listas de arrays."""
        # Se for um único array numpy, converte para lista
        if isinstance(signals, np.ndarray):
            if signals.ndim == 1:
                return [[signals]]
            elif signals.ndim == 2:
                return [[signals[i]] for i in range(signals.shape[0])]
            else:
                raise ValueError("Arrays com mais de 2 dimensões não são suportados")
        
        # Se for uma lista, processa cada elemento
        normalized = []
        for signal in signals:
            if isinstance(signal, (list, tuple)):
                # Se for uma lista/tupla, assume que são múltiplos sinais para o mesmo subplot
                normalized.append([s for s in signal if s is not None])
            elif signal is not None:
                # Se for um array numpy, coloca em uma lista
                normalized.append([signal])
        
        return normalized
    
    def _normalize_labels(self, labels, num_subplots):
        """Normaliza os rótulos para o formato esperado."""
        if labels is None:
            return [f"Signal {i+1}" for i in range(num_subplots)]
        
        # Se for string, aplica a todos os subplots
        if isinstance(labels, str):
            return [labels] * num_subplots
        
        # Se for uma lista, garante que tenha o tamanho correto
        if len(labels) != num_subplots:
            raise ValueError(f"Número de labels ({len(labels)}) não corresponde ao número de subplots ({num_subplots})")
        
        return labels
    
    def _normalize_titles(self, titles, num_subplots):
        """Normaliza os títulos para o formato esperado."""
        if titles is None:
            return [""] * num_subplots
        
        # Se for string, aplica a todos os subplots
        if isinstance(titles, str):
            return [titles] * num_subplots
        
        # Se for uma lista, garante que tenha o tamanho correto
        if len(titles) != num_subplots:
            raise ValueError(f"Número de títulos ({len(titles)}) não corresponde ao número de subplots ({num_subplots})")
        
        return titles
    
    def _normalize_limits(self, limits, num_subplots):
        """Normaliza os limites dos eixos."""
        if limits is None:
            return [None] * num_subplots
        if isinstance(limits, (tuple, list)) and len(limits) == 2 and all(isinstance(x, (int, float)) for x in limits):
            return [limits] * num_subplots
        if len(limits) != num_subplots:
            raise ValueError(f"Número de limites ({len(limits)}) não corresponde ao número de subplots ({num_subplots})")
        return limits
    
    def _normalize_colors(self, colors, signals):
        """Normaliza as cores para cada sinal."""
        if colors is None:
            # Cores padrão do matplotlib
            return [[f'C{i % 10}' for i in range(len(group))] for group in signals]
            
        # Se for uma única cor, aplica a todos os sinais
        if isinstance(colors, str):
            return [[colors] * len(group) for group in signals]
            
        # Se for uma lista de cores, verifica se corresponde ao número de grupos
        if len(colors) != len(signals):
            raise ValueError(f"Número de grupos de cores ({len(colors)}) não corresponde ao número de subplots ({len(signals)})")
            
        # Para cada grupo de cores, garante que seja uma lista
        normalized = []
        for group_colors, signal_group in zip(colors, signals):
            if isinstance(group_colors, str):
                normalized.append([group_colors] * len(signal_group))
            else:
                if len(group_colors) != len(signal_group):
                    raise ValueError(f"Número de cores ({len(group_colors)}) não corresponde ao número de sinais no grupo ({len(signal_group)})")
                normalized.append(group_colors)
                
        return normalized
    
    def apply_style(self, ax: plt.Axes, idx: int):
        """
        Aplica estilos ao eixo.
        """
        ax.grid(True, **self.style.get('grid', {'alpha': 0.6, 'linestyle': ':'}))

        # Aplica limites dos eixos se especificados
        if self.xlim and self.xlim[idx] is not None:
            ax.set_xlim(self.xlim[idx])
        if self.ylim and self.ylim[idx] is not None:
            ax.set_ylim(self.ylim[idx])

        # Configura legenda sempre no canto superior direito
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(
                loc="upper right",
                frameon=True,
                edgecolor="black",
                facecolor="white",
                fancybox=True,
                fontsize=self.style.get("legend_fontsize", 12)
            )
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_alpha(1.0)
    
    def _save_or_show(self, save_path: Optional[str] = None, **save_kwargs) -> None:
        """
        Salva ou mostra a figura.
        
        Args:
            save_path: Caminho para salvar a figura. Pode ser relativo ao diretório do script.
            **save_kwargs: Argumentos adicionais para savefig
        """
        import os
        
        plt.tight_layout()
        if save_path:
            # Se o caminho for relativo, converte para caminho absoluto baseado no diretório do script
            if not os.path.isabs(save_path):
                # Obtém o diretório do script atual
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Junta com o caminho relativo fornecido
                save_path = os.path.abspath(os.path.join(script_dir, save_path))
            
            # Cria os diretórios necessários se não existirem
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Salva a figura
            self.fig.savefig(save_path, bbox_inches="tight", **save_kwargs)
            plt.close(self.fig)
        else:
            plt.show()
    
    @abstractmethod
    def plot(self, *args, **kwargs):
        """Método abstrato que deve ser implementado pelas classes filhas."""
        pass

    def _apply_style(self, ax, label, title):
        ax.set_title(title)
        leg = ax.legend(loc="upper right", frameon=True, edgecolor="black",
                        facecolor="white", fontsize=12, fancybox=True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_alpha(1.0)

class TimePlot(BasePlot):
    """Classe para plotagem de sinais no domínio do tempo."""

    def __init__(self, signals, t=None, **kwargs):
        super().__init__(signals, **kwargs)
        self.t = t if t is not None else np.arange(len(self.signals[0]))

    def plot(self, save_path: Optional[str] = None, xlim: Optional[Tuple[float, float]] = None) -> None:
        """
        Plota os sinais no domínio do tempo.
        """
        for i, (signal_group, label_group, title, color_group) in enumerate(
            zip(self.signals, self.labels, self.titles, self.colors)
        ):
            ax = self.axes.flat[i]   # ✅ usa apenas o eixo já criado

            if isinstance(label_group, str):
                label_group = [label_group] * len(signal_group)

            for sig, lbl, color in zip(signal_group, label_group, color_group):
                line_style = self.style.get("line", {}).copy()
                line_style["color"] = color
                ax.plot(self.t, sig, label=lbl, **line_style)

            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(title)

            # Ajusta os limites do eixo x se passado no método
            if xlim is not None:
                ax.set_xlim(xlim)

            # Aplica estilo (inclui grid e legenda fixa no canto sup. dir.)
            self.apply_style(ax, i)

            # Se só tem um sinal sem label, remove legenda
            if len(signal_group) == 1 and (not label_group or label_group[0] is None) and hasattr(ax, "legend_"):
                ax.legend_.remove()

        self._save_or_show(save_path)


class FrequencyPlot(BasePlot):
    """Classe para plotagem de sinais no domínio da frequência."""

    def __init__(self, signals, fs: float, fc: float = 0, **kwargs):
        super().__init__(signals, **kwargs)
        self.fs = fs
        self.fc = fc

        # Pré-computa FFTs
        self.freqs = np.fft.fftshift(np.fft.fftfreq(len(self.signals[0][0]), d=1/fs))
        if fc > 1000:
            self.freqs = self.freqs / 1000
            self.x_label = "Frequência (kHz)"
            self.x_limit = (-2.5 * fc / 1000, 2.5 * fc / 1000)
            self.x_limit_comp = (-0.5 * fc / 1000, 0.5 * fc / 1000)
        else:
            self.x_label = "Frequência (Hz)"
            self.x_limit = (-2.5 * fc, 2.5 * fc)
            self.x_limit_comp = (-0.5 * fc, 0.5 * fc)

        # Calcula FFTs de todos os sinais
        self.fft_signals = []
        for group in self.signals:
            fft_group = [np.fft.fftshift(np.fft.fft(sig)) for sig in group]
            self.fft_signals.append(fft_group)

    def plot(self, save_path: Optional[str] = None, **kwargs) -> None:
        """
        Plota os sinais no domínio da frequência (em dB).
        """
        for i, (fft_group, labels, title, color_group) in enumerate(
            zip(self.fft_signals, self.labels, self.titles, self.colors)
        ):
            ax = self.axes.flat[i]

            if isinstance(labels, str):
                labels = [labels] * len(fft_group)

            for fft_signal, label, color in zip(fft_group, labels, color_group):
                y = mag2db(fft_signal)  # usa mesma função de plots.py
                line_style = self.style.get('line', {}).copy()
                line_style['color'] = color
                ax.plot(self.freqs, y, label=label, **line_style)

            # Configurações do subplot
            ax.set_xlabel(self.x_label)
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title(title)
            ax.set_ylim(-80, 5)
            ax.grid(True)

            # Aplica limites (usa compressores para dI/dQ, total para sinal)
            if i == 0:
                ax.set_xlim(self.x_limit)
            else:
                ax.set_xlim(self.x_limit_comp)

            self.apply_style(ax, i)

        self._save_or_show(save_path, **kwargs)


class ConstellationPlot(BasePlot):
    """Classe para plotagem de sinais no domínio do tempo (I e Q) e diagrama de constelação."""

    def __init__(self, signals, fs: float, Rb: float, amplitude: Optional[float] = None, **kwargs):
        """
        Inicializa o plot de constelação.
        
        Args:
            signals: Lista contendo [dI, dQ]
            fs: Frequência de amostragem
            Rb: Taxa de bits
            amplitude: Amplitude dos pontos da constelação (opcional)
            **kwargs: Argumentos adicionais para BasePlot
        """
        if not isinstance(signals, (list, tuple)) or len(signals) != 2:
            raise ValueError("signals deve ser uma lista [dI, dQ]")

        self.dI, self.dQ = signals
        self.fs = fs
        self.Rb = Rb
        self.amplitude = amplitude

        # Monta sinais para BasePlot
        t = np.arange(len(self.dI)) / fs
        signals_proc = [[self.dI, self.dQ], self.dI + 1j * self.dQ]
        super().__init__(signals_proc, **kwargs)
        self.t = t

    def plot(self, save_path: Optional[str] = None, **kwargs) -> None:
        """
        Plota I/Q no tempo e o diagrama de constelação.
        
        Args:
            save_path: Caminho para salvar a figura
        """
        # --- Normalização (centraliza no 0) ---
        dI_c = self.dI - np.mean(self.dI)
        dQ_c = self.dQ - np.mean(self.dQ)

        # Amplitude para pontos ideais
        if self.amplitude is None:
            amp = np.mean(np.abs(np.concatenate([dI_c * 1.1, dQ_c * 1.1])))
        else:
            amp = self.amplitude

        # --- Subplot 1: sinais I/Q no tempo ---
        ax_time = self.axes.flat[0]
        ax_time.plot(self.t, dI_c, color=self.colors[0][0],
                     label=self.labels[0] if self.labels else "$dI(t)$", **self.style.get('line', {}))
        ax_time.plot(self.t, dQ_c, color=self.colors[0][1],
                     label=self.labels[1] if self.labels else "$dQ(t)$", **self.style.get('line', {}))
        ax_time.set_xlabel("Tempo (s)")
        ax_time.set_ylabel("Amplitude")
        ax_time.set_title(self.titles[0] if self.titles else "Sinais I/Q")
        ax_time.grid(True, **self.style.get('grid', {}))
        # Adiciona a legenda com estilo personalizado se fornecido
        legend_style = self.style.get('legend', {})
        leg_time = ax_time.legend(**legend_style)
        leg_time.get_frame().set_facecolor('white')
        leg_time.get_frame().set_edgecolor('black')
        leg_time.get_frame().set_alpha(1.0)

        # Aplica limites SOMENTE no tempo
        if self.xlim and self.xlim[0] is not None:
            ax_time.set_xlim(self.xlim[0])
        if self.ylim and self.ylim[0] is not None:
            ax_time.set_ylim(self.ylim[0])

        # --- Subplot 2: Constelação ---
        ax_const = self.axes.flat[1]

        ax_const.scatter(dI_c, dQ_c, color="darkgreen", alpha=0.5, s=10, label="Amostras $IQ$")
        qpsk_points = np.array([[amp, amp], [amp, -amp], [-amp, amp], [-amp, -amp]])
        ax_const.scatter(qpsk_points[:, 0], qpsk_points[:, 1],
                         color="red", s=160, marker="x", label="Pontos Ideais", linewidth=5)

        ax_const.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_const.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax_const.set_xlabel("Componente em Fase $I$")
        ax_const.set_ylabel("Componente em Quadratura $Q$")
        ax_const.set_title(self.titles[1] if len(self.titles) > 1 else "Diagrama de Constelação")
        ax_const.grid(True, alpha=0.3)
        # Adiciona a legenda com estilo personalizado se fornecido
        leg_iq = ax_const.legend(**legend_style)
        leg_iq.get_frame().set_facecolor('white')
        leg_iq.get_frame().set_edgecolor('black')
        leg_iq.get_frame().set_alpha(1.0)

        # Sempre deixa quadrado se solicitado
        if "aspect" in self.style:
            ax_const.set_aspect(self.style["aspect"])

        self._save_or_show(save_path, **kwargs)






class BitPlot(BasePlot):
    """Classe para plotagem de sequências binárias."""
    
    def __init__(self, bits, **kwargs):
        """
        Inicializa o plot de bits.
        
        Args:
            bits: Lista de sequências de bits
            **kwargs: Argumentos adicionais para BasePlot
        """
        super().__init__(bits, **kwargs)
        
    def plot(self, save_path: Optional[str] = None, show_bits: bool = True, **kwargs) -> None:
        """
        Plota a sequência de bits.
        
        Args:
            save_path: Caminho para salvar a figura
            show_bits: Se True, mostra os valores dos bits no gráfico
            **kwargs: Argumentos adicionais para _save_or_show
        """
        for i, (bits, label, title) in enumerate(zip(self.signals, self.labels, self.titles)):
            ax = self.axes.flat[i]
            
            # Cria o sinal com degraus
            bits_up = np.repeat(bits, 2)
            x = np.arange(len(bits_up)) / 2
            
            # Plota o sinal
            ax.step(x, bits_up, where='post', label=label, 
                   **self.style.get('line', {'linewidth': 2}))
            
            # Adiciona os valores dos bits acima do gráfico
            if show_bits:
                for j, bit in enumerate(bits):
                    ax.text(j + 0.5, 1.15, str(bit), 
                           ha='center', va='bottom', 
                           fontsize=self.style.get('bit_fontsize', 12))
            
            # Adiciona linhas verticais para cada bit
            bit_edges = np.arange(0, len(bits) + 1)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
            
            # Configuração dos eixos
            ax.set_xlim(0, len(bits))
            ax.set_ylim(-0.2, 1.4)
            ax.set_yticks([0, 1])
            ax.set_xlabel('Índice do Bit')
            ax.set_ylabel('Valor')
            ax.set_title(title or 'Sequência de Bits')
            ax.grid(False)
            
            self.apply_style(ax)
        
        self._save_or_show(save_path, **kwargs)


# Funções de conveniência
def plot_time(signals, t=None, labels=None, titles=None, save_path=None, **kwargs):
    """Função de conveniência para plotar sinais no tempo."""
    plot = TimePlot(signals, t=t, labels=labels, titles=titles, **kwargs)
    plot.plot(save_path=save_path)
    return plot

def plot_freq(signals, fs, fc=0, labels=None, titles=None, save_path=None, **kwargs):
    """Função de conveniência para plotar espectros de frequência."""
    plot = FrequencyPlot(signals, fs=fs, fc=fc, labels=labels, titles=titles, **kwargs)
    plot.plot(save_path=save_path)
    return plot

def plot_constellation(iq_data, labels=None, titles=None, save_path=None, **kwargs):
    """Função de conveniência para plotar diagramas de constelação."""
    plot = ConstellationPlot(iq_data, labels=labels, titles=titles, **kwargs)
    plot.plot(save_path=save_path)
    return plot

def plot_bits(bits, labels=None, titles=None, save_path=None, **kwargs):
    """Função de conveniência para plotar sequências de bits."""
    plot = BitPlot(bits, labels=labels, titles=titles, **kwargs)
    plot.plot(save_path=save_path)
    return plot