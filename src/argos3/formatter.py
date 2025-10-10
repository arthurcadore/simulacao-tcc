# """
# Implements a pulse modulator compatible with the PPT-A3 standard.
#
# Author: Arthur Cadore
# Date: 28-07-2025
# """

from matplotlib.pyplot import xlabel
import numpy as np
from .plotter import ImpulseResponsePlot, TimePlot, SymbolsPlot, FrequencyPlot, PowerSpectralDensityPlot, create_figure, save_figure
from .encoder import Encoder
from .env_vars import *

class Formatter:
    def __init__(self, alpha=0.8, fs=128_000, Rb=400, span=6, type="RRC", prefix_duration=0.082, channel=None, bits_per_symbol=1):
        r"""
        Initializes a pulse modulator, with pulse response $g(t)$ used to prepare symbols as $I[n]$ and $Q[n]$ for modulation. The deployment of the pulse response $g(t)$ for each channel is illustrated in the figure below.

        ![pageplot](../assets/pulse_modulate.svg)

        Args:
            alpha (float): Roll-off factor of the RRC pulse.
            fs (int): Sampling frequency.
            Rb (int): Bit rate.
            span (int): Pulse duration in terms of bit periods.
            type (str): Type of pulse.
            prefix_duration (int): Duration of the pure carrier at the beginning of the vector
            channel (str): Channel to be formatted, only $I$ and $Q$ are supported.
            bits_per_symbol (int): Number of bits per symbol.

        Raises:
            ValueError: If the pulse type is not supported.
            ValueError: If the channel is not supported.

        Examples: 
            >>> import argos3
            >>> import numpy as np 
            >>> 
            >>> Xn = np.random.randint(0, 2, 20)
            >>> Yn = np.random.randint(0, 2, 20)
            >>>
            >>> print(Xn)
            [0 1 1 1 0 1 1 1 0 0 0 0 0 0 1 0 1 1 1 1]
            >>> print(Yn)
            [0 0 0 1 0 1 1 0 0 1 0 0 0 1 0 0 1 1 0 1]
            >>> 
            >>> In = argos3.Encoder(method="NRZ").encode(Xn)
            >>> Qn = argos3.Encoder(method="NRZ").encode(Yn)
            >>> 
            >>> formatter_I = argos3.Formatter(Rb=1000, type="RRC", channel="I", bits_per_symbol=1, prefix_duration=0.01)
            >>> formatter_Q = argos3.Formatter(Rb=1000, type="RRC", channel="Q", bits_per_symbol=2, prefix_duration=0.01)
            >>> 
            >>> dI = formatter_I.apply_format(In, add_prefix=True)
            >>> dQ = formatter_Q.apply_format(Qn, add_prefix=True)
            >>> 
            >>> print(dI[:10])
            [-0.07484442 -0.07466479 -0.07461029 -0.07454939 -0.07448207 -0.07440832
             -0.07432813 -0.0742415  -0.07414841 -0.07404887]
            >>> 
            >>> print(dQ[:10])
            [-0.09576597 -0.09565452 -0.09570885 -0.09574086 -0.09575048 -0.09573767
             -0.09570238 -0.0956446  -0.09556429 -0.09546146]

            - Bitstream Plot Example: ![pageplot](assets/example_formatter_time.svg)

        <div class="referencia">
        <b>Reference:</b><br>
        EEL7062, Princípios de Sistemas de Comunicação, Richard Demo Souza (Pg. 55)
        </div>
        """

        # verify channel input
        if channel not in ["I", "Q"]:
            raise ValueError("Channel not supported. Use 'I' or 'Q'.")
        
        # Attributes
        self.channel = channel
        self.prefix_duration = prefix_duration  
        self.alpha = alpha
        self.bits_per_symbol = bits_per_symbol
        self.fs = fs
        self.Rb = Rb * bits_per_symbol
        self.Tb = 1 / self.Rb
        self.sps = int(fs / self.Rb)
        self.span = span
        self.t_rc = np.linspace(-span * self.Tb, span * self.Tb, span * self.sps * 2)

        # type mapping
        type_map = {
            "rrc": 0,
            "manchester": 1,
            "rect": 2
        }
        type = type.lower()
        if type not in type_map:
            raise ValueError("Tipo de pulso inválido. Use 'RRC', 'Manchester' ou 'Rect'.")
        
        self.type = type_map[type]

        # generate pulse response for each type
        if self.type == 0:  # RRC
            self.g = self.rrc_pulse()
            self.w = self.calculate_bandwidth()
        elif self.type == 1:  # Manchester
            self.g, self.g_left, self.g_right = self.manchester_pulse()
            self.w = self.calculate_bandwidth()
        elif self.type == 2:  # Rect
            self.g = self.rect_pulse()
            self.w = self.calculate_bandwidth()

    def rect_pulse(self):
        r"""
        Generates a rectangular (NRZ) pulse of duration $T_b$. The pulse response $g(t)$ is defined by the expression below.
        
        $$
            g_{rect}(t) =
            \begin{cases}
                1, & |t| \leq T_b/2 \\
                0, & \text{otherwise}
            \end{cases}
        $$

        Where: 
            - $g_{rect}(t)$: Rectangular pulse in the time domain.
            - $T_b$: Bit period.
            - $t$: Time vector.

        Args:
            None

        Returns:
            np.ndarray: Rectangular pulse response $g(t)$.

        Examples: 
            - Impulse Response Example: ![pageplot](assets/example_formatter_impulse_rect.svg)
        """
        g = np.where(np.abs(self.t_rc) <= self.Tb/2, 1.0, 0.0)
        g = g / np.sqrt(np.sum(g**2))
        return g

    def rrc_pulse(self, shift=0.0):
        r"""
        Generates the Root Raised Cosine ($RRC$) pulse. The $RRC$ pulse in the time domain is defined by the expression below.

        $$
            g_{RRC}(t) = \frac{(1 - \alpha) sinc((1- \alpha) t / T_b) + \alpha (4/\pi) \cos(\pi (1 + \alpha) t / T_b)}{1 - (4 \alpha t / T_b)^2}
        $$

        Where: 
            - $g_{RRC}(t)$: $RRC$ pulse in the time domain.
            - $\alpha$: Roll-off factor of the pulse.
            - $T_b$: Bit period.
            - $t$: Time vector.

        Args: 
            shift (float): Time shift, used to shift the pulse in the time domain.

        Returns:
           rc (np.ndarray): Normalized RRC pulse.

        Examples: 
            - Impulse Response Example: ![pageplot](assets/example_formatter_impulse.svg)
        """
        self.t_rc = np.array(self.t_rc, dtype=float)
        # apply time shift
        t_shifted = self.t_rc - shift

        # create the rc array
        rc = np.zeros_like(t_shifted)

        # calculate the rc
        for i, ti in enumerate(t_shifted):
            if np.isclose(ti, 0.0):
                rc[i] = 1.0 + self.alpha * (4/np.pi - 1)
            elif self.alpha != 0 and np.isclose(np.abs(ti), self.Tb/(4*self.alpha)):
                rc[i] = (self.alpha/np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi/(4*self.alpha)) +
                    (1 - 2/np.pi) * np.cos(np.pi/(4*self.alpha))
                )
            else:
                num = np.sin(np.pi * ti * (1 - self.alpha) / self.Tb) + \
                      4 * self.alpha * (ti / self.Tb) * np.cos(np.pi * ti * (1 + self.alpha) / self.Tb)
                den = np.pi * ti * (1 - (4 * self.alpha * ti / self.Tb) ** 2) / self.Tb
                rc[i] = num / den

        # normalize energy to 1
        rc = rc / np.sqrt(np.sum(rc**2))
        return rc
    
    def manchester_pulse(self):
        r"""
        Manchester pulse, defined as the difference of two RRC pulses symmetrically shifted, as shown in the expression below.

        $$
            g_{MAN}(t) = g_{RRC}(t + T_b/2) - g_{RRC}(t - T_b/2)
        $$

        Where: 
            - $g_{MAN}(t)$: Manchester pulse in the time domain.
            - $g_{RRC}(t)$: RRC pulse in the time domain.
            - $T_b$: Bit period.
            - $t$: Time vector.

        Examples: 
            - Impulse Response Example: ![pageplot](assets/example_formatter_impulse_man.svg)
        """
        g_left = -self.rrc_pulse(shift=self.Tb/2)
        g_right = +self.rrc_pulse(shift=-self.Tb/2)
        g = g_left + g_right

        # normalize energy to 1
        g = g / np.sqrt(np.sum(g**2))

        return g, g_left, g_right

    def calculate_bandwidth(self):
        r"""
        Calculates the bandwidth (bilateral) of the pulse response.

        For RRC pulse, the bandwidth is calculated as 
        $$
            W = 2 \cdot \frac{1+\alpha}{2 \cdot T_b}
        $$

        where: 
            - $W$: Bandwidth of the pulse response.
            - $\alpha$: Roll-off factor of the pulse.
            - $T_b$: Bit period.

        For Manchester pulse, the bandwidth is calculated as 
        $$
            W = 2 \cdot \frac{1+\alpha}{T_b}
        $$

        where: 
            - $W$: Bandwidth of the pulse response.
            - $\alpha$: Roll-off factor of the pulse.
            - $T_b$: Bit period.

        For Rect pulse, the bandwidth is calculated as 
        $$
            W = 2 \cdot \frac{1}{T_b}
        $$

        where: 
            - $W$: Bandwidth of the pulse response.
            - $T_b$: Bit period.

        Returns:
            float: Bandwidth of the pulse response.

        Examples: 
            - Frequency Domain Bandwidth Example: ![pageplot](assets/example_formatter_freq_band.svg)
        """
        if self.type == 0:  # RRC
            return  ((1+self.alpha)/(2*self.Tb))
        elif self.type == 1:  # Manchester
            return  ((1+self.alpha)/(2*self.Tb))
        elif self.type == 2:  # Rect
            return  self.Rb

    def apply_format(self, symbols, add_prefix=True):
        r"""
        Applies the formatting process to the input symbols using the initialized pulse. Also adds a prefix to the signal if `add_prefix` is True. The formatting process is given by: 

        $$
           d(t) = \sum_{n} x[n] \cdot g(t - nT_b)
        $$

        Where: 
            - $d(t)$: Formatted signal output.
            - $x$: Input symbol vector.
            - $g(t)$: Formatting pulse.
            - $n$: Bit index.
            - $T_b$: Bit period.

        Args:
            symbols (np.ndarray): Input symbol vector.
        
        Returns:
            out_symbols (np.ndarray): Formatted symbol vector.
        """

        # add prefix if required
        if add_prefix:
            symbols = self.add_prefix(symbols)

        pulse = self.g
        # samples per symbol (bits per symbol)
        sps = int(self.fs / (self.Rb / self.bits_per_symbol))

        self.symbolsup = np.zeros(len(symbols) * sps)
        self.symbolsup[::sps] = symbols
        out_sys = np.convolve(self.symbolsup, pulse, mode='same')

        return out_sys

    def add_prefix(self, symbols):
        """
        Adds a pure carrier prefix to the signal. For the $I$ channel, adds a vector of symbols $+1$, for the $Q$ channel, adds a vector of symbols $0$, with duration of `prefix_duration`. 
        
        Since applying the `IQ` modulator we have a pure carrier at the beginning of the signal, according to the expression below: 

        $$
            s(t) = 1(t) \cdot \cos(2\pi f_c t) - 0(t) \cdot \sin(2\pi f_c t) \mapsto s(t) = \cos(2\pi f_c t)
        $$

        Where: 
            - $s(t)$: Modulated signal.
            - $1(t)$ and $0(t)$: Pure carrier prefix.
            - $f_c$: Carrier frequency.
            - $t$: Time vector.
        
        Args:
            symbols (np.ndarray): Symbol vector to be formatted.
        
        Returns:
            symbols (np.ndarray): Symbol vector with prefix added.
        """

        # verify channel
        if self.channel == "I":
            carrier = np.ones(int(self.prefix_duration * self.Rb / self.bits_per_symbol))
        elif self.channel == "Q":
            carrier = np.zeros(int(self.prefix_duration * self.Rb / self.bits_per_symbol))

        # add prefix
        symbols = np.concatenate([carrier, symbols])
        return symbols


if __name__ == "__main__":

    fs=128_000
    Rb=1000

    X = np.random.randint(0, 2, 2000)
    Y = np.random.randint(0, 2, 2000)

    encoder_I = Encoder(method="NRZ")
    encoded_Q = Encoder(method="NRZ")

    In = encoder_I.encode(X)
    Qn = encoded_Q.encode(Y)
    
    formatterI = Formatter(alpha=0.8, fs=fs, Rb=Rb, span=10, type="RRC", channel="I", bits_per_symbol=1, prefix_duration=0.005)
    formatterQ = Formatter(alpha=0.8, fs=fs, Rb=Rb, span=10, type="Manchester", channel="Q", bits_per_symbol=2, prefix_duration=0.005)
    
    dI1 = formatterI.apply_format(In, add_prefix=False)
    dQ1 = formatterQ.apply_format(Qn, add_prefix=False)
    
    print("Xn:",  ' '.join(f"{x:+d}" for x in In[:10]))
    print("Yn:",  ' '.join(f"{y:+d}" for y in Qn[:10]))
    print("dI:", ''.join(str(b) for b in dI1[:5]))
    print("dQ:", ''.join(str(b) for b in dQ1[:5]))

    fig_impulse, grid_impulse = create_figure(1, 1, figsize=(16, 5))

    ImpulseResponsePlot(
        fig_impulse, grid_impulse, (0, 0),
        formatterI.t_rc, formatterI.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        amp_norm=True, 
        title=IMPULSE_TITLE
    ).plot()

    fig_impulse.tight_layout()
    save_figure(fig_impulse, "example_formatter_impulse.pdf")

    fig_impulse_man, grid_impulse_man = create_figure(2, 1, figsize=(16, 9))

    ImpulseResponsePlot(
        fig_impulse_man, grid_impulse_man, (0, 0),
        formatterQ.t_rc, 
        [formatterQ.g_left, formatterQ.g_right],
        t_unit="ms",
        colors=[COLOR_AUX1, COLOR_AUX2],
        label=[r"$g_{L}(t)$", r"$g_{R}(t)$"], 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        title=IMPULSE_TITLE
    ).plot()

    ImpulseResponsePlot(
        fig_impulse_man, grid_impulse_man, (1, 0),
        formatterQ.t_rc, formatterQ.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        amp_norm=True, 
    ).plot()

    fig_impulse_man.tight_layout()
    save_figure(fig_impulse_man, "example_formatter_impulse_man.pdf")

    formatterRect = Formatter(fs=fs, Rb=Rb, type="Rect", channel="I", bits_per_symbol=1, prefix_duration=0)

    fig_impulse_rect, grid_impulse_rect = create_figure(1, 1, figsize=(16, 5))
    ImpulseResponsePlot(
        fig_impulse_rect, grid_impulse_rect, (0, 0),
        formatterRect.t_rc, formatterRect.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        amp_norm=True, 
        title=IMPULSE_TITLE
    ).plot()

    fig_impulse_rect.tight_layout()
    save_figure(fig_impulse_rect, "example_formatter_impulse_rect.pdf")

    fig_format, grid_format = create_figure(4, 2, figsize=(16, 12))

    SymbolsPlot(
        fig_format, grid_format, (0,0),
        symbols_list=[In],
        sections=[(r"$I[n]$", len(In))],
        colors=[COLOR_I],
        xlabel=SYMBOLS_X, 
        ylabel=SYMBOLS_Y, 
        title=I_CHANNEL_TITLE,
        xlim=[0,20]
    ).plot()

    SymbolsPlot(
        fig_format, grid_format, (0,1),
        symbols_list=[Qn],
        sections=[(r"$Q[n]$", len(Qn))],
        colors=[COLOR_Q],
        xlabel=SYMBOLS_X, 
        ylabel=SYMBOLS_Y, 
        title=Q_CHANNEL_TITLE,
        xlim=[0,20]
    ).plot()
        
    TimePlot(
        fig_format, grid_format, (1,0),
        t= np.arange(len(formatterI.symbolsup)) / formatterI.fs,
        signals=[formatterI.symbolsup],
        labels=[r"$d_I(t)$"],
        amp_norm=True,
        colors=COLOR_I,
        xlim=[0,20]
    ).plot()
    
    TimePlot(
        fig_format, grid_format, (1,1),
        t= np.arange(len(formatterQ.symbolsup)) / formatterQ.fs,
        signals=[formatterQ.symbolsup],
        labels=[r"$d_Q(t)$"],
        amp_norm=True,
        colors=COLOR_Q,
        xlim=[0,20]
    ).plot()
    

    ImpulseResponsePlot(
        fig_format, grid_format, (2,0),
        formatterI.t_rc, formatterI.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        amp_norm=True,
    ).plot()

    ImpulseResponsePlot(
        fig_format, grid_format, (2,1),
        formatterQ.t_rc, formatterQ.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        amp_norm=True, 
    ).plot()


    TimePlot(
        fig_format, grid_format, (3,0),
        t= np.arange(len(dI1)) / formatterI.fs,
        signals=[dI1],
        labels=[r"$d_I(t)$"],
        amp_norm=True,
        colors=COLOR_I,
        xlim=[0,20]
    ).plot()
    
    TimePlot(
        fig_format, grid_format, (3,1),
        t= np.arange(len(dQ1)) / formatterQ.fs,
        signals=[dQ1],
        labels=[r"$d_Q(t)$"],
        amp_norm=True,
        colors=COLOR_Q,
        xlim=[0,20]
    ).plot()
    
    fig_format.tight_layout()
    save_figure(fig_format, "example_formatter_time.pdf")


    fig_freq, grid_freq = create_figure(2, 2, figsize=(16, 12))

    ImpulseResponsePlot(
        fig_freq, grid_freq, (0,0),
        formatterI.t_rc, formatterI.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        amp_norm=True, 
    ).plot()

    ImpulseResponsePlot(
        fig_freq, grid_freq, (0,1),
        formatterQ.t_rc, formatterQ.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        amp_norm=True, 
    ).plot()

    FrequencyPlot(
        fig_freq, grid_freq, (1,0),
        formatterI.fs,
        dI1,
        labels=[r"$D_I(f)$"],
        xlim=(-formatterI.w*1.5/1000, formatterI.w*1.5/1000),
        colors=COLOR_I,
        fc=0,
        bandwidth=formatterI.w
    ).plot()

    FrequencyPlot(
        fig_freq, grid_freq, (1,1),
        formatterQ.fs,
        dQ1,
        labels=[r"$D_Q(f)$"],
        xlim=(-formatterQ.w*1.5/1000, formatterQ.w*1.5/1000),
        colors=COLOR_Q,
        fc=0,
        bandwidth=formatterQ.w
    ).plot()

    fig_freq.tight_layout()
    save_figure(fig_freq, "example_formatter_freq.pdf")

    fig_freq, grid_freq = create_figure(1, 2, figsize=(16, 6))

    FrequencyPlot(
        fig_freq, grid_freq, (0,0),
        formatterI.fs,
        dI1,
        labels=[r"$D_I(f)$"],
        xlim=(-formatterI.w*1.5/1000, formatterI.w*1.5/1000),
        colors=COLOR_I,
        fc=0,
        bandwidth=formatterI.w
    ).plot()

    FrequencyPlot(
        fig_freq, grid_freq, (0,1),
        formatterQ.fs,
        dQ1,
        labels=[r"$D_Q(f)$"],
        xlim=(-formatterQ.w*1.5/1000, formatterQ.w*1.5/1000),
        colors=COLOR_Q,
        fc=0,
        bandwidth=formatterQ.w
    ).plot()

    fig_freq.tight_layout()
    save_figure(fig_freq, "example_formatter_freq_band.pdf")


    fig_power, grid_power = create_figure(3, 2, figsize=(16, 16))

    PowerSpectralDensityPlot(
        fig_power, grid_power, (0,0),
        formatterI.fs,
        formatterI.symbolsup,
        labels=[r"$d_I(t)$"],
        title=I_CHANNEL_TITLE,
        xlim=(-10, 10),
        colors=COLOR_I,
    ).plot()

    PowerSpectralDensityPlot(
        fig_power, grid_power, (0,1),
        formatterQ.fs,
        formatterQ.symbolsup,
        labels=[r"$d_Q(t)$"],
        title=Q_CHANNEL_TITLE,
        xlim=(-10, 10),
        colors=COLOR_Q,
    ).plot()

    PowerSpectralDensityPlot(
        fig_power, grid_power, (1,0),
        formatterI.fs,
        dI1,
        labels=[r"$d_I(t)$"],
        title=I_CHANNEL_TITLE,
        xlim=(-10, 10),
        colors=COLOR_I,
    ).plot()

    PowerSpectralDensityPlot(
        fig_power, grid_power, (1,1),
        formatterQ.fs,
        dQ1,
        labels=[r"$d_Q(t)$"],
        title=Q_CHANNEL_TITLE,
        xlim=(-10, 10),
        colors=COLOR_Q,
    ).plot()

    fig_power.tight_layout()
    save_figure(fig_power, "example_formatter_power.pdf")

    # COMPARAÇÃO RRC e RECT
    InUp = formatterRect.apply_format(In)
    QnUp = formatterRect.apply_format(Qn)

    fig_comp, grid_comp = create_figure(3, 2, figsize=(16, 12))

    ImpulseResponsePlot(
        fig_comp, grid_comp, (0, 0),
        formatterI.t_rc, formatterI.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        amp_norm=True, 
        title=IMPULSE_TITLE
    ).plot()

    ImpulseResponsePlot(
        fig_comp, grid_comp, (0, 1),
        formatterRect.t_rc, formatterRect.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=IMPULSE_XLIM, 
        amp_norm=True, 
        title=IMPULSE_TITLE
    ).plot()

    TimePlot(
        fig_comp, grid_comp, (1, 0),
        signals=[dI1, dQ1],
        t=np.arange(len(dI1)) / fs,
        colors=[COLOR_I,  COLOR_Q],
        labels=[r"$d_I(t)$", r"$d_Q(t)$"],
        amp_norm=True,
        xlim=[0,20],
        title="$RRC$ Pulse Modulation"
    ).plot()

    TimePlot(
        fig_comp, grid_comp, (1, 1),
        signals=[InUp, QnUp],
        t=np.arange(len(InUp)) / fs,
        colors=[COLOR_I,  COLOR_Q],
        labels=[r"$d_I(t)$", r"$d_Q(t)$"],
        xlim=[0,20],
        title="$Rect$ Pulse Modulation"
    ).plot()

    PowerSpectralDensityPlot(
        fig_comp, grid_comp, (2, 0),
        fs,
        signals=[dI1, dQ1],
        labels=[r"$d_I(t)$", r"$d_Q(t)$"],
        colors=[COLOR_I,  COLOR_Q],
        xlim=(-10, 10),
        nperseg=1024,
        ylim=(-120, -20),
        amp_norm=True,
    ).plot()

    PowerSpectralDensityPlot(
        fig_comp, grid_comp, (2, 1),
        fs,
        signals=[InUp, QnUp],
        labels=[r"$d_I(t)$", r"$d_Q(t)$"],
        colors=[COLOR_I,  COLOR_Q],
        xlim=(-10, 10),
        nperseg=1024,
        ylim=(-120, -20),
        amp_norm=True,
    ).plot()

    fig_comp.tight_layout()
    save_figure(fig_comp, "example_formatter_comp.pdf")


    # Codificado NRZ-Man Formatado

    fs=128_000
    Rb=1000

    X = np.random.randint(0, 2, 20)
    Y = np.random.randint(0, 2, 20)

    encoder_I = Encoder(method="NRZ")
    encoded_Q = Encoder(method="Manchester")

    In = encoder_I.encode(X)
    Qn = encoded_Q.encode(Y)
    
    formatterI = Formatter(alpha=0.8, fs=fs, Rb=Rb/2, span=10, type="RRC", channel="I", bits_per_symbol=1, prefix_duration=0.005)
    formatterQ = Formatter(alpha=0.8, fs=fs, Rb=Rb, span=10, type="RRC", channel="Q", bits_per_symbol=1, prefix_duration=0.005)
    
    dI2 = formatterI.apply_format(In, add_prefix=False)
    dQ2 = formatterQ.apply_format(Qn, add_prefix=False)

    fig_format, grid_format = create_figure(3, 2, figsize=(16, 12))

    ImpulseResponsePlot(
        fig_format, grid_format, (0,slice(0,2)),
        formatterI.t_rc, formatterI.g,
        t_unit="ms",
        colors=COLOR_IMPULSE,
        label=r"$g(t)$", 
        xlabel=IMPULSE_X, 
        ylabel=IMPULSE_Y, 
        xlim=(-10,10), 
        amp_norm=True, 
        title="RRC Impulse Response"
    ).plot()

    SymbolsPlot(
        fig_format, grid_format, (1,0),
        symbols_list=[In],
        sections=[(r"$I[n]$", len(In))],
        colors=[COLOR_I],
        xlabel=SYMBOLS_X, 
        ylabel=SYMBOLS_Y, 
        title=I_CHANNEL_TITLE
    ).plot()

    SymbolsPlot(
        fig_format, grid_format, (1,1),
        symbols_list=[Qn],
        sections=[(r"$Q[n]$", len(Qn))],
        colors=[COLOR_Q],
        xlabel=SYMBOLS_X, 
        ylabel=SYMBOLS_Y, 
        title=Q_CHANNEL_TITLE
    ).plot()
        
    TimePlot(
        fig_format, grid_format, (2,0),
        t= np.arange(len(dI2)) / formatterI.fs,
        signals=[dI2],
        labels=[r"$d_I(t)$"],
        amp_norm=True,
        colors=COLOR_I,
    ).plot()
    
    TimePlot(
        fig_format, grid_format, (2,1),
        t= np.arange(len(dQ2)) / formatterQ.fs,
        signals=[dQ2],
        labels=[r"$d_Q(t)$"],
        amp_norm=True,
        colors=COLOR_Q,
    ).plot()

    fig_format.tight_layout()
    save_figure(fig_format, "example_formatter_comp_rrc.pdf")