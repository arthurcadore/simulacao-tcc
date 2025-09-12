"""
Implementa um formatador de pulso para transmissão de sinais digitais. 

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
from .preamble import Preamble
from .formatter import Formatter
from .encoder import Encoder
from .plotter import create_figure, save_figure, TimePlot

class Synchronizer:
    def __init__(self, fs=128_000, Rb=400):
        self.fs = fs
        self.Rb = Rb
        self.Tb = 1 / Rb
        self.sps = int(fs / Rb)
        self.preamble = Preamble()
        self.preamble_sI = self.preamble.preamble_sI
        self.preamble_sQ = self.preamble.preamble_sQ
        self.encoder_I = Encoder(method="NRZ")
        self.encoder_Q = Encoder(method="Manchester")
        self.formatterI = Formatter(alpha=0.8, fs=fs, Rb=Rb, span=6, type="RRC", channel="I")
        self.formatterQ = Formatter(alpha=0.8, fs=fs, Rb=Rb, span=6, type="RRC", channel="Q")
        self.sincronized_word_I = self.formatterI.apply_format(self.encoder_I.encode(self.preamble_sI), add_prefix=False)
        self.sincronized_word_Q = self.formatterQ.apply_format(self.encoder_Q.encode(self.preamble_sQ), add_prefix=False)
    

if __name__ == "__main__":
    
    synchronizer = Synchronizer()

    fig_format, grid_format = create_figure(2,1, figsize=(16, 9))

    TimePlot(
        fig_format, grid_format, (0,0),
        t= np.arange(len(synchronizer.sincronized_word_I)) / synchronizer.formatterI.fs,
        signals=[synchronizer.sincronized_word_I],
        labels=[r"$d_I(t)$"],
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
        labels=[r"$d_Q(t)$"],
        title=r"Canal $Q$",
        colors="darkblue",
        style={
            "line": {"linewidth": 2, "alpha": 1},
            "grid": {"color": "gray", "linestyle": "--", "linewidth": 0.5}
        }
    ).plot()
    
    fig_format.tight_layout()
    save_figure(fig_format, "example_synchronizer_word.pdf")
    
        
