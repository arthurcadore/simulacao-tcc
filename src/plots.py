import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.gridspec as gridspec
import __main__
from collections import defaultdict

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

def mag2db(signal):
    mag = np.abs(signal)
    mag /= np.max(mag)
    return 20 * np.log10(mag + 1e-12)

def superplot(ax, signal, label, color):
    sig_up = np.repeat(signal, 2)
    x = np.arange(0, len(signal) * 2) / 2
    bit_edges = np.arange(0, len(signal) + 1)

    ax.step(x, sig_up, where='post', label=label, color=color, linewidth=2)
    ax.set_xlim(0, len(signal))
    ax.set_ylim(-0.2, 1.4)
    ax.set_yticks([0, 1])
    ax.grid(False)

    for i, bit in enumerate(signal):
        ax.text(i + 0.5, 1.15, str(bit), ha='center', va='bottom', fontsize=12)
    for pos in bit_edges:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

    leg = ax.legend(
        loc='upper right', frameon=True, edgecolor='black',
        facecolor='white', fontsize=12, fancybox=True
    )
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_alpha(1.0)


class Plotter:
    def __init__(self):
        pass

    def plot_time_domain(self, s1, s2, t, label1, label2, title1, title2, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        ax1.plot(t, s1, label=label1, color='blue')
        ax1.set_title(title1)
        ax1.set_xlim(0, 0.1)
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        leg1 = ax1.legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)

        ax2.plot(t, s2, label=label2, color='red')
        title = f'{title2}'
        ax2.set_title(title)
        ax2.set_xlim(0, 0.1)
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        leg2 = ax2.legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg2.get_frame().set_facecolor('white')
        leg2.get_frame().set_edgecolor('black')
        leg2.get_frame().set_alpha(1.0)

        self._save_or_show(fig, save_path)

    def plot_frequency_domain(self, s1, s2, fs, fc, label1, label2, title1, title2, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(s1), d=1/fs))
        
        if fc > 1000:
            freqs = freqs / 1000
            x_label = "Frequência (kHz)"
            x_limit = (-2.5 * fc / 1000, 2.5 * fc / 1000)
        else:
            x_label = "Frequência (Hz)"
            x_limit = (-2.5 * fc, 2.5 * fc)

        fft_clean = np.fft.fftshift(np.fft.fft(s1))
        fft_clean_db = mag2db(fft_clean)
        ax1.plot(freqs, fft_clean_db, color='blue', label=label1)
        ax1.set_title(title1)
        ax1.set_ylim(-80, 5)
        ax1.set_xlim(*x_limit)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True)
        leg1 = ax1.legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)

        fft_noisy = np.fft.fftshift(np.fft.fft(s2))
        fft_noisy_db = mag2db(fft_noisy)
        ax2.plot(freqs, fft_noisy_db, color='red', label=label2)
        ax2.set_title(title2)
        ax2.set_ylim(-80, 5)
        ax2.set_xlim(*x_limit)
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_xlabel(x_label)
        ax2.grid(True)
        leg2 = ax2.legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg2.get_frame().set_facecolor('white')
        leg2.get_frame().set_edgecolor('black')
        leg2.get_frame().set_alpha(1.0)

        self._save_or_show(fig, save_path)
    
    def plot_bits(self, bits_list, sections=None, colors=None, save_path=None):
        all_bits = np.concatenate(bits_list)
        bits_up = np.repeat(all_bits, 2)
        x = np.arange(len(bits_up))

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.set_xlim(0, len(bits_up))
        ax.set_ylim(-0.2, 1.2)
        ax.grid(False)
        ax.set_yticks([0, 1])
        bit_edges = np.arange(0, len(bits_up) + 1, 2)
        for pos in bit_edges:
            ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        if sections:
            start_bit = 0
            for i, (sec_name, sec_len) in enumerate(sections):
                bit_start = start_bit * 2
                bit_end = (start_bit + sec_len) * 2
                color = colors[i] if colors else 'black'

                if i > 0:
                    bit_start -= 1

                ax.step(
                    x[bit_start:bit_end],
                    bits_up[bit_start:bit_end],
                    where='post',
                    color=color,
                    linewidth=1.5,
                    label=sec_name if i == 0 or sec_name not in sections[:i] else None
                )
                start_bit += sec_len
        else:
            ax.step(x, bits_up, where='post', color='black', linewidth=1.5, label='Bits')

        ax.set_xlabel('Index')
        ax.set_ylabel('Valor')
        leg = ax.legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_alpha(1.0)

        self._save_or_show(fig, save_path)

    def plot_encode(self, s1, s2, s3, s4, label1, label2, label3, label4, title1, title2, title3, title4, save_path=None):
        s1_up = np.repeat(s1, 2)
        s2_up = np.repeat(s2, 2)
        x = np.arange(len(s1_up))
        bit_edges = np.arange(0, len(s1_up) + 1, 2)

        fig, axs = plt.subplots(4, 1, sharex=True)

        def setup_grid(ax):
            ax.set_xlim(0, len(s1_up))
            ax.set_ylim(-0.2, 1.4)
            ax.grid(False)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        axs[0].step(x, s1_up, where='post', label=label1, color='navy', linewidth=3)
        for i, bit in enumerate(s1):
            axs[0].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[0].set_ylabel(title1)
        leg0 = axs[0].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        setup_grid(axs[0])

        axs[1].step(x, s3, where='post', label=label3, color='darkred', linewidth=3)
        for i in range(len(s1)):
            pair = ''.join(str(b) for b in s3[2 * i:2 * i + 2])
            axs[1].text(i * 2 + 1, 1.15, pair, ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[1].set_ylabel(title3)
        leg1 = axs[1].legend( 
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)
        setup_grid(axs[1])

        axs[2].step(x, s2_up, where='post', label=label2, color='navy', linewidth=3)
        for i, bit in enumerate(s2):
            axs[2].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[2].set_ylabel(title2)
        leg2 = axs[2].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg2.get_frame().set_facecolor('white')
        leg2.get_frame().set_edgecolor('black')
        leg2.get_frame().set_alpha(1.0)
        setup_grid(axs[2])

        axs[3].step(x, s4, where='post', label=label4, color='darkred', linewidth=3)
        for i in range(len(s2)):
            pair = ''.join(str(b) for b in s4[2 * i:2 * i + 2])
            axs[3].text(i * 2 + 1, 1.15, pair, ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[3].set_ylabel(title4)
        leg3 = axs[3].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg3.get_frame().set_facecolor('white')
        leg3.get_frame().set_edgecolor('black')
        leg3.get_frame().set_alpha(1.0)
        setup_grid(axs[3])

        plt.xlabel('Bits')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        self._save_or_show(fig, save_path)

    def plot_conv(self, s1, s2, s3, label1, label2, label3, title1, title2, title3, save_path=None):

        s1_up = np.repeat(s1, 2)
        s2_up = np.repeat(s2, 2)
        s3_up = np.repeat(s3, 2)
        x = np.arange(len(s2_up))
        bit_edges = np.arange(0, len(s2_up) + 1, 2)

        # Configuração do gráfico
        fig, axs = plt.subplots(3, 1, sharex=True)
        def setup_grid(ax):
            ax.set_xlim(0, len(s2_up))
            ax.set_ylim(-0.2, 1.4)
            ax.grid(False)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
        
        # Canal I
        axs[0].step(x, s1_up, where='post', label=label1, color='darkgreen', linewidth=3)
        for i, bit in enumerate(s1):
            axs[0].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[0].set_ylabel(title1)
        leg0 = axs[0].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        axs[0].set_yticks([0, 1])
        setup_grid(axs[0])

        # Canal I
        axs[1].step(x, s2_up, where='post', label=label2, color='navy', linewidth=3)
        for i, bit in enumerate(s2):
            axs[1].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[1].set_ylabel(title2)
        leg0 = axs[1].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        axs[1].set_yticks([0, 1])
        setup_grid(axs[1])

        # Canal Q
        axs[2].step(x, s3_up, where='post', label=label3, color='darkred', linewidth=3)
        for i, bit in enumerate(s3):
            axs[2].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[2].set_ylabel(title3)
        leg1 = axs[2].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)
        axs[2].set_yticks([0, 1])
        setup_grid(axs[2])

        # Configuração do layout
        plt.xlabel('Bits')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Salvar ou mostrar o gráfico
        self._save_or_show(fig, save_path)

    def plot_trellis(self, trellis, num_steps=5, initial_state=0, save_path=None):
        states_per_time = defaultdict(set)
        states_per_time[0].add(initial_state)
        branches = []
        for t in range(num_steps):
            for state in states_per_time[t]:
                for bit in [0, 1]:
                    next_state, out = trellis[state][bit]
                    module = sum(np.abs(out))
                    branches.append((t, state, bit, next_state, module, out))
                    states_per_time[t+1].add(next_state)

        all_states = sorted(set(s for states in states_per_time.values() for s in states))
        state_to_x = {s: i for i, s in enumerate(all_states)}
        num_states = len(all_states)

        fig = plt.figure(figsize=(0.50*num_states, 1*num_steps))
        ax = fig.gca()
        ax.set_xlabel('Estado')
        ax.set_ylabel('Intervalo de tempo')
        ax.set_xticks(range(num_states))

        # Formata os labels do eixo x com dois dígitos hexadecimais
        ax.set_xticklabels([f"{hex(s)[2:].upper():0>2}" for s in all_states])
        ax.set_yticks(range(num_steps+1))
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax.grid(True, axis='y', linestyle=':', alpha=0.2)
        ax.invert_yaxis()

        for t, state, bit, next_state, module, out in branches:
            x = [state_to_x[state], state_to_x[next_state]]
            y = [t, t+1]
            color = 'C0' if bit == 0 else 'C1'
            ax.plot(x, y, color=color, lw=2, alpha=0.8)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='C0', lw=2, label='Bit de entrada 0'),
            Line2D([0], [0], color='C1', lw=2, label='Bit de entrada 1')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=20)

        for t in range(num_steps+1):
            for state in states_per_time[t]:
                ax.plot(state_to_x[state], t, 'o', color='k', markersize=8)

        self._save_or_show(fig, save_path)

    def plot_scrambler(self, s1, s2, s3, s4, s5, s6, label1, label2, label3, label4, label5, label6, save_path=None):
        
        fig, axs = plt.subplots(3, 2, figsize=(16, 9), sharex=True)

        superplot(axs[0, 0], s1, label1, "navy")
        superplot(axs[0, 1], s2, label2, "darkred")
        superplot(axs[1, 0], s3, label3, "navy")
        superplot(axs[1, 1], s4, label4, "darkred")
        superplot(axs[2, 0], s5, label5, "navy")
        superplot(axs[2, 1], s6, label6, "darkred")

        axs[2, 0].set_xlabel("Bits")
        axs[2, 1].set_xlabel("Bits")
        axs[0, 0].set_ylabel("Original")
        axs[1, 0].set_ylabel("Embaralhado")
        axs[2, 0].set_ylabel("Desembaralhado")
        

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        self._save_or_show(fig, save_path)

    def plot_preamble(self, s1, s2, label1, label2, title1, title2, save_path=None):

        # Superamostragem do sinal para o plot
        s1_up = np.repeat(s1, 2)
        s2_up = np.repeat(s2, 2)
        x = np.arange(len(s1_up))
        bit_edges = np.arange(0, len(s1_up) + 1, 2)

        # Configuração do gráfico
        fig, axs = plt.subplots(2, 1, sharex=True)
        def setup_grid(ax):
            ax.set_xlim(0, len(s1_up))
            ax.set_ylim(-0.2, 1.4)
            ax.grid(False)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
        
        # Canal I
        axs[0].step(x, s1_up, where='post', label=label1, color='navy', linewidth=3)
        for i, bit in enumerate(s1):
            axs[0].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[0].set_ylabel(title1)
        leg0 = axs[0].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        axs[0].set_yticks([0, 1])
        setup_grid(axs[0])

        # Canal Q
        axs[1].step(x, s2_up, where='post', label=label2, color='darkred', linewidth=3)
        for i, bit in enumerate(s2):
            axs[1].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')
        axs[1].set_ylabel(title2)
        leg1 = axs[1].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)
        axs[1].set_yticks([0, 1])
        setup_grid(axs[1])

        # Configuração do layout
        plt.xlabel('Bits')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Salvar ou mostrar o gráfico
        self._save_or_show(fig, save_path)

    def plot_mux(self, s1, s2, s3, s4, label1, label2, label3, label4, title1, title2, save_path=None):
        # Concatenação
        s5 = np.concatenate((s1, s3))
        s6 = np.concatenate((s2, s4))

        # Superamostragem
        s5_up = np.repeat(s5, 2)
        s6_up = np.repeat(s6, 2)
        x = np.arange(len(s5_up))
        bit_edges = np.arange(0, len(s5_up) + 1, 2)

        fig, axs = plt.subplots(2, 1, sharex=True)

        def setup_grid(ax):
            ax.set_xlim(0, len(s5_up))
            ax.set_ylim(-0.2, 1.4)
            ax.grid(False)
            for pos in bit_edges:
                ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)

        # --- Channel I ---
        sep_I = len(s1) * 2
        x_I1 = x[:sep_I]
        x_I2 = x[sep_I - 1:]  # Inclui o último ponto anterior

        y_I1 = s5_up[:sep_I]
        y_I2 = s5_up[sep_I - 1:]

        axs[0].step(x_I1, y_I1, where='post', color='navy', linewidth=3, label=label1)
        axs[0].step(x_I2, y_I2, where='post', color='darkred', linewidth=3, label=label2)

        for i, bit in enumerate(s5):
            axs[0].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')

        axs[0].set_ylabel(title1)
        leg0 = axs[0].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg0.get_frame().set_facecolor('white')
        leg0.get_frame().set_edgecolor('black')
        leg0.get_frame().set_alpha(1.0)
        axs[0].set_yticks([0, 1])
        axs[0].set_yticks([0, 1])
        setup_grid(axs[0])

        # --- Channel Q ---
        sep_Q = len(s2) * 2
        x_Q1 = x[:sep_Q]
        x_Q2 = x[sep_Q - 1:]

        y_Q1 = s6_up[:sep_Q]
        y_Q2 = s6_up[sep_Q - 1:]

        axs[1].step(x_Q1, y_Q1, where='post', color='navy', linewidth=3, label=label3)
        axs[1].step(x_Q2, y_Q2, where='post', color='darkred', linewidth=3, label=label4)

        for i, bit in enumerate(s6):
            axs[1].text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=16, fontweight='bold')

        axs[1].set_ylabel(title2)
        leg1 = axs[1].legend(
                    loc='upper right', frameon=True, edgecolor='black',
                    facecolor='white', fontsize=12, fancybox=True
                )
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)
        axs[1].set_yticks([0, 1])
        setup_grid(axs[1])

        # Layout final
        plt.xlabel('Bits')

        num_bits = len(s5)
        step = 5
        axs[1].set_xticks(np.arange(0, num_bits * 2, step * 2))      
        axs[1].set_xticklabels(np.arange(0, num_bits, step)) 

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        self._save_or_show(fig, save_path)

    def plot_filter(self, h, t_rc, tb, span, fs, s1, s2, label_h, label1, label2, title_h, title1, title2, t_xlim, save_path=None):

        t_interp = np.arange(len(s1)) / fs

        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

        # Pulso RRC
        ax_rcc = fig.add_subplot(gs[0, :])
        ax_rcc.plot(t_rc, h, label=label_h, color='red', linewidth=2)
        ax_rcc.set_title(title_h)
        ax_rcc.set_ylabel('Amplitude')
        ax_rcc.grid(True)
        leg_rcc = ax_rcc.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_rcc.get_frame().set_facecolor('white')
        leg_rcc.get_frame().set_edgecolor('black')
        leg_rcc.get_frame().set_alpha(1.0)
        ax_rcc.set_xlim(-span*tb, span*tb)

        # Sinal I
        ax_I = fig.add_subplot(gs[1, 0])
        ax_I.plot(t_interp, s1, label= label1, color='navy', linewidth=2)
        ax_I.set_title(title1)
        ax_I.set_xlabel('Tempo (s)')
        ax_I.set_ylabel('Amplitude')
        ax_I.set_xlim(0, t_xlim)
        ax_I.grid(True)
        leg_I = ax_I.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_I.get_frame().set_facecolor('white')
        leg_I.get_frame().set_edgecolor('black')
        leg_I.get_frame().set_alpha(1.0)

        # Sinal Q
        ax_Q = fig.add_subplot(gs[1, 1])
        ax_Q.plot(t_interp, s2, label=label2, color='darkgreen', linewidth=2)
        ax_Q.set_title(title2)
        ax_Q.set_xlabel('Tempo (s)')
        ax_Q.set_xlim(0, t_xlim)
        ax_Q.grid(True)
        leg_Q = ax_Q.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_Q.get_frame().set_facecolor('white')
        leg_Q.get_frame().set_edgecolor('black')
        leg_Q.get_frame().set_alpha(1.0)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4)

        self._save_or_show(fig, save_path)

    def plot_modulation_time(self, s1, s2, s3, label1, label2, label3, title1, title2, fs, t_xlim, save_path=None):
        t = np.arange(len(s1)) / fs
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, s1, label=label1, color='navy', linewidth=2)
        ax1.plot(t, s2, label=label2, color='darkgreen', linewidth=2)
        ax1.set_xlim(0, t_xlim)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(title1)
        ax1.grid(True)
        leg1 = ax1.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='white', fontsize=14, fancybox=True)
        leg1.get_frame().set_facecolor('white')
        leg1.get_frame().set_edgecolor('black')
        leg1.get_frame().set_alpha(1.0)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(t, s3, label=label3, color='darkred', linewidth=1.5)
        ax2.set_xlim(0, t_xlim)
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(title2)
        ax2.grid(True)
        leg2 = ax2.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='white', fontsize=14, fancybox=True)
        leg2.get_frame().set_facecolor('white')
        leg2.get_frame().set_edgecolor('black')
        leg2.get_frame().set_alpha(1.0)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)

        self._save_or_show(fig, save_path)

    def plot_modulation_eye(self, s1, s2, label1, label2, title1, title2, fs, Rb, save_path=None):
        sps = int(fs / Rb)
        eye_len = 2 * sps
        n_traces = (len(s1) - eye_len) // sps
        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # Diagrama de olho dI
        ax_eyeI = fig.add_subplot(gs[0, 0])
        for i in range(n_traces):
            start = i * sps
            ax_eyeI.plot(np.arange(eye_len) / fs * 1e3, s1[start:start+eye_len], color='navy', alpha=0.18, linewidth=2, label=label1)
        ax_eyeI.set_xlabel('Tempo (ms)')
        ax_eyeI.set_ylabel('Amplitude')
        ax_eyeI.set_title(title1)
        ax_eyeI.grid(True)
        ax_eyeI.set_xlim(0, eye_len / fs * 1e3)

        # Diagrama de olho dQ
        ax_eyeQ = fig.add_subplot(gs[0, 1])
        for i in range(n_traces):
            start = i * sps
            ax_eyeQ.plot(np.arange(eye_len) / fs * 1e3, s2[start:start+eye_len], color='darkgreen', alpha=0.18, linewidth=2, label=label2)
        ax_eyeQ.set_xlabel('Tempo (ms)')
        ax_eyeQ.set_ylabel('Amplitude')
        ax_eyeQ.set_title(title2)
        ax_eyeQ.grid(True)
        ax_eyeQ.set_xlim(0, eye_len / fs * 1e3)

        plt.tight_layout()
        self._save_or_show(fig, save_path)

    def plot_modulation_iq(self, s1, s2, label1, label2, title1, title2, save_path=None, amplitude=None):

        s1_c = s1 - np.mean(s1)
        s2_c = s2 - np.mean(s2)
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        ax_const = fig.add_subplot(gs[0, 0])
        if amplitude is None:
            amp = np.mean(np.abs(np.concatenate([s1_c*1.1, s2_c*1.1])))
        else:
            amp = amplitude

        ax_iq = fig.add_subplot(gs[0, 1])
        ax_iq.scatter(s1_c, s2_c, color='darkgreen', alpha=0.5, s=2, label=label1)
        ax_iq.set_xlabel(r'$I$')
        ax_iq.set_ylabel(r'$Q$')
        ax_iq.set_title(title1)

        for v in [-amp, amp]:
            ax_iq.axhline(v, color='darkred', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
            ax_iq.axvline(v, color='darkred', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
        from matplotlib.lines import Line2D
        custom_legend = [Line2D([0], [0], marker='o', color='w', label='Amostras IQ',
                                markerfacecolor='darkgreen', markersize=16, alpha=0.7)]
        leg_iq = ax_iq.legend(handles=custom_legend,
                loc='upper right', frameon=True, edgecolor='black',
                facecolor='white', fontsize=12, fancybox=True
            )
        leg_iq.get_frame().set_facecolor('white')
        leg_iq.get_frame().set_edgecolor('black')
        leg_iq.get_frame().set_alpha(1.0)
        ax_iq.set_xlim(-0.06, 0.06)
        ax_iq.set_ylim(-0.06, 0.06)
        ax_iq.set_aspect('equal')

        qpsk_points = np.array([[amp, amp], [amp, -amp], [-amp, amp], [-amp, -amp]])
        ax_const.scatter(qpsk_points[:, 0], qpsk_points[:, 1], color='black', s=160, marker='o', label=label2, linewidth=5)
        ax_const.set_xlabel(r'$I$')
        ax_const.set_ylabel(r'$Q$')
        ax_const.set_title(title2)

        for v in [-amp, amp]:
            ax_const.axhline(v, color='darkred', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
            ax_const.axvline(v, color='darkred', linestyle='--', linewidth=1, alpha=0.6, zorder=0)
        leg_const = ax_const.legend(
                loc='upper right', frameon=True, edgecolor='black',
                facecolor='white', fontsize=12, fancybox=True
            )
        leg_const.get_frame().set_facecolor('white')
        leg_const.get_frame().set_edgecolor('black')
        leg_const.get_frame().set_alpha(1.0)
        ax_const.set_xlim(-0.06, 0.06)
        ax_const.set_ylim(-0.06, 0.06)
        ax_const.set_aspect('equal')

        plt.tight_layout()
        self._save_or_show(fig, save_path)

    def plot_modulation_freq(self, s1, s2, s3, label1, label2, label3, title1, title2, title3, fs, fc, save_path=None):

        freqs = np.fft.fftshift(np.fft.fftfreq(len(s3), d=1/fs))
        fft_dI = np.fft.fftshift(np.fft.fft(s1))
        fft_dQ = np.fft.fftshift(np.fft.fft(s2))
        fft_s = np.fft.fftshift(np.fft.fft(s3))

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2)

        if fc > 1000:
            freqs = freqs / 1000
            x_label = "Frequência (kHz)"
            x_limit = (-2.5 * fc / 1000, 2.5 * fc / 1000)
            x_limit_comp = (- 0.5 * fc / 1000, 0.5 * fc / 1000)
        else:
            x_label = "Frequência (Hz)"
            x_limit = (-2.5 * fc, 2.5 * fc)
            x_limit_comp = (- 0.5 * fc, 0.5 * fc)


        # Linha 1 - espectro de s(t)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(freqs, mag2db(fft_s), color='darkred', label=label3)
        ax1.set_xlim(x_limit)
        ax1.set_xlabel(x_label)
        ax1.set_ylim(-80, 5)
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title(title3)
        ax1.grid(True)
        ax1.legend()

        # Linha 2 - espectro de dI(t)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(freqs, mag2db(fft_dI), color='navy', label=label1)
        ax2.set_xlim(x_limit_comp)
        ax2.set_xlabel(x_label)
        ax2.set_ylim(-80, 5)
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title(title1)
        ax2.grid(True)
        ax2.legend()

        # Linha 2 - espectro de dQ(t)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(freqs, mag2db(fft_dQ), color='darkgreen', label=label2)
        ax3.set_xlim(x_limit_comp)
        ax3.set_xlabel(x_label)
        ax3.set_ylim(-80, 5)
        ax3.set_ylabel('Magnitude (dB)')    
        ax3.set_title(title2)
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        self._save_or_show(fig, save_path)


    def _save_or_show(self, fig, path):
        if path:
            base_dir = os.path.dirname(os.path.abspath(__main__.__file__))
            full_path = os.path.join(base_dir, path)

            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            fig.savefig(full_path)
        else:
            plt.show()