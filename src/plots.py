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
    r"""
    Converte a magnitude do sinal para escala logarítmica (dB), seguindo a expressão: 

    $$
    Signal_{DB} = 20 \cdot \log_{10}(|signal| + 10^{-12})
    $$

    Nota: 
        O termo $10^{-12}$ é adicionado para evitar logaritmo de zero, que é indefinido.
    """
    mag = np.abs(signal)
    mag /= np.max(mag)
    return 20 * np.log10(mag + 1e-12)

def superplot(ax, signal, label, color):
    r""" Plota um sinal no eixo especificado, com superamostragem para melhor visualização.
    Args:
        ax (matplotlib.axes.Axes): Eixo onde o sinal será plotado.
        signal (np.ndarray): Sinal a ser plotado.
        label (str): Rótulo do sinal.
        color (str): Cor do sinal.
    """
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
        r""" 
        Inicializa uma instância do plotter de gráficos.        
        """
        pass

    def plot_time_domain(self, s1, s2, t, label1, label2, title1, title2, save_path=None):
        r"""
        Plota os sinais no domínio do tempo.

        ![timed](assets/transmitter_modulator_time.svg)

        Args:
            s1 (np.ndarray): Primeiro sinal a ser plotado.
            s2 (np.ndarray): Segundo sinal a ser plotado.
            t (np.ndarray): Vetor de tempo correspondente aos sinais.
            label1 (str): Rótulo do primeiro sinal.
            label2 (str): Rótulo do segundo sinal.
            title1 (str): Título do primeiro gráfico.
            title2 (str): Título do segundo gráfico.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """

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
        r"""
        Plota os sinais no domínio da frequência.

        ![freqd](assets/transmitter_modulator_freq.svg)

        Args:
            s1 (np.ndarray): Primeiro sinal a ser plotado.
            s2 (np.ndarray): Segundo sinal a ser plotado.
            fs (float): Frequência de amostragem dos sinais.
            fc (float): Frequência central para o eixo x.
            label1 (str): Rótulo do primeiro sinal.
            label2 (str): Rótulo do segundo sinal.
            title1 (str): Título do primeiro gráfico.
            title2 (str): Título do segundo gráfico.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """

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
        r"""
        Plota uma sequência de bits, com a opção de destacar seções específicas.

        ![bits](assets/transmitter_datagram.svg)

        Args:
            bits_list (list of np.ndarray): Lista de arrays de bits a serem plotados.
            sections (list of tuples, optional): Lista de seções a destacar, cada uma como (nome, comprimento).
            colors (list of str, optional): Cores para as seções destacadas.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """
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
        r"""
        Plota os sinais de entrada e saída do codificador.
        ![encode](assets/transmitter_encoder.svg)

        Args:
            s1 (np.ndarray): Sinal de entrada I.
            s2 (np.ndarray): Sinal de entrada Q.
            s3 (np.ndarray): Sinal de saída I codificado.
            s4 (np.ndarray): Sinal de saída Q codificado.
            label1 (str): Rótulo do sinal I de entrada.
            label2 (str): Rótulo do sinal Q de entrada.
            label3 (str): Rótulo do sinal I codificado.
            label4 (str): Rótulo do sinal Q codificado.
            title1 (str): Título do sinal I de entrada.
            title2 (str): Título do sinal Q de entrada.
            title3 (str): Título do sinal I codificado.
            title4 (str): Título do sinal Q codificado.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """
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
        r"""
        Plota os sinais de entrada e saída do codificador convolucional.

        ![conv](assets/transmitter_convolutional.svg)

        Args:
            s1 (np.ndarray): Sinal de entrada I.
            s2 (np.ndarray): Sinal de entrada Q.
            s3 (np.ndarray): Sinal de saída codificado.
            label1 (str): Rótulo do sinal I de entrada.
            label2 (str): Rótulo do sinal Q de entrada.
            label3 (str): Rótulo do sinal codificado.
            title1 (str): Título do sinal I de entrada.
            title2 (str): Título do sinal Q de entrada.
            title3 (str): Título do sinal codificado.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """

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
        r"""
        Plota o diagrama de treliça de um codificador convolucional.

        ![trellis'](assets/example_trelica.svg)
        
        Args:
            trellis (dict): Dicionário representando o treliça, onde as chaves são estados e os valores são tuplas (próximo_estado, saída).
            num_steps (int): Número de passos no tempo a serem plotados.
            initial_state (int): Estado inicial do treliça.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        
        """
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

    # TODO: adicionar indexes pra o plot do embaralhador.
    def plot_scrambler(self, s1, s2, s3, s4, s5, s6, label1, label2, label3, label4, label5, label6, save_path=None):
        r"""
        Plota os sinais de entrada e saída do embaralhador.
        
        ![scrambler](assets/example_scrambling.svg)

        Args:
            s1 (np.ndarray): Sinal de entrada I.
            s2 (np.ndarray): Sinal de entrada Q.
            s3 (np.ndarray): Sinal embaralhado I.
            s4 (np.ndarray): Sinal embaralhado Q.
            s5 (np.ndarray): Sinal desembaralhado I.
            s6 (np.ndarray): Sinal desembaralhado Q.
            label1 (str): Rótulo do sinal I de entrada.
            label2 (str): Rótulo do sinal Q de entrada.
            label3 (str): Rótulo do sinal embaralhado I.
            label4 (str): Rótulo do sinal embaralhado Q.
            label5 (str): Rótulo do sinal desembaralhado I.
            label6 (str): Rótulo do sinal desembaralhado Q.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """
        
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
        r"""
        Plota o preâmbulo do transmissor, mostrando os sinais I e Q.

        ![preamble.](assets/transmitter_preamble.svg)
        
        Args:
            s1 (np.ndarray): Sinal I do preâmbulo.
            s2 (np.ndarray): Sinal Q do preâmbulo.
            label1 (str): Rótulo do sinal I.
            label2 (str): Rótulo do sinal Q.
            title1 (str): Título do sinal I.
            title2 (str): Título do sinal Q.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """

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
        r"""
        Plota o multiplexador do transmissor, mostrando os sinais I e Q.

        ![mux](assets/transmitter_multiplexing.svg)
        
        Args:
            s1 (np.ndarray): Sinal I do canal I.
            s2 (np.ndarray): Sinal Q do canal I.
            s3 (np.ndarray): Sinal I do canal Q.
            s4 (np.ndarray): Sinal Q do canal Q.
            label1 (str): Rótulo do sinal I do canal I.
            label2 (str): Rótulo do sinal Q do canal I.
            label3 (str): Rótulo do sinal I do canal Q.
            label4 (str): Rótulo do sinal Q do canal Q.
            title1 (str): Título do sinal I do canal I.
            title2 (str): Título do sinal Q do canal I.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """
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

        r"""
        Plota o filtro RRC e os sinais I e Q após a filtragem.

        ![filter](assets/transmitter_filter.svg)

        Args:
            h (np.ndarray): Resposta ao impulso do filtro RRC.
            t_rc (np.ndarray): Tempo correspondente à resposta ao impulso.
            tb (float): Tempo de símbolo.
            span (int): Número de spans do filtro.
            fs (float): Frequência de amostragem.
            s1 (np.ndarray): Sinal I filtrado.
            s2 (np.ndarray): Sinal Q filtrado.
            label_h (str): Rótulo do filtro.
            label1 (str): Rótulo do sinal I.
            label2 (str): Rótulo do sinal Q.
            title_h (str): Título do filtro.
            title1 (str): Título do sinal I.
            title2 (str): Título do sinal Q.
            t_xlim (float): Limite do eixo x para os sinais I e Q.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """

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
        r"""
        Plota os sinais de modulação no domínio do tempo.

        ![modulation_time](assets/transmitter_modulator_time.svg)

        Args:
            s1 (np.ndarray): Sinal I modulado.
            s2 (np.ndarray): Sinal Q modulado.
            s3 (np.ndarray): Sinal de modulação resultante.
            label1 (str): Rótulo do sinal I.
            label2 (str): Rótulo do sinal Q.
            label3 (str): Rótulo do sinal de modulação.
            title1 (str): Título do sinal I.
            title2 (str): Título do sinal Q.
            fs (float): Frequência de amostragem.
            t_xlim (float): Limite do eixo x para os sinais I e Q.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """

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
        r"""
        Plota o diagrama de olho dos sinais I e Q após a modulação.

        ![modulation_eye](assets/example_eye.png)

        Args:
            s1 (np.ndarray): Sinal I modulado.
            s2 (np.ndarray): Sinal Q modulado.
            label1 (str): Rótulo do sinal I.
            label2 (str): Rótulo do sinal Q.
            title1 (str): Título do sinal I.
            title2 (str): Título do sinal Q.
            fs (float): Frequência de amostragem.
            Rb (float): Taxa de bits.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """

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

        r"""
        Plota o diagrama IQ dos sinais I e Q após a modulação.

        ![modulation_iq](assets/transmitter_modulator_iq.png)
        
        Args:
            s1 (np.ndarray): Sinal I modulado.
            s2 (np.ndarray): Sinal Q modulado.
            label1 (str): Rótulo do sinal I.
            label2 (str): Rótulo do sinal Q.
            title1 (str): Título do sinal I.
            title2 (str): Título do sinal Q.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
            amplitude (float, optional): Amplitude dos pontos QPSK. Se None, a amplitude é calculada automaticamente.
        """

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
        r"""
        Plota o espectro dos sinais I, Q e o sinal modulado no domínio da frequência.

        ![modulation_freq](assets/transmitter_modulator_freq.svg)

        Args:
            s1 (np.ndarray): Sinal I modulado.
            s2 (np.ndarray): Sinal Q modulado.
            s3 (np.ndarray): Sinal modulado resultante.
            label1 (str): Rótulo do sinal I.
            label2 (str): Rótulo do sinal Q.
            label3 (str): Rótulo do sinal modulado.
            title1 (str): Título do sinal I.
            title2 (str): Título do sinal Q.
            title3 (str): Título do sinal modulado.
            fs (float): Frequência de amostragem.
            fc (float): Frequência de portadora.
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """

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

    def plot_demodulation_freq(self, s1, s2, label1, label2, title1, title2, fs, fc, save_path=None):
        fig_fft_prod = plt.figure(figsize=(16, 8))
        gs_fft = gridspec.GridSpec(2, 1)

        # FFT de y_I_
        YI_f = np.fft.fftshift(np.fft.fft(s1))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(YI_f), d=1/fs))
        YI_db = mag2db(YI_f)

        ax_fft_i = fig_fft_prod.add_subplot(gs_fft[0])
        ax_fft_i.plot(freqs, YI_db, color='blue', label=r"$|X_I(f)|$")
        ax_fft_i.set_xlim(-2.5 * fc, 2.5 * fc)
        ax_fft_i.set_ylim(-60, 5)
        ax_fft_i.set_title(r"Espectro do canal I - $x_I(t)$")
        ax_fft_i.set_xlabel("Frequência (Hz)")
        ax_fft_i.set_ylabel("Magnitude (dB)")
        ax_fft_i.grid(True)
        ax_fft_i.legend()
        leg_fft_i = ax_fft_i.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_fft_i.get_frame().set_facecolor('white')
        leg_fft_i.get_frame().set_edgecolor('black')
        leg_fft_i.get_frame().set_alpha(1.0)

        # FFT de y_Q_
        YQ_f = np.fft.fftshift(np.fft.fft(s2))
        YQ_db = mag2db(YQ_f)

        ax_fft_q = fig_fft_prod.add_subplot(gs_fft[1])
        ax_fft_q.plot(freqs, YQ_db, color='green', label=r"$|Y_Q(f)|$")
        ax_fft_q.set_xlim(-2.5 * fc, 2.5 * fc)
        ax_fft_q.set_ylim(-60, 5)
        ax_fft_q.set_title(r"Espectro do canal Q - $y_Q(t)$")
        ax_fft_q.set_xlabel("Frequência (Hz)")
        ax_fft_q.set_ylabel("Magnitude (dB)")
        ax_fft_q.grid(True)
        ax_fft_q.legend()
        leg_fft_q = ax_fft_q.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_fft_q.get_frame().set_facecolor('white')
        leg_fft_q.get_frame().set_edgecolor('black')
        leg_fft_q.get_frame().set_alpha(1.0)

        plt.tight_layout()
        self._save_or_show(fig_fft_prod, save_path)

    def plot_freq_receiver(self, y_I, y_Q, fs, fc, save_path=None):
        """
        Plota os espectros de frequência dos canais I e Q após a demodulação.

        Args:
            y_I (np.ndarray): Sinal do canal I no domínio do tempo
            y_Q (np.ndarray): Sinal do canal Q no domínio do tempo
            fs (float): Frequência de amostragem (Hz)
            fc (float): Frequência da portadora (Hz)
            save_path (str, optional): Caminho para salvar a figura. Se None, mostra a figura.
        """
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 1)

        # FFT do canal I
        YI_f = np.fft.fftshift(np.fft.fft(y_I))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(y_I), d=1/fs))
        YI_db = mag2db(YI_f)  # Normalizado

        ax_i = fig.add_subplot(gs[0])
        ax_i.plot(freqs, YI_db, color='blue', label=r"$|Y_I(f)|$")
        ax_i.set_xlim(-2.5 * fc, 2.5 * fc)
        ax_i.set_ylim(-60, 5)
        ax_i.set_title(r"Espectro do canal I - $y_I(t)$")
        ax_i.set_xlabel("Frequência (Hz)")
        ax_i.set_ylabel("Magnitude (dB)")
        ax_i.grid(True)
        leg_i = ax_i.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_i.get_frame().set_facecolor('white')
        leg_i.get_frame().set_edgecolor('black')
        leg_i.get_frame().set_alpha(1.0)

        # FFT do canal Q
        YQ_f = np.fft.fftshift(np.fft.fft(y_Q))
        YQ_db = mag2db(YQ_f)  # Normalizado

        ax_q = fig.add_subplot(gs[1])
        ax_q.plot(freqs, YQ_db, color='green', label=r"$|Y_Q(f)|$")
        ax_q.set_xlim(-2.5 * fc, 2.5 * fc)
        ax_q.set_ylim(-60, 5)
        ax_q.set_title(r"Espectro do canal Q - $y_Q(t)$")
        ax_q.set_xlabel("Frequência (Hz)")
        ax_q.set_ylabel("Magnitude (dB)")
        ax_q.grid(True)
        leg_q = ax_q.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_q.get_frame().set_facecolor('white')
        leg_q.get_frame().set_edgecolor('black')
        leg_q.get_frame().set_alpha(1.0)

        plt.tight_layout()
        self._save_or_show(fig, save_path)


    def plot_impulse_response(self, t_imp, impulse_response, label_imp, t_unit="ms", xlim=None, save_path=None):
        """
        Plota apenas a resposta ao impulso de um filtro.

        Args:
            t_imp (np.ndarray): Vetor de tempo da resposta ao impulso.
            impulse_response (np.ndarray): Amostras da resposta ao impulso.
            label_imp (str): Rótulo da resposta ao impulso.
            title_imp (str): Título do gráfico.
            t_unit (str, optional): Unidade de tempo no eixo X ("ms" ou "s"). Default é "ms".
            save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico será exibido na tela.
        """
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        if t_unit == "ms":
            t_plot = t_imp * 1000
            x_label = "Tempo (ms)"
        else:
            t_plot = t_imp
            x_label = "Tempo (s)"

        ax.plot(t_plot, impulse_response, color='red', label=label_imp, linewidth=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        if xlim is not None:
            ax.set_xlim(-xlim, xlim)

        leg = ax.legend(loc='upper right', frameon=True, edgecolor='black',
                        facecolor='white', fontsize=12, fancybox=True)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_alpha(1.0)

        plt.tight_layout()
        self._save_or_show(fig, save_path)

    def plot_filtered_signals(self, t_imp, impulse_response, t_interp, d_I_rec, d_Q_rec,
                              label_imp, label_I, label_Q,
                              title_imp, title_I, title_Q, t_xlim, save_path=None):
        """
        Plota a resposta ao impulso do filtro passa-baixa e os sinais I e Q filtrados.

        Args:
            t_imp (np.ndarray): Vetor de tempo da resposta ao impulso.
            impulse_response (np.ndarray): Amostras da resposta ao impulso.
            t_interp (np.ndarray): Vetor de tempo dos sinais filtrados.
            d_I_rec (np.ndarray): Sinal I filtrado.
            d_Q_rec (np.ndarray): Sinal Q filtrado.
            label_imp (str): Rótulo da resposta ao impulso.
            label_I (str): Rótulo do canal I filtrado.
            label_Q (str): Rótulo do canal Q filtrado.
            title_imp (str): Título do gráfico da resposta ao impulso.
            title_I (str): Título do gráfico do canal I.
            title_Q (str): Título do gráfico do canal Q.
            t_xlim (float): Limite do eixo X para os canais I e Q.
            save_path (str, optional): Caminho para salvar o gráfico.
        """

        fig_filt = plt.figure(figsize=(16, 10))
        gs_filt = gridspec.GridSpec(3, 1)

        # Resposta ao impulso
        ax_imp = fig_filt.add_subplot(gs_filt[0])
        ax_imp.plot(t_imp * 1000, impulse_response, color='red', label=label_imp, linewidth=2)
        ax_imp.set_title(title_imp)
        ax_imp.set_xlabel("Tempo (ms)")
        ax_imp.set_ylabel("Amplitude")
        ax_imp.grid(True)
        leg_imp = ax_imp.legend(loc='upper right', frameon=True, edgecolor='black',
                                facecolor='white', fontsize=12, fancybox=True)
        leg_imp.get_frame().set_facecolor('white')
        leg_imp.get_frame().set_edgecolor('black')
        leg_imp.get_frame().set_alpha(1.0)

        # I filtrado
        ax_fi = fig_filt.add_subplot(gs_filt[1])
        ax_fi.plot(t_interp, d_I_rec, color='blue', label=label_I)
        ax_fi.set_title(title_I)
        ax_fi.set_xlim(0, t_xlim)
        ax_fi.set_xlabel("Tempo (s)")
        ax_fi.set_ylabel("Amplitude")
        ax_fi.grid(True)
        leg_fi = ax_fi.legend(loc='upper right', frameon=True, edgecolor='black',
                              facecolor='white', fontsize=12, fancybox=True)
        leg_fi.get_frame().set_facecolor('white')
        leg_fi.get_frame().set_edgecolor('black')
        leg_fi.get_frame().set_alpha(1.0)

        # Q filtrado
        ax_fq = fig_filt.add_subplot(gs_filt[2])
        ax_fq.plot(t_interp, d_Q_rec, color='green', label=label_Q)
        ax_fq.set_title(title_Q)
        ax_fq.set_xlim(0, t_xlim)
        ax_fq.set_xlabel("Tempo (s)")
        ax_fq.set_ylabel("Amplitude")
        ax_fq.grid(True)
        leg_fq = ax_fq.legend(loc='upper right', frameon=True, edgecolor='black',
                              facecolor='white', fontsize=12, fancybox=True)
        leg_fq.get_frame().set_facecolor('white')
        leg_fq.get_frame().set_edgecolor('black')
        leg_fq.get_frame().set_alpha(1.0)

        plt.tight_layout()
        self._save_or_show(fig_filt, save_path)

    def plot_lowpass_filter(self, t_imp, impulse_response, t_interp, d_I_rec, d_Q_rec, t_xlim=0.1, save_path=None):
        """
        Plota a resposta ao impulso de um filtro passa-baixa e os sinais I e Q após filtragem.

        Args:
            t_imp (np.ndarray): Vetor de tempo da resposta ao impulso (em segundos).
            impulse_response (np.ndarray): Amostras da resposta ao impulso.
            t_interp (np.ndarray): Vetor de tempo dos sinais filtrados (em segundos).
            d_I_rec (np.ndarray): Sinal I filtrado.
            d_Q_rec (np.ndarray): Sinal Q filtrado.
            t_xlim (float): Limite do eixo X (em segundos) para os sinais filtrados.
            save_path (str, optional): Caminho para salvar a figura. Se None, exibe na tela.
        """
        fig_filt = plt.figure(figsize=(16, 10))
        gs_filt = gridspec.GridSpec(3, 1)

        # Resposta ao impulso
        ax_imp = fig_filt.add_subplot(gs_filt[0])
        ax_imp.plot(t_imp * 1000, impulse_response, color='red', label='Resposta ao Impulso - FPB', linewidth=2)
        ax_imp.set_title("Resposta ao Impulso do Filtro Passa-Baixa")
        ax_imp.set_xlim(0, 2)
        ax_imp.set_xlabel("Tempo (ms)")
        ax_imp.set_ylabel("Amplitude")
        ax_imp.grid(True)
        leg_imp = ax_imp.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_imp.get_frame().set_facecolor('white')
        leg_imp.get_frame().set_edgecolor('black')
        leg_imp.get_frame().set_alpha(1.0)

        # I filtrado
        ax_fi = fig_filt.add_subplot(gs_filt[1])
        ax_fi.plot(t_interp, d_I_rec, color='blue', label=r"$d_I(t)$ filtrado")
        ax_fi.set_title("Canal I após filtragem passa-baixa")
        ax_fi.set_xlim(0, t_xlim)
        ax_fi.set_xlabel("Tempo (s)")
        ax_fi.set_ylabel("Amplitude")
        ax_fi.grid(True)
        leg_fi = ax_fi.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_fi.get_frame().set_facecolor('white')
        leg_fi.get_frame().set_edgecolor('black')
        leg_fi.get_frame().set_alpha(1.0)

        # Q filtrado
        ax_fq = fig_filt.add_subplot(gs_filt[2])
        ax_fq.plot(t_interp, d_Q_rec, color='green', label=r"$d_Q(t)$ filtrado")
        ax_fq.set_title("Canal Q após filtragem passa-baixa")
        ax_fq.set_xlim(0, t_xlim)
        ax_fq.set_xlabel("Tempo (s)")
        ax_fq.set_ylabel("Amplitude")
        ax_fq.grid(True)
        leg_fq = ax_fq.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_fq.get_frame().set_facecolor('white')
        leg_fq.get_frame().set_edgecolor('black')
        leg_fq.get_frame().set_alpha(1.0)

        plt.tight_layout()
        self._save_or_show(fig_filt, save_path)

    def plot_lowpass_freq(self, t_imp, impulse_response,
                                       y_I_, y_Q_, d_I_rec, d_Q_rec,
                                       fs, fc, save_path=None):
        """
        Plota a resposta ao impulso de um filtro passa-baixa e os espectros
        antes e depois da filtragem para os canais I e Q.

        Args:
            t_imp (np.ndarray): Vetor de tempo da resposta ao impulso (em segundos).
            impulse_response (np.ndarray): Resposta ao impulso do filtro.
            y_I_ (np.ndarray): Sinal I antes da filtragem.
            y_Q_ (np.ndarray): Sinal Q antes da filtragem.
            d_I_rec (np.ndarray): Sinal I após filtragem.
            d_Q_rec (np.ndarray): Sinal Q após filtragem.
            fs (float): Frequência de amostragem (Hz).
            fc (float): Frequência central para o plot (Hz).
            save_path (str, optional): Caminho para salvar a figura. Se None, exibe na tela.
        """
        fig_spec = plt.figure(figsize=(16, 10))
        gs_spec = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

        # Linha 1 (0, :) - Resposta ao impulso
        ax_impulse = fig_spec.add_subplot(gs_spec[0, :])
        ax_impulse.plot(t_imp * 1000, impulse_response, color='red', linewidth=2, label='Resposta ao Impulso - FPB')
        ax_impulse.set_title("Resposta ao Impulso do Filtro Passa-Baixa")
        ax_impulse.set_xlabel("Tempo (ms)")
        ax_impulse.set_xlim(0, 2)
        ax_impulse.set_ylabel("Amplitude")
        ax_impulse.grid(True)
        leg_impulse = ax_impulse.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_impulse.get_frame().set_facecolor('white')
        leg_impulse.get_frame().set_edgecolor('black')
        leg_impulse.get_frame().set_alpha(1.0)

        # Frequências
        freqs = np.fft.fftshift(np.fft.fftfreq(len(y_I_), d=1/fs))

        # FFT antes da filtragem (y'_I)
        YI_db = mag2db(np.fft.fftshift(np.fft.fft(y_I_)))
        ax_yi = fig_spec.add_subplot(gs_spec[1, 0])
        ax_yi.plot(freqs, YI_db, color='blue', label=r"$|X'_I(f)|$")
        ax_yi.set_xlim(-2.5 * fc, 2.5 * fc)
        ax_yi.set_ylim(-80, 5)
        ax_yi.set_title(r"Espectro de $x'_I(t)$ (Antes do FPB)")
        ax_yi.set_xlabel("Frequência (Hz)")
        ax_yi.set_ylabel("Magnitude (dB)")
        ax_yi.grid(True)
        leg_yi = ax_yi.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_yi.get_frame().set_facecolor('white')
        leg_yi.get_frame().set_edgecolor('black')
        leg_yi.get_frame().set_alpha(1.0)

        # FFT depois da filtragem (d_I)
        DI_db = mag2db(np.fft.fftshift(np.fft.fft(d_I_rec)))
        ax_di = fig_spec.add_subplot(gs_spec[1, 1])
        ax_di.plot(freqs, DI_db, color='darkblue', label=r"$|d'_I(f)|$")
        ax_di.set_xlim(-2.5 * fc, 2.5 * fc)
        ax_di.set_ylim(-80, 5)
        ax_di.set_title(r"Espectro de $d'_I(t)$ (Após FPB)")
        ax_di.set_xlabel("Frequência (Hz)")
        ax_di.set_ylabel("Magnitude (dB)")
        ax_di.grid(True)
        leg_di = ax_di.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_di.get_frame().set_facecolor('white')
        leg_di.get_frame().set_edgecolor('black')
        leg_di.get_frame().set_alpha(1.0)

        # FFT antes da filtragem (y'_Q)
        YQ_db = mag2db(np.fft.fftshift(np.fft.fft(y_Q_)))
        ax_yq = fig_spec.add_subplot(gs_spec[2, 0])
        ax_yq.plot(freqs, YQ_db, color='green', label=r"$|Y'_Q(f)|$")
        ax_yq.set_xlim(-2.5 * fc, 2.5 * fc)
        ax_yq.set_ylim(-90, 5)
        ax_yq.set_title(r"Espectro de $y'_Q(t)$ (Antes do FPB)")
        ax_yq.set_xlabel("Frequência (Hz)")
        ax_yq.set_ylabel("Magnitude (dB)")
        ax_yq.grid(True)
        leg_yq = ax_yq.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_yq.get_frame().set_facecolor('white')
        leg_yq.get_frame().set_edgecolor('black')
        leg_yq.get_frame().set_alpha(1.0)

        # FFT depois da filtragem (d_Q)
        DQ_db = mag2db(np.fft.fftshift(np.fft.fft(d_Q_rec)))
        ax_dq = fig_spec.add_subplot(gs_spec[2, 1])
        ax_dq.plot(freqs, DQ_db, color='darkgreen', label=r"$|d_Q(f)|$")
        ax_dq.set_xlim(-2.5 * fc, 2.5 * fc)
        ax_dq.set_ylim(-90, 5)
        ax_dq.set_title(r"Espectro de $d'_Q(t)$ (Após FPB)")
        ax_dq.set_xlabel("Frequência (Hz)")
        ax_dq.set_ylabel("Magnitude (dB)")
        ax_dq.grid(True)
        leg_dq = ax_dq.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_dq.get_frame().set_facecolor('white')
        leg_dq.get_frame().set_edgecolor('black')
        leg_dq.get_frame().set_alpha(1.0)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4)
        self._save_or_show(fig_spec, save_path)

    def plot_matched_filter(self, t_rc, g_matched, t_matched, d_I_matched, d_Q_matched,
                            label_imp, label_I, label_Q,
                            title_imp, title_I, title_Q,
                            t_xlim=0.1, save_path=None):
        """
        Plota a resposta ao impulso do filtro casado e os sinais I e Q após a filtragem.

        Args:
            t_rc (np.ndarray): Vetor de tempo da resposta ao impulso do filtro casado.
            g_matched (np.ndarray): Amostras da resposta ao impulso.
            t_matched (np.ndarray): Vetor de tempo dos sinais filtrados.
            d_I_matched (np.ndarray): Sinal I filtrado.
            d_Q_matched (np.ndarray): Sinal Q filtrado.
            label_imp (str): Rótulo da resposta ao impulso.
            label_I (str): Rótulo do canal I filtrado.
            label_Q (str): Rótulo do canal Q filtrado.
            title_imp (str): Título do gráfico da resposta ao impulso.
            title_I (str): Título do gráfico do canal I.
            title_Q (str): Título do gráfico do canal Q.
            t_xlim (float, optional): Limite do eixo X para os canais I e Q.
            save_path (str, optional): Caminho para salvar o gráfico.
        """
        fig_match = plt.figure(figsize=(16, 10))
        gs_match = gridspec.GridSpec(3, 1)

        # Resposta ao impulso do filtro casado
        ax_mh = fig_match.add_subplot(gs_match[0])
        ax_mh.plot(t_rc * 1000, g_matched, color='red', label=label_imp)
        ax_mh.set_title(title_imp)
        ax_mh.set_xlim(-15, 15)
        ax_mh.set_xlabel("Tempo (ms)")
        ax_mh.set_ylabel("Amplitude")
        ax_mh.grid(True)
        leg_mh = ax_mh.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_mh.get_frame().set_facecolor('white')
        leg_mh.get_frame().set_edgecolor('black')
        leg_mh.get_frame().set_alpha(1.0)

        # Canal I após filtro casado
        ax_i_m = fig_match.add_subplot(gs_match[1])
        ax_i_m.plot(t_matched, d_I_matched, color='blue', label=label_I)
        ax_i_m.set_title(title_I)
        ax_i_m.set_xlim(0, t_xlim)
        ax_i_m.set_xlabel("Tempo (s)")
        ax_i_m.set_ylabel("Amplitude")
        ax_i_m.grid(True)
        leg_i_m = ax_i_m.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_i_m.get_frame().set_facecolor('white')
        leg_i_m.get_frame().set_edgecolor('black')
        leg_i_m.get_frame().set_alpha(1.0)

        # Canal Q após filtro casado
        ax_q_m = fig_match.add_subplot(gs_match[2])
        ax_q_m.plot(t_matched, d_Q_matched, color='green', label=label_Q)
        ax_q_m.set_title(title_Q)
        ax_q_m.set_xlim(0, t_xlim)
        ax_q_m.set_xlabel("Tempo (s)")
        ax_q_m.set_ylabel("Amplitude")
        ax_q_m.grid(True)
        leg_q_m = ax_q_m.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
        )
        leg_q_m.get_frame().set_facecolor('white')
        leg_q_m.get_frame().set_edgecolor('black')
        leg_q_m.get_frame().set_alpha(1.0)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4)

        self._save_or_show(fig_match, save_path)

    def plot_matched_filter_freq(self, t_rc, g_matched,
                                     d_I_rec, d_Q_rec,
                                     d_I_matched, d_Q_matched,
                                     fs, fc,
                                     label_imp, label_I_before, label_I_after,
                                     label_Q_before, label_Q_after,
                                     title_imp, title_I_before, title_I_after,
                                     title_Q_before, title_Q_after,
                                     save_path=None):
        """
        Plota o espectro dos sinais I e Q antes e depois da filtragem casada,
        junto com a resposta ao impulso do filtro casado.

        Args:
            t_rc (np.ndarray): Vetor de tempo da resposta ao impulso do filtro casado.
            g_matched (np.ndarray): Resposta ao impulso do filtro casado.
            d_I_rec (np.ndarray): Sinal I antes do filtro casado.
            d_Q_rec (np.ndarray): Sinal Q antes do filtro casado.
            d_I_matched (np.ndarray): Sinal I após o filtro casado.
            d_Q_matched (np.ndarray): Sinal Q após o filtro casado.
            fs (float): Frequência de amostragem (Hz).
            fc (float): Frequência central para o plot (Hz).
            label_imp (str): Rótulo da resposta ao impulso.
            label_I_before (str): Rótulo do canal I antes do filtro casado.
            label_I_after (str): Rótulo do canal I após o filtro casado.
            label_Q_before (str): Rótulo do canal Q antes do filtro casado.
            label_Q_after (str): Rótulo do canal Q após o filtro casado.
            title_imp (str): Título do gráfico da resposta ao impulso.
            title_I_before (str): Título do espectro do canal I antes do filtro casado.
            title_I_after (str): Título do espectro do canal I após o filtro casado.
            title_Q_before (str): Título do espectro do canal Q antes do filtro casado.
            title_Q_after (str): Título do espectro do canal Q após o filtro casado.
            save_path (str, optional): Caminho para salvar a figura.
        """
        # FFT antes do filtro casado
        DI_f = np.fft.fftshift(np.fft.fft(d_I_rec))
        DQ_f = np.fft.fftshift(np.fft.fft(d_Q_rec))

        # FFT após o filtro casado
        DIM_f = np.fft.fftshift(np.fft.fft(d_I_matched))
        DQM_f = np.fft.fftshift(np.fft.fft(d_Q_matched))

        freqs = np.fft.fftshift(np.fft.fftfreq(len(d_I_rec), d=1/fs))

        DI_db = mag2db(DI_f)
        DQ_db = mag2db(DQ_f)
        DIM_db = mag2db(DIM_f)
        DQM_db = mag2db(DQM_f)

        fig_match_spec = plt.figure(figsize=(16, 10))
        gs_match_spec = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

        # Resposta ao impulso
        ax_imp_m = fig_match_spec.add_subplot(gs_match_spec[0, :])
        ax_imp_m.plot(t_rc * 1000, g_matched, color='red', linewidth=2, label=label_imp)
        ax_imp_m.set_title(title_imp)
        ax_imp_m.set_xlim(-15, 15)
        ax_imp_m.set_xlabel("Tempo (ms)")
        ax_imp_m.set_ylabel("Amplitude")
        ax_imp_m.grid(True)
        leg_imp = ax_imp_m.legend(loc='upper right', frameon=True, edgecolor='black',
                                  facecolor='white', fontsize=12, fancybox=True)
        leg_imp.get_frame().set_facecolor('white')
        leg_imp.get_frame().set_edgecolor('black')
        leg_imp.get_frame().set_alpha(1.0)

        # Canal I antes
        ax_i_before = fig_match_spec.add_subplot(gs_match_spec[1, 0])
        ax_i_before.plot(freqs, DI_db, color='navy', label=label_I_before)
        ax_i_before.set_title(title_I_before)
        ax_i_before.set_xlabel("Frequência (Hz)")
        ax_i_before.set_ylabel("Magnitude (dB)")
        ax_i_before.set_xlim(-fc, fc)
        ax_i_before.set_ylim(-90, 5)
        ax_i_before.grid(True)
        leg_ib = ax_i_before.legend(loc='upper right', frameon=True, edgecolor='black',
                                    facecolor='white', fontsize=12, fancybox=True)
        leg_ib.get_frame().set_facecolor('white')
        leg_ib.get_frame().set_edgecolor('black')
        leg_ib.get_frame().set_alpha(1.0)

        # Canal I após
        ax_i_after = fig_match_spec.add_subplot(gs_match_spec[1, 1])
        ax_i_after.plot(freqs, DIM_db, color='navy', label=label_I_after)
        ax_i_after.set_title(title_I_after)
        ax_i_after.set_xlabel("Frequência (Hz)")
        ax_i_after.set_ylabel("Magnitude (dB)")
        ax_i_after.set_xlim(-fc, fc)
        ax_i_after.set_ylim(-90, 5)
        ax_i_after.grid(True)
        leg_ia = ax_i_after.legend(loc='upper right', frameon=True, edgecolor='black',
                                   facecolor='white', fontsize=12, fancybox=True)
        leg_ia.get_frame().set_facecolor('white')
        leg_ia.get_frame().set_edgecolor('black')
        leg_ia.get_frame().set_alpha(1.0)

        # Canal Q antes
        ax_q_before = fig_match_spec.add_subplot(gs_match_spec[2, 0])
        ax_q_before.plot(freqs, DQ_db, color='darkgreen', label=label_Q_before)
        ax_q_before.set_title(title_Q_before)
        ax_q_before.set_xlabel("Frequência (Hz)")
        ax_q_before.set_ylabel("Magnitude (dB)")
        ax_q_before.set_xlim(-fc, fc)
        ax_q_before.set_ylim(-90, 5)
        ax_q_before.grid(True)
        leg_qb = ax_q_before.legend(loc='upper right', frameon=True, edgecolor='black',
                                    facecolor='white', fontsize=12, fancybox=True)
        leg_qb.get_frame().set_facecolor('white')
        leg_qb.get_frame().set_edgecolor('black')
        leg_qb.get_frame().set_alpha(1.0)

        # Canal Q após
        ax_q_after = fig_match_spec.add_subplot(gs_match_spec[2, 1])
        ax_q_after.plot(freqs, DQM_db, color='green', label=label_Q_after)
        ax_q_after.set_title(title_Q_after)
        ax_q_after.set_xlabel("Frequência (Hz)")
        ax_q_after.set_ylabel("Magnitude (dB)")
        ax_q_after.set_xlim(-fc, fc)
        ax_q_after.set_ylim(-90, 5)
        ax_q_after.grid(True)
        leg_qa = ax_q_after.legend(loc='upper right', frameon=True, edgecolor='black',
                                   facecolor='white', fontsize=12, fancybox=True)
        leg_qa.get_frame().set_facecolor('white')
        leg_qa.get_frame().set_edgecolor('black')
        leg_qa.get_frame().set_alpha(1.0)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.4)

        self._save_or_show(fig_match_spec, save_path)

    def plot_sampled_signals(self, t_matched, d_I_matched, d_Q_matched,
                             t_samples, I_samples, Q_samples,
                             label_I, label_I_samples,
                             label_Q, label_Q_samples,
                             title_I, title_Q,
                             t_xlim=0.1, save_path=None):
        """
        Plota os sinais I e Q após o filtro casado com os pontos de amostragem.
    
        Args:
            t_matched (np.ndarray): Vetor de tempo dos sinais filtrados.
            d_I_matched (np.ndarray): Sinal I após o filtro casado.
            d_Q_matched (np.ndarray): Sinal Q após o filtro casado.
            t_samples (np.ndarray): Instantes de amostragem.
            I_samples (np.ndarray): Amostras do canal I.
            Q_samples (np.ndarray): Amostras do canal Q.
            label_I (str): Rótulo do sinal I filtrado.
            label_I_samples (str): Rótulo das amostras I.
            label_Q (str): Rótulo do sinal Q filtrado.
            label_Q_samples (str): Rótulo das amostras Q.
            title_I (str): Título do gráfico do canal I.
            title_Q (str): Título do gráfico do canal Q.
            t_xlim (float, optional): Limite do eixo X para ambos os gráficos.
            save_path (str, optional): Caminho para salvar o gráfico.
        """
        fig_sample = plt.figure(figsize=(16, 8))
        gs_sample = gridspec.GridSpec(2, 1)
    
        # Canal I
        ax_si = fig_sample.add_subplot(gs_sample[0])
        ax_si.plot(t_matched, d_I_matched, color='blue', label=label_I)
        ax_si.stem(t_samples, I_samples, linefmt='k-', markerfmt='ko', basefmt=" ", label=label_I_samples)
        ax_si.set_title(title_I)
        ax_si.set_xlabel("Tempo (s)")
        ax_si.set_ylabel("Amplitude")
        ax_si.set_xlim(0, t_xlim)
        ax_si.grid(True)
        leg_si = ax_si.legend(loc='upper right', frameon=True, edgecolor='black',
                              facecolor='white', fontsize=12, fancybox=True)
        leg_si.get_frame().set_facecolor('white')
        leg_si.get_frame().set_edgecolor('black')
        leg_si.get_frame().set_alpha(1.0)
    
        # Canal Q
        ax_sq = fig_sample.add_subplot(gs_sample[1])
        ax_sq.plot(t_matched, d_Q_matched, color='green', label=label_Q)
        ax_sq.stem(t_samples, Q_samples, linefmt='k-', markerfmt='ko', basefmt=" ", label=label_Q_samples)
        ax_sq.set_title(title_Q)
        ax_sq.set_xlabel("Tempo (s)")
        ax_sq.set_ylabel("Amplitude")
        ax_sq.set_xlim(0, t_xlim)
        ax_sq.grid(True)
        leg_sq = ax_sq.legend(loc='upper right', frameon=True, edgecolor='black',
                              facecolor='white', fontsize=12, fancybox=True)
        leg_sq.get_frame().set_facecolor('white')
        leg_sq.get_frame().set_edgecolor('black')
        leg_sq.get_frame().set_alpha(1.0)
    
        plt.tight_layout()
        self._save_or_show(fig_sample, save_path)

    def _save_or_show(self, fig, path):
        if path:
            base_dir = os.path.dirname(os.path.abspath(__main__.__file__))
            full_path = os.path.join(base_dir, path)

            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            fig.savefig(full_path)
        else:
            plt.show()