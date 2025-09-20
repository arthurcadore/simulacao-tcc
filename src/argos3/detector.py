# """
# Implementação de um detector de portadora para recepção PTT-A3.

# Autor: Arthur Cadore
# Data: 07-09-2025
# """

import numpy as np
from .plotter import create_figure, save_figure, PowerMatrixPlot, MatrixSquarePlot, DetectionFrequencyPlot
from .datagram import Datagram
from .transmitter import Transmitter
from .noise import NoiseEBN0, Noise
from .receiver import Receiver


class CarrierDetector:
    def __init__(self, fs: float = 128_000, seg_ms: float = 10.0,
                 threshold: float = -10,
                 freq_window: tuple[float, float] = (1000, 9000), bandwidth: float = 1600):
        """
        Inicializa um detector de portadora, utilizado para detectar possíveis portadoras no sinal recebido.

        Args:
            fs (float): Frequência de amostragem [Hz]
            seg_ms (float): Duração de cada segmento [ms]
            threshold (float): Limiar de potência para detecção
            freq_window (tuple[float, float]): Intervalo de frequências (`f_min`, `f_max`).Frequências fora deste intervalo serão descartadas.
        
        Raises:
            ValueError: Se a frequência de amostragem for menor ou igual a zero.
            ValueError: Se o comprimento de cada segmento for menor ou igual a zero.

        Example: 
            - Segmentos de tempo: ![pageplot](assets/example_detector_freq.svg)
            - Waterfall: ![pageplot](assets/example_detector_decision_matrix.svg)


        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-2097-CNES (Seção 3.3)
        </div>
        """
        if fs <= 0:
            raise ValueError("A frequência de amostragem deve ser maior que zero.")
        if seg_ms <= 0:
            raise ValueError("O comprimento de cada segmento deve ser maior que zero.")

        self.fs = fs
        self.ts = 1 / self.fs
        self.seg_s = seg_ms / 1000.0
        self.N = int(self.fs * self.seg_s)
        self.threshold = threshold
        self.freq_window = freq_window
        self.bandwidth = bandwidth

        # Valores fixos de espectro
        self.delta_f = self.fs / self.N
        self.span = self.delta_f / 2

        # Quantos bins de FFT correspondem à largura de banda
        self.bandwidth_bins = int(self.bandwidth / self.delta_f)


        self.power_matrix = None
        self.detected_matrix = None
        self.decision_matrix = None
        
    def segment_signal(self, signal: np.ndarray) -> list[np.ndarray]:
        r"""
        Divide o sinal recebido em segmentos de tempo $x_n[m]$, cada segmento com `seg_ms` de duração, conforme a expressão abaixo. 

        $$
        x_n[m] = s(t_{n} + mT_s)
        $$

        Sendo: 
            - $x_n[m]$ : Segmento de tempo $n$.
            - $s(t)$ : Sinal recebido.
            - $T_s$ : Período de amostragem.
            - $m$ : Número do segmento.
            - $t_n$ : Instante de início do segmento $n$.

        Args:
            signal (np.ndarray): sinal recebido

        Returns:
            list[np.ndarray]: lista de segmentos de tempo
        """
        segments = []
        total_samples = len(signal)

        for start in range(0, total_samples, self.N):
            end = min(start + self.N, total_samples)
            segments.append(signal[start:end])

        return segments


    def analyze_signal(self, signal: np.ndarray):
        r"""
        Calcula a FFT de cada segmento $x_n[m]$, usando a expressão abaixo. 
        
        $$
            X_n[k] = \sum_{m=0}^{N-1} x_n[m]\, e^{-j2\pi km/N} 
        $$

        Sendo: 
            - $X_n[k]$ : Transformada de Fourier do segmento $n$.
            - $x_n[m]$ : Segmento de tempo $n$.
            - $N$ : Número de amostras do segmento.
            - $k$ : Número da transformada de Fourier.
            - $m$ : Número da amostra.
            - $T_s$ : Período de amostragem.
            - $e^{-j2\pi km/N}$ : Exponencial complexa.

        Em seguida, calcula a potência espectral $P_n[k]$ em $dB$, e divide pelo número de amostras $N$ contidas no segmento para normalização.

        $$
            P_n[k] = \frac{|X_n[k]|^2}{N}
        $$

        Sendo: 
            - $P_n[k]$ : Potência espectral do segmento $n$, normalizada em $dB$.
            - $X_n[k]$ : Transformada de Fourier do segmento $n$.
            - $N$ : Número de amostras do segmento.

        Args:
            segment (np.ndarray): segmento de tempo

        Returns:
            freqs (tuple[np.ndarray,np.ndarray]): tupla com as frequências e a potência espectral em $dB$

        Example: 
            ![pageplot](assets/example_detector_power_matrix.svg)
        """

        segments = self.segment_signal(signal)
        n_segments = len(segments)
        n_freqs = self.N // 2 + 1

        self.power_matrix = np.zeros((n_segments, n_freqs))

        for i, seg in enumerate(segments):
            X = np.fft.rfft(seg, n=self.N)
            P_bin = (np.abs(X) ** 2) / len(seg)  # normaliza pelo tamanho real do segmento
            P_db = 10.0 * np.log10(P_bin + 1e-12)  # evita log(0)
            self.power_matrix[i, :] = P_db


    def detect(self, s: np.ndarray):
        r"""
        Detecta possíveis portadoras no sinal, comparando $P_n[k]$ com o limiar $P_t$, para cada índice $k$ da FFT.

        $$
            f_n[k] =
            \begin{cases}
            \dfrac{k}{N} \cdot f_s, & \text{se } P_n[k] > P_t\\
            \text{não detectada}, & \text{se } P_n[k] \leq P_t
            \end{cases}
        $$

        Sendo: 
            - $f_n[k]$ : frequência detectada no segmento $n$.
            - $P_n[k]$ : potência espectral do segmento $n$.
            - $P_t$ : limiar de potência.
            - $N$ : número de amostras do segmento.
            - $f_s$ : frequência de amostragem.
            - $k$ : índice da FFT.
            - `não detectada`: Frequência ignorada no processo de detecção.  

        Args:
            s (np.ndarray): sinal recebido

        Returns:
            results (list[tuple[np.ndarray, list[float]]]): lista de tuplas com os segmentos e as frequências detectadas

        Example: 
            ![pageplot](assets/example_detector_detection_matrix.svg)
        """
        # Calcula matriz de potência FFT
        self.analyze_signal(s)

        n_segments, n_freqs = self.power_matrix.shape
        self.detected_matrix = np.zeros((n_segments, n_freqs), dtype=int)

        # Frequências reais dos bins da FFT
        freqs = np.fft.rfftfreq(self.N, d=self.ts)

        for i in range(n_segments):
            P_db = self.power_matrix[i, :]

            # Máscara de detecção pelo limiar
            mask = P_db > self.threshold

            # Restringe à janela de frequências
            if self.freq_window is not None:
                fmin, fmax = self.freq_window
                mask &= (freqs >= fmin) & (freqs <= fmax)

            detected_bins = np.where(mask)[0]

            for k in detected_bins:
                if i > 0 and self.detected_matrix[i-1, k] in (1, 2):
                    # Confirmada (foi detectada no segmento anterior como 1 ou 2)
                    self.detected_matrix[i, k] = 2
                else:
                    # Detectada mas não confirmada
                    self.detected_matrix[i, k] = 1

        self.decision()

    def decision(self):
        """
        Retorna apenas frequências que foram detectadas em dois segmentos consecutivos. A tolerância é dada pela resolução espectral da FFT, $\Delta f$, conforme a expressão abaixo. 

        $$
            \Delta f = \dfrac{f_s}{N}
        $$

        Sendo: 
            - $\Delta f$ : resolução espectral da FFT.
            - $f_s$ : frequência de amostragem.
            - $N$ : número de amostras do segmento.

        Args:
            results (list[tuple[np.ndarray, list[float]]]): saída de self.detect()
            confirmed_freqs (list[float]): lista de frequências confirmadas como portadora

        Returns:
            confirmed_freqs (list[float]): lista de frequências confirmadas como portadora

        Example: 
            ![pageplot](assets/example_detector_decision_matrix.svg)
        """

        self.decision_matrix = np.copy(self.detected_matrix)
        n_segments, n_freqs = self.detected_matrix.shape

        # matriz auxiliar para controlar spans existentes
        span_matrix = np.zeros_like(self.detected_matrix, dtype=bool)

        runs = []

        half_span = (self.bandwidth_bins - 1) // 2  # calcula metade do span para aplicar acima e abaixo do centro

        for i in range(n_segments):
            for k in range(n_freqs):
                # só processa centros detectados (2) que não estão dentro de um span existente
                if self.detected_matrix[i, k] != 2 or span_matrix[i, k]:
                    continue

                center_k = k
                s = i + 1  # começa no próximo segmento
                zero_count = 0
                start_s = s

                # aplica o 4 e o span no primeiro segmento após o 2
                lower = max(center_k - half_span, 0)
                upper = min(center_k + half_span, n_freqs - 1)
                self.decision_matrix[s, lower:upper + 1] = np.where(
                    np.arange(lower, upper + 1) == center_k,
                    4,  # centro
                    3   # span
                )
                span_matrix[s, lower:upper + 1] = True

                s += 1  # avança para continuar o loop de extensão

                # agora continua preenchendo a sequência enquanto houver atividade
                while s < n_segments and zero_count < 2:
                    neighbors = [center_k]
                    if center_k > 0:
                        neighbors.append(center_k - 1)
                    if center_k < n_freqs - 1:
                        neighbors.append(center_k + 1)

                    found_activity = False
                    for look_ahead in range(0, 3):
                        idx = s + look_ahead
                        if idx >= n_segments:
                            break
                        if any(self.detected_matrix[idx, nb] in (1, 2) for nb in neighbors):
                            found_activity = True
                            break

                    # aplica o span no segmento atual
                    self.decision_matrix[s, lower:upper + 1] = np.where(
                        np.arange(lower, upper + 1) == center_k,
                        4,
                        3
                    )
                    span_matrix[s, lower:upper + 1] = True

                    if found_activity:
                        zero_count = 0
                    else:
                        zero_count += 1

                    s += 1

                runs.append((start_s, s - 1, center_k))

    def return_channels(self):
        """
        Varre a decision_matrix e retorna as frequências confirmadas.
        com o início e fim do segmento onde a portadora foi demodulada.

        Returns:
            channels (list[tuple[float, int, int]]): Lista de tuplas (freq_Hz, start_segment, end_segment)
        """
        if not hasattr(self, 'decision_matrix'):
            raise ValueError("A decision_matrix ainda não foi criada. Execute self.decision() antes.")

        n_segments, n_freqs = self.decision_matrix.shape
        visited = np.zeros_like(self.decision_matrix, dtype=bool)
        channels = []

        # Frequências reais dos bins da FFT
        freqs = np.fft.rfftfreq(self.N, d=self.ts)

        for i in range(n_segments):
            for k in range(n_freqs):
                # só processa centros não visitados
                if self.decision_matrix[i, k] != 4 or visited[i, k]:
                    continue

                start_segment = i
                s = i
                # percorre os segmentos enquanto houver 4 no centro
                while s < n_segments and self.decision_matrix[s, k] == 4:
                    visited[s, k] = True
                    s += 1
                end_segment = s - 1

                channels.append((freqs[k], start_segment, end_segment))

        print("Frequências confirmadas (Hz) com início e fim de segmentos:")
        for f, start, end in channels:
            print(f"Frequência {f:.1f} Hz: segmento {start} -> {end}")

        return channels

if __name__ == "__main__":

    fs = 128_000
    Rb = 400
    
    fc1 = 2000
    fc2 = 4500
    fc3 = 7000
    
    datagram = Datagram(pcdnum=1234, numblocks=1, seed=11)
    transmitter1 = Transmitter(fc=fc1, fs=fs, Rb=Rb, output_print=False, output_plot=False, carrier_length=0.08)
    transmitter2 = Transmitter(fc=fc2, fs=fs, Rb=Rb, output_print=False, output_plot=False, carrier_length=0.08)
    transmitter3 = Transmitter(fc=fc3, fs=fs, Rb=Rb, output_print=False, output_plot=False, carrier_length=0.08)

    t1, s1 = transmitter1.transmit(datagram)
    t2, s2 = transmitter2.transmit(datagram)
    t3, s3 = transmitter3.transmit(datagram)

    ebn0_db = 20
    noise1 = NoiseEBN0(ebn0_db=ebn0_db, seed=11,length_multiplier=2, position_factor=np.random.uniform(0, 1))
    sn1 = noise1.add_noise(s1)
    noise2 = NoiseEBN0(ebn0_db=ebn0_db, seed=12,length_multiplier=2, position_factor=np.random.uniform(0, 1))
    sn2 = noise2.add_noise(s2)
    noise3 = NoiseEBN0(ebn0_db=ebn0_db, seed=13,length_multiplier=2, position_factor=np.random.uniform(0, 1))
    sn3 = noise3.add_noise(s3)
    
    st = sn1 + sn2 + sn3

    # Detecção de portadora
    threshold = -12
    detector = CarrierDetector(fs=transmitter1.fs, seg_ms=10, threshold=threshold) 

    detector.detect(st.copy())

    # Heatmap da potência
    fig, grid = create_figure(1, 1, figsize=(16, 9))
    PowerMatrixPlot(fig, grid, 0,
                detector.power_matrix,
                fs=detector.fs, N=detector.N).plot()
    save_figure(fig, "example_detector_power_matrix.pdf")

    # Heatmap da detecção
    fig, grid = create_figure(1, 1)
    MatrixSquarePlot(fig, grid, 0,
                 detector.detected_matrix,
                 fs=detector.fs, N=detector.N).plot()

    save_figure(fig, "example_detector_detection_matrix.pdf")

    # Heatmap da decisão
    fig, grid = create_figure(1, 1)
    MatrixSquarePlot(fig, grid, 0,
                 detector.decision_matrix,
                 fs=detector.fs, N=detector.N).plot()
    save_figure(fig, "example_detector_decision_matrix.pdf")

    # plota o espectro do sinal no segmento desejado
    seg_index = 10
    fig, grid = create_figure(2, 1)
    DetectionFrequencyPlot(fig, grid, 0, 
              fs=transmitter1.fs, 
              signal=detector.power_matrix[seg_index, :], 
              threshold=detector.threshold, 
              xlim=(1, 9),
              title="Detecção de portadora de $s(t)$ - Segmento %d" % seg_index,
              labels=["$S(f)$"],
              colors="darkred",
              freqs_detected=detector.detected_matrix[seg_index, :]
    ).plot()
    DetectionFrequencyPlot(fig, grid, 1, 
              fs=transmitter1.fs, 
              signal=detector.power_matrix[seg_index+1, :], 
              threshold=detector.threshold, 
              xlim=(1, 9),
              title="Detecção de portadora de $s(t)$ - Segmento %d" % (seg_index+1),
              labels=["$S(f)$"],
              colors="darkred",
              freqs_detected=detector.detected_matrix[seg_index+1, :]
    ).plot()
    save_figure(fig, "example_detector_freq.pdf")

    # Recepção: 
    channels = detector.return_channels()
    segments = detector.segment_signal(st)  # lista de segmentos do sinal original
    
    for idx, (freq, start_seg, end_seg) in enumerate(channels, start=1):
        print(f"\n ==============================================")
        print(f"\n ==== RECEPÇÃO DE s(t) COM f_c = {freq:.1f} Hz ==== \n")
    
        # Seleciona apenas os segmentos necessários e concatena
        selected_signal = np.concatenate(segments[start_seg-1:end_seg + 1])
    
        # Instancia o receptor
        receiver = Receiver(fc=freq, fs=detector.fs, Rb=Rb, output_print=True, output_plot=True)
        datagramRX, success = receiver.receive(selected_signal)
    
        if not success:
            bitsTX = datagram.streambits 
            bitsRX = datagramRX
            print("Bits TX: ", ''.join(str(b) for b in bitsTX))
            print("Bits RX: ", ''.join(str(b) for b in bitsRX))
    
            # Calcula a Taxa de Erro de Bit (BER)
            num_errors = sum(1 for tx, rx in zip(bitsTX, bitsRX) if tx != rx)
            ber = num_errors / len(bitsTX)
    
            print(f"Número de erros: {num_errors}")
            print(f"Taxa de Erro de Bit (BER): {ber:.6f}")
