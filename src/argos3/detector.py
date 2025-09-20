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
                 freq_window: tuple[float, float] = (1000, 9000), bandwidth: float = 1500):
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
            ![pageplot](assets/example_detector_freq.svg)

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
        """
        Divide o sinal inteiro em segmentos de N amostras.
        O último segmento pode ser menor que N.
        """
        segments = []
        total_samples = len(signal)

        for start in range(0, total_samples, self.N):
            end = min(start + self.N, total_samples)
            segments.append(signal[start:end])

        return segments


    def analyze_signal(self, signal: np.ndarray):
        """
        Segmenta e calcula a matriz de potência FFT do sinal inteiro.
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
        """
        Detecta possíveis portadoras no sinal.
        Cria self.detected_matrix com os valores:
            0 = sem frequência detectada
            1 = frequência detectada
            2 = frequência confirmada (detectada também no segmento anterior)
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
        if self.detected_matrix is None:
            raise ValueError("É necessário rodar detect() antes de decision().")

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



if __name__ == "__main__":

    fs = 128_000
    Rb = 400
    
    fc1 = 3000
    
    datagram = Datagram(pcdnum=1234, numblocks=1, seed=11)
    transmitter1 = Transmitter(fc=fc1, fs=fs, Rb=Rb, output_print=False, output_plot=False, carrier_length=0.08)
    t1, s1 = transmitter1.transmit(datagram)
    st = s1
    
    ebn0 = 15
    add_noise = NoiseEBN0(ebn0_db=ebn0, seed=11,length_multiplier=2, position_factor=0.5)
    st = add_noise.add_noise(st)
    
    # Detecção de portadora
    threshold = -12
    detector = CarrierDetector(fs=transmitter1.fs, seg_ms=20, threshold=threshold) 

    detector.detect(st.copy())

    fig, grid = create_figure(1, 1)

    # Heatmap da potência
    PowerMatrixPlot(fig, grid, 0,
                detector.power_matrix,
                fs=detector.fs, N=detector.N,
                title="Matriz de Potência").plot()
    
    save_figure(fig, "example_detector_power_matrix.pdf")

    fig, grid = create_figure(1, 1)
    # Heatmap da detecção
    MatrixSquarePlot(fig, grid, 0,
                 detector.detected_matrix,
                 fs=detector.fs, N=detector.N,
                 title="Matriz de Detecção").plot()

    save_figure(fig, "example_detector_detection_matrix.pdf")


    fig, grid = create_figure(1, 1)

    # Heatmap da decisão
    MatrixSquarePlot(fig, grid, 0,
                 detector.decision_matrix,
                 fs=detector.fs, N=detector.N,
                 title="Matriz de Decisão").plot()
    
    save_figure(fig, "example_detector_decision_matrix.pdf")


    # plota o espectro do sinal no segmento 5
    seg_index = 7
    fig, grid = create_figure(1, 1)
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
    
    save_figure(fig, "example_detector_freq.pdf")