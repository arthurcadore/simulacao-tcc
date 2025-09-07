"""
Implementação de um detector de portadora para recepção PTT-A3.

Autor: Arthur Cadore
Data: 07-09-2025
"""

import numpy as np
from .plotter import create_figure, save_figure, DetectionFrequencyPlot
from .datagram import Datagram
from .transmitter import Transmitter
from .noise import NoiseEBN0
from .receiver import Receiver


class CarrierDetector:
    def __init__(self, fs: float, seg_ms: float = 10.0, segments: int = 2,
                 threshold: float = -10,
                 freq_window: tuple[float, float] = (1000, 9000)):
        """
        Inicializa um detector de portadora, utilizado para detectar possíveis portadoras no sinal recebido.

        Args:
            fs (float): Frequência de amostragem [Hz]
            seg_ms (float): Duração de cada segmento [ms]
            segments (int): Número de segmentos a analisar
            threshold (float): Limiar de potência para detecção
            freq_window (tuple[float, float]): Intervalo de frequências (`f_min`, `f_max`).Frequências fora deste intervalo serão descartadas.
        
        Raises:
            ValueError: Se a frequência de amostragem for menor ou igual a zero.
            ValueError: Se o comprimento de cada segmento for menor ou igual a zero.
            ValueError: Se o número de segmentos for menor que 1.

        Exemplo: 
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
        if segments < 1:
            raise ValueError("Deve haver pelo menos 1 segmento.")

        self.fs = fs
        self.ts = 1 / fs
        self.seg_ms = seg_ms / 1000.0
        self.seg_samples = int(fs * self.seg_ms)
        self.segments = segments
        self.threshold = threshold
        self.freq_window = freq_window
        self.delta_f = fs / self.seg_samples
        self.span = self.delta_f / 2

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
        total_samples = self.seg_samples * self.segments

        if len(signal) < total_samples:
            raise ValueError(
                f"Sinal insuficiente: esperado {total_samples} amostras, mas recebido {len(signal)}."
            )

        signal = signal[:total_samples]
        segments = []
        for i in range(self.segments):
            start = i * self.seg_samples
            end = start + self.seg_samples
            segments.append(signal[start:end])
        return segments

    def analyze_segment(self, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

        Em seguida, calcula a potência espectral $P_n[k]$ em $dB$, conforme a expressão abaixo.

        $$
            P_n[k] = \frac{|X_n[k]|^2}{N}
        $$

        Sendo: 
            - $P_n[k]$ : Potência espectral do segmento $n$.
            - $X_n[k]$ : Transformada de Fourier do segmento $n$.
            - $N$ : Número de amostras do segmento.
            '
        Args:
            segment (np.ndarray): segmento de tempo

        Returns:
            tuple[np.ndarray, np.ndarray]: tupla com as frequências e a potência espectral em $dB$
        """
        N = len(segment)
        X = np.fft.rfft(segment, n=N)
        P_bin = (np.abs(X) ** 2) / (N + 1e-20)
        P_db = 10.0 * np.log10(P_bin + 1e-20)

        freqs = np.fft.rfftfreq(N, d=self.ts)
        return freqs, P_db

    def detect(self, s: np.ndarray) -> list[tuple[np.ndarray, list[float]]]:
        r"""
        Detecta possíveis portadoras no sinal, comparando $P_n[k]$ com o limiar $P_t$.  

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
            list[tuple[np.ndarray, list[float]]]: lista de tuplas com os segmentos e as frequências detectadas
        """
        segments = self.segment_signal(s)
        results = []

        for seg in segments:
            freqs, P_db = self.analyze_segment(seg)

            mask = P_db > self.threshold

            if self.freq_window is not None:
                fmin, fmax = self.freq_window
                mask &= (freqs >= fmin) & (freqs <= fmax)

            freqs_detected = freqs[mask]
            results.append((seg, freqs_detected.tolist()))

        return results

    def check_frequencies(self, results: list[tuple[np.ndarray, list[float]]], tolerance_hz=50): 
        """
        Retorna apenas frequências que foram detectadas em dois segmentos consecutivos.

        Args:
            results (list[tuple[np.ndarray, list[float]]]): saída de self.detect()
            tolerance_hz (float): diferença máxima para considerar que duas frequências são iguais

        Returns:
            list[float]: lista de frequências confirmadas como portadora
        """
        if len(results) < 2:
            return []

        confirmed_freqs = []
        prev_freqs = results[0][1]

        # percorre segmentos a partir do segundo
        for seg, freqs_detected in results[1:]:
            for f in freqs_detected:
                if any(abs(f - pf) <= tolerance_hz for pf in prev_freqs):
                    confirmed_freqs.append(f)
            prev_freqs = freqs_detected

        confirmed_freqs = list(sorted(set(confirmed_freqs)))
        return confirmed_freqs

if __name__ == "__main__":

    fc = np.random.randint(2, 14)*500
    
    print("Frequência Portadora: ", fc)
    
    datagram = Datagram(pcdnum=1234, numblocks=1)
    bitsTX = datagram.streambits  
    transmitter = Transmitter(datagram, fc=fc, output_print=False, output_plot=True)
    t, s = transmitter.run()
    
    ebn0_db = 20
    add_noise = NoiseEBN0(ebn0_db=ebn0_db)
    s_noisy = add_noise.add_noise(s)
    
    threshold = -8.5
    detector = CarrierDetector(fs=transmitter.fs, seg_ms=20, segments=2, threshold=threshold)
    
    results = detector.detect(s_noisy.copy())

    for idx, (seg, freqs) in enumerate(results, start=1):
        print(f"Segmento {idx}: {len(freqs)} frequências -> {freqs}")

    fig, grid = create_figure(2,1)
    DetectionFrequencyPlot(fig, grid, 0, 
              fs=transmitter.fs, 
              signal=results[0][0], 
              threshold=threshold, 
              xlim=(1, 9),
              title="Detecção de portadora de $s(t)$ - Segmento 1",
              labels=["$S(f)$"],
              colors="darkred",
              freqs_detected=results[0][1]
    ).plot()

    DetectionFrequencyPlot(fig, grid, 1, 
              fs=transmitter.fs, 
              signal=results[1][0], 
              threshold=threshold, 
              xlim=(1, 9),
              title="Detecção de portadora de $s(t)$ - Segmento 2",
              labels=["$S(f)$"],
              colors="darkred",
              freqs_detected=results[1][1]
    ).plot()

    fig.tight_layout()
    save_figure(fig, "example_detector_freq.pdf")

    confirmed_freqs = detector.check_frequencies(results)
    print("\nFrequências confirmadas:", confirmed_freqs)

    # Para cada frequência confirmada, executa a recepção
    for idx, freq in enumerate(confirmed_freqs, start=1):
        receiver = Receiver(fc=freq, fs=transmitter.fs, Rb=transmitter.Rb, output_print=False, output_plot=True)
        bitsRX = receiver.run(s_noisy.copy(), t.copy())
            
        try:
            datagramRX = Datagram(streambits=bitsRX)
            print("\n",datagramRX.parse_datagram())

        except Exception as e:
            print("Bits TX: ", ''.join(str(b) for b in bitsTX))
            print("Bits RX: ", ''.join(str(b) for b in bitsRX))
                
            num_errors = sum(1 for tx, rx in zip(bitsTX, bitsRX) if tx != rx)
            ber = num_errors / len(bitsTX)
                
            print(f"Número de erros: {num_errors}")