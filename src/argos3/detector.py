# """
# Implementation of carrier detector for PTT-A3 reception.
#
# Author: Arthur Cadore
# Date: 07-09-2025
# """

import numpy as np
from .plotter import create_figure, save_figure, WaterfallPlot, WaterfallDecisionPlot, DetectionFrequencyPlot, Waterfall3DPlot
from .datagram import Datagram
from .transmitter import Transmitter
from .receiver import Receiver
from .channel import Channel
from .env_vars import *


class CarrierDetector:
    def __init__(self, fs: float = 128_000, seg_ms: float = 10.0, threshold: float = -10, freq_window: tuple[float, float] = (0000, 10000), bandwidth: float = 1600, history: int = 4):
        """
        Initializes a carrier detector, used to detect possible carriers in the received signal.

        ![pageplot](../assets/detector.svg)

        Args:
            fs (float): Sampling frequency [Hz]
            seg_ms (float): Duration of each segment [ms]
            threshold (float): Power threshold for detection
            freq_window (tuple[float, float]): Frequency interval (`f_min`, `f_max`). Frequencies outside this interval will be discarded.
            bandwidth (float): Bandwidth of span [Hz], used to ignore other frequencies that can be confirmed.
            history (int): Number of previous segments to consider for carrier detection. 
        
        Raises:
            ValueError: If the sampling frequency is less than or equal to zero.
            ValueError: If the segment duration is less than or equal to zero.

        Examples:
            >>> import argos3
            >>> import numpy as np 
            >>> 
            >>> fc = np.random.randint(10,80)*100
            >>> print(fc)
            2400
            >>>
            >>> transmitter = argos3.Transmitter(fc=fc, output_print=False, output_plot=False)
            >>> t, s = transmitter.transmit(argos3.Datagram(pcdnum=1234, numblocks=1))
            >>> 
            >>> channel = argos3.Channel(duration=1, noise_mode="ebn0", noise_db=20)
            >>> channel.add_signal(s, position_factor=0.5)
            >>> channel.add_noise()
            >>> st = channel.channel
            >>> 
            >>> detector = argos3.CarrierDetector(seg_ms=10, threshold=-15)
            >>> detector.detect(st.copy())
            >>> detections = detector.return_channels()
            >>>
            >>> print(detections)
            [(np.float64(2400.0), 41, 65)]
            >>>                   
            >>> first_sample = int((detections[0][1] - 5) * detector.fs * detector.seg_s)
            >>> last_sample = int(detections[0][2] * detector.fs * detector.seg_s)
            >>> st_prime = st[first_sample:last_sample]
            >>> 
            >>> receiver = argos3.Receiver(fc=detections[0][0], output_print=False, output_plot=False)
            >>> datagramRX, success = receiver.receive(st_prime)
            >>> 
            >>> print(success)
            True
            >>> print(datagramRX.parse_datagram())
            {
              "msglength": 1,
              "pcdid": 1234,
              "data": {
                "bloco_1": {
                  "sensor_1": 37,
                  "sensor_2": 198,
                  "sensor_3": 9
                }
              },
              "tail": 7
            }
            
            - Waterfall Diagram: ![pageplot](assets/example_detector_waterfall.svg)
            - Frequency Domain Segment: ![pageplot](assets/example_detector_freq.svg)
            - Waterfall Detection Diagram: ![pageplot](assets/example_detector_waterfall_detection.svg)
            - Waterfall Decision Diagram: ![pageplot](assets/example_detector_waterfall_decision.svg)


        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-2097-CNES (Section 3.3)
        </div>
        """
        if fs <= 0:
            raise ValueError("The sampling frequency must be greater than zero.")
        if seg_ms <= 0:
            raise ValueError("The segment duration must be greater than zero.")

        # Attributes
        self.fs = fs
        self.ts = 1 / self.fs
        self.seg_s = seg_ms / 1000.0
        self.N = int(self.fs * self.seg_s)
        self.threshold = threshold
        self.freq_window = freq_window
        self.bandwidth = bandwidth
        self.history = history

        # Fixed values of spectrum
        self.delta_f = self.fs / self.N
        self.span = self.delta_f / 2

        # Number of FFT bins corresponding to the bandwidth
        self.bandwidth_bins = int(self.bandwidth / self.delta_f)

        # Matrices
        self.power_matrix = None
        self.detected_matrix = None
        self.decision_matrix = None
        
    def segment_signal(self, signal: np.ndarray) -> list[np.ndarray]:
        r"""
        Divides the received signal into segments of time $x_n[m]$, each segment with `seg_ms` duration, according to the expression below. 

        $$
        x_n[m] = s(t_{n} + mT_s)
        $$

        Where: 
            - $x_n[m]$ : Segment of time $n$.
            - $s(t)$ : Received signal.
            - $T_s$ : Sampling period.
            - $m$ : Segment number.
            - $t_n$ : Start time of segment $n$.

        Args:
            signal (np.ndarray): Received signal

        Returns:
            list[np.ndarray]: List of time segments
        """

        segments = []
        total_samples = len(signal)

        # Segment signal
        for start in range(0, total_samples, self.N):
            end = min(start + self.N, total_samples)
            segments.append(signal[start:end])

        return segments


    def analyze_signal(self, signal: np.ndarray):
        r"""
        Calculates the FFT of each segment $x_n[m]$, using the expression below. 
        
        $$
            X_n[k] = \sum_{m=0}^{N-1} x_n[m]\, e^{-j2\pi km/N} 
        $$

        Where: 
            - $X_n[k]$ : FFT of segment $n$.
            - $x_n[m]$ : Segment of time $n$.
            - $N$ : Number of samples in the segment.
            - $k$ : Number of the Fourier transform.
            - $m$ : Sample number.
            - $T_s$ : Sampling period.
            - $e^{-j2\pi km/N}$ : Complex exponential.

        Then, calculates the power spectral density $P_n[k]$ in $dB$, and divides by the number of samples $N$ contained in the segment for normalization.

        $$
            P_n[k] = \frac{|X_n[k]|^2}{N}
        $$

        Where: 
            - $P_n[k]$ : Power spectral density of segment $n$, normalized in $dB$.
            - $X_n[k]$ : FFT of segment $n$.
            - $N$ : Number of samples in the segment.

        Args:
            signal (np.ndarray): Received signal

        Returns:
            freqs (tuple[np.ndarray,np.ndarray]): tuple with frequencies and power spectral density in $dB$

        Examples: 
            - Waterfall Diagram: ![pageplot](assets/example_detector_waterfall.svg)
            - 3D Waterfall Diagram: ![pageplot](assets/example_detector_waterfall_3d.svg)
        """

        # Segments
        segments = self.segment_signal(signal)
        n_segments = len(segments)

        # Number of FFT bins
        n_freqs = self.N // 2 + 1

        # Power matrix (m x n), where m is the number of segments and n is the number of FFT bins
        self.power_matrix = np.zeros((n_segments, n_freqs))

        # Calculate power spectral density for each segment and stores in the power matrix
        for i, seg in enumerate(segments):
            X = np.fft.rfft(seg, n=self.N)
            P_bin = (np.abs(X) ** 2) / len(seg)
            P_db = 10.0 * np.log10(P_bin + 1e-12)  
            self.power_matrix[i, :] = P_db

    def detect(self, signal: np.ndarray):
        r"""
        Detects possible carriers in the signal, comparing $P_n[k]$ with the threshold $P_t$, for each index $k$ of the FFT, as shown below.

        $$
            f_n[k] =
            \begin{cases}
            \dfrac{k}{N} \cdot f_s, & \text{if } P_n[k] > P_t\\
            \text{not detected}, & \text{if } P_n[k] \leq P_t
            \end{cases}
        $$

        Where: 
            - $f_n[k]$ : Detected frequency in segment $n$.
            - $P_n[k]$ : Power spectral density of segment $n$.
            - $P_t$ : Power threshold.
            - $N$ : Number of samples in the segment.
            - $f_s$ : Sampling frequency.
            - $k$ : Index of the FFT.
            - `not detected`: Frequency ignored in the detection process.  

        Args:
            signal (np.ndarray): Received signal

        Returns:
            results (list[tuple[np.ndarray, list[float]]]): List of tuples with the segments and detected frequencies

        Examples: 
            - Waterfall Detection Diagram: ![pageplot](assets/example_detector_waterfall_detection.svg)
        """
        # Calculates the power matrix FFT
        self.analyze_signal(signal)

        n_segments, n_freqs = self.power_matrix.shape
        self.detected_matrix = np.zeros((n_segments, n_freqs), dtype=int)

        # Real frequencies of the FFT bins
        freqs = np.fft.rfftfreq(self.N, d=self.ts)

        for i in range(n_segments):
            P_db = self.power_matrix[i, :]

            # Detection threshold mask
            mask = P_db > self.threshold

            # Restricts to the frequency window
            if self.freq_window is not None:
                fmin, fmax = self.freq_window
                mask &= (freqs >= fmin) & (freqs <= fmax)

            detected_bins = np.where(mask)[0]

            for k in detected_bins:
                if i >= self.history:
                    # Confirms only if the last 'history' are exactly 1
                    past_values = self.detected_matrix[i-self.history:i, k]
                    if np.all(past_values == 1):

                        # frequency confirmed, goes to demodulation
                        self.detected_matrix[i, k] = 2
                    else:
                        # frequency detected, but not confirmed
                        self.detected_matrix[i, k] = 1
                else:
                    # only detected, without history
                    self.detected_matrix[i, k] = 1

        self.decision()

    def decision(self):
        """
        Returns only frequencies that were detected in two consecutive segments. The tolerance is given by the FFT spectral resolution, $\Delta f$, according to the expression below. 

        $$
            \Delta f = \dfrac{f_s}{N}
        $$

        Where: 
            - $\Delta f$ : FFT spectral resolution.
            - $f_s$ : sampling frequency.
            - $N$ : number of samples in the segment.

        Args:
            signal (np.ndarray): Received signal

        Returns:
            confirmed_freqs (list[float]): list of confirmed carrier frequencies

        Examples: 
            - Waterfall Decision Diagram: ![pageplot](assets/example_detector_waterfall_decision.svg)
        """

        self.decision_matrix = np.copy(self.detected_matrix)
        n_segments, n_freqs = self.detected_matrix.shape

        # auxiliary matrix to control existing spans
        span_matrix = np.zeros_like(self.detected_matrix, dtype=bool)

        runs = []

        half_span = (self.bandwidth_bins - 1) // 2  # calculates half span to apply above and below the center

        for i in range(n_segments):
            for k in range(n_freqs):
                # only processes confirmed centers (2) that are not inside an existing span
                if self.detected_matrix[i, k] != 2 or span_matrix[i, k]:
                    continue

                center_k = k
                s = i + 1
                zero_count = 0
                start_s = s

                # applies "4" and span in the first segment after "2"
                lower = max(center_k - half_span, 0)
                upper = min(center_k + half_span, n_freqs - 1)
                self.decision_matrix[s, lower:upper + 1] = np.where(
                    np.arange(lower, upper + 1) == center_k,
                    4,  # center
                    3   # span
                )
                span_matrix[s, lower:upper + 1] = True

                # advances to continue the extension loop
                s += 1

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

                    # applies span in the current segment
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
        Scans the decision_matrix and returns the confirmed frequencies.
        with the start and end segment where the carrier was demodulated.

        Returns:
            channels (list[tuple[float, int, int]]): List of tuples (freq_Hz, start_segment, end_segment)
        """
        if not hasattr(self, 'decision_matrix'):
            raise ValueError("A decision_matrix ainda não foi criada. Execute self.decision() antes.")

        n_segments, n_freqs = self.decision_matrix.shape
        visited = np.zeros_like(self.decision_matrix, dtype=bool)
        channels = []

        # Frequencies of the FFT bins
        freqs = np.fft.rfftfreq(self.N, d=self.ts)

        for i in range(n_segments):
            for k in range(n_freqs):
                # only processes confirmed centers (4) that are not visited
                if self.decision_matrix[i, k] != 4 or visited[i, k]:
                    continue

                start_segment = i
                s = i
                # goes through the segments while there is 4 in the center
                while s < n_segments and self.decision_matrix[s, k] == 4:
                    visited[s, k] = True
                    s += 1
                end_segment = s - 1

                channels.append((freqs[k], start_segment, end_segment))

        return channels

if __name__ == "__main__":

    fs = 128_000
    Rb = 400
    
    fc1 = np.random.randint(10, 30)*100
    fc2 = fc1 + 2500
    fc3 = fc2 + 2500
    
    datagram1 = Datagram(pcdnum=1234, numblocks=1, seed=11)
    datagram2 = Datagram(pcdnum=1234, numblocks=4, seed=11)
    datagram3 = Datagram(pcdnum=1234, numblocks=8, seed=11)

    print("ut1: ", ''.join(str(b) for b in datagram1.streambits))
    print("ut2: ", ''.join(str(b) for b in datagram2.streambits))
    print("ut3: ", ''.join(str(b) for b in datagram3.streambits))

    transmitter1 = Transmitter(fc=fc1, fs=fs, Rb=Rb, output_print=False, output_plot=False, carrier_length=0.08)
    transmitter2 = Transmitter(fc=fc2, fs=fs, Rb=Rb, output_print=False, output_plot=False, carrier_length=0.08)
    transmitter3 = Transmitter(fc=fc3, fs=fs, Rb=Rb, output_print=False, output_plot=False, carrier_length=0.08)
    transmitter4 = Transmitter(fc=fc1, fs=fs, Rb=Rb, output_print=False, output_plot=False, carrier_length=0.08)

    t1, s1 = transmitter1.transmit(datagram1)
    t2, s2 = transmitter2.transmit(datagram2)
    t3, s3 = transmitter3.transmit(datagram3)
    t4, s4 = transmitter4.transmit(datagram1)

    channel = Channel(fs=fs, duration=1, noise_mode="ebn0", noise_db=20, seed=11)

    p1 = np.random.choice(np.arange(0, 0.21, 0.1))
    p2 = np.random.choice(np.arange(0, 1.01, 0.1))  
    p3 = np.random.choice(np.arange(0, 1.01, 0.1))  
    p4 = p1 + 0.6

    s1 = channel.add_signal(s1, position_factor=p1)
    s2 = channel.add_signal(s2, position_factor=p2)
    s3 = channel.add_signal(s3, position_factor=p3)
    s4 = channel.add_signal(s4, position_factor=p4)

    channel.add_noise()
    st = channel.channel

    threshold = -15
    detector = CarrierDetector(fs=fs, seg_ms=10, threshold=threshold) 
    detector.detect(st.copy())
    fig, grid = create_figure(1, 1, figsize=(16, 9))

    WaterfallPlot(fig, grid, 0,
                detector.power_matrix,
                fs=detector.fs, N=detector.N,
                title=WATERFALL_TITLE
    ).plot()
    save_figure(fig, "example_detector_waterfall.pdf")

    fig, grid = create_figure(1, 1, figsize=(12, 12))
    Waterfall3DPlot(fig, grid, 0,
                      detector.power_matrix,
                      fs=detector.fs,
                      N=detector.N,
                      freq_window=detector.freq_window,
                      threshold=detector.threshold,
                      elev=2, azim=-10,
                      title=WATERFALL_TITLE
    ).plot()
    
    save_figure(fig, "example_detector_waterfall_3d.pdf")

    fig, grid = create_figure(1, 1)
    WaterfallDecisionPlot(fig, grid, 0,
                 detector.detected_matrix,
                 fs=detector.fs, 
                 legend_list=["Detected", "Confirmed"],
                 N=detector.N,
                 title=WATERFALL_DETECTION_TITLE
    ).plot()

    save_figure(fig, "example_detector_waterfall_detection.pdf")

    fig, grid = create_figure(1, 1)
    WaterfallDecisionPlot(fig, grid, 0,
                 detector.decision_matrix,
                 fs=detector.fs, 
                 legend_list=["Detected", "Confirmed", "Span", "Demodulation"],
                 N=detector.N,
                 title=WATERFALL_DECISION_TITLE
    ).plot()
    save_figure(fig, "example_detector_waterfall_decision.pdf")

    seg_index = 1
    fig, grid = create_figure(2, 1)
    DetectionFrequencyPlot(fig, grid, 0, 
              fs=fs, 
              signal=detector.power_matrix[seg_index, :], 
              threshold=detector.threshold, 
              xlim=(0, 10),
              title="Detection of $s(t)$ - Segment %d" % seg_index,
              labels=["$S(f)$"],
              colors=COLOR_COMBINED,
              freqs_detected=detector.detected_matrix[seg_index, :]
    ).plot()
    DetectionFrequencyPlot(fig, grid, 1, 
              fs=fs, 
              signal=detector.power_matrix[seg_index+1, :], 
              threshold=detector.threshold, 
              xlim=(0, 10),
              title="Detection of $s(t)$ - Segment %d" % (seg_index+1),
              labels=["$S(f)$"],
              colors=COLOR_COMBINED,
              freqs_detected=detector.detected_matrix[seg_index+1, :]
    ).plot()
    save_figure(fig, "example_detector_freq.pdf")

    channels = detector.return_channels()

    print("Confirmed frequencies (Hz) with start and end segments:")
    for f, start, end in channels:
        print(f"Frequency {f:.1f} Hz: segment {start} -> {end}")
    
    for idx, (freq, start_seg, end_seg) in enumerate(channels, start=1):
        print(f"\n ==============================================")
        print(f"\n ==== RECEPTION OF s(t) WITH f_c = {freq:.1f} Hz ==== \n")
    
        first_segment = int((start_seg - 5) * detector.fs * detector.seg_s)
        last_segment = int(end_seg * detector.fs * detector.seg_s)
        selected_signal = st[first_segment:last_segment]
    
        receiver = Receiver(fc=freq, fs=detector.fs, Rb=Rb, output_print=True, output_plot=True)
        datagramRX, success = receiver.receive(selected_signal)
    
        if not success:
            bitsRX = datagramRX
            print("Decoding failed: ")
            print("Bits RX: ", ''.join(str(b) for b in bitsRX))
