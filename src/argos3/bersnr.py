# """
# Implementation of BER vs Eb/N0 simulation for the ARGOS-3 standard.
#
# Author: Arthur Cadore
# Date: 8-09-2025
# """

import numpy as np
import concurrent.futures
from scipy.special import erfc
from tqdm import tqdm

from .datagram import Datagram
from .transmitter import Transmitter
from .receiver import Receiver
from .noise import NoiseEBN0
from .data import ExportData, ImportData
from .convolutional import EncoderConvolutional, DecoderViterbi
from .plotter import create_figure, save_figure, BersnrPlot

def interpolate(positions, ref_points, ref_values):
    r"""
    Defines the number of repetitions as a function of $Eb/N0$, using linear interpolation between reference points, given by the expression below.

    $$
    r = r_{i} + \frac{(EBN0 - EBN0_{i})}{(EBN0_{i+1} - EBN0_{i})} \cdot (r_{i+1} - r_{i})
    $$

    Where:
        - $r$: Number of repetitions.
        - $EBN0$: $Eb/N_0$ ratio in decibels.
        - $r_i$ and $r_{i+1}$: Number of repetitions at the nearest reference points.
        - $EBN0_i$ and $EBN0_{i+1}$: $Eb/N_0$ ratios at the nearest reference points.

    Args: 
        positions (int): Total number of points to be generated.
        ref_points (array-like): Reference points. 
        ref_values (array-like): Values corresponding to the reference points.
    
    Returns:
        interpolated_values (np.ndarray): Array of interpolated values, rounded to integers.
    """
    # Convert to numpy arrays
    ref_points = np.array(ref_points)
    ref_values = np.array(ref_values)
    
    # Perform linear interpolation using np.interp
    interpolated_values = np.interp(np.linspace(ref_points[0], ref_points[-1], positions), ref_points, ref_values)
    
    # Round values and convert to integers
    interpolated_values = np.round(interpolated_values).astype(int)
    
    return interpolated_values

class BERSNR_ARGOS: 
    def __init__(self, EbN0_values=np.arange(0, 10, 1), num_workers=56, numblocks=8, max_repetitions=2000, error_values=None):
        r"""
        Simulates the BER vs Eb/N0 for the ARGOS-3 standard.

        Args:
            EbN0_values (array-like): Values of Eb/N0 for which the simulation will be performed.
            num_workers (int): Number of threads for parallelization.
            numblocks (int): Number of data blocks for each datagram.
            max_repetitions (int): Maximum number of repetitions for each Eb/N0 value.
            error_values (array-like): Maximum number of errors for each Eb/N0 value.
        
        Raises:
            ValueError: If the number of errors is not the same as the number of Eb/N0 values.

        Examples: 
            - BER vs Eb/N0 Plot Example: ![pageplot](assets/ber_vs_ebn0.svg)
            
        """
        if len(error_values) != len(EbN0_values):
            raise ValueError("error_values must have the same length as EbN0_values")

        # Fixed system parameters
        self.fs = 128_000
        self.Rb = 400
        self.fc = 4000

        # Variable system parameters
        self.EbN0_values = EbN0_values
        self.num_workers = num_workers
        self.numblocks = numblocks
        self.max_repetitions = max_repetitions
        self.error_values = error_values

        # TX Chain
        self.datagramTX = Datagram(pcdnum=1234, numblocks=numblocks, seed=10)
        self.bitsTX = self.datagramTX.streambits
        self.bitsSent = len(self.bitsTX)

        # Generating fixed signal s(t)
        self.t, self.s = Transmitter(fc=self.fc, output_print=False, output_plot=False, fs=self.fs, Rb=self.Rb).transmit(datagram=self.datagramTX) 

        # RX Chain
        self.receiver = Receiver(fc=self.fc, output_print=False, output_plot=False, fs=self.fs, Rb=self.Rb) 

    def simulate(self, ebn0_db):
        # Adding noise to the signal
        add_noise = NoiseEBN0(ebn0_db, fs=self.fs, Rb=self.Rb, seed=10)
        s_noisy = add_noise.add_noise(self.s)
        
        # Receiving bits
        bitsRX = self.receiver.receive(s_noisy)

        # Counting errors between bitsTX and bitsRX
        num_errors = sum(1 for tx, rx in zip(self.bitsTX, bitsRX) if tx != rx)
        return num_errors

    def run(self):
        r"""
        Runs the BER vs Eb/N0 simulation for the ARGOS-3 standard.

        Returns:
            ber_results (list): List of tuples (Eb/N0, BER) for each Eb/N0 value.
        """
        ber_results = []

        # Parallelize the simulations for each Eb/N0 using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Monitor the simulation progress for each Eb/N0 iteration
            for ebn0_db in self.EbN0_values:
                total_errors = 0
                repetitions = 0

                with tqdm(total=self.max_repetitions, desc=f"Simulando Eb/N0 = {ebn0_db} dB", ncols=100) as pbar:
                    while repetitions < self.max_repetitions and total_errors < self.error_values[int(ebn0_db)]:

                        # Create multiple simulations (one for each worker)
                        futures = [executor.submit(self.simulate, ebn0_db) for _ in range(self.num_workers)]  

                        for future in futures:
                            # Wait for the task to complete
                            num_errors = future.result() 
                            total_errors += num_errors
                            repetitions += 1
                            pbar.update(1)  

                        # If the error limit is reached, break the simulation
                        if total_errors >= self.error_values[int(ebn0_db)]:
                            break

                # Calculate the total number of bits transmitted (Repetitions * Bits of datagram)
                total_bits_transmitted = repetitions * self.bitsSent

                # Calculate the BER
                if total_bits_transmitted > 0:
                    ber = (total_errors + 1) / (total_bits_transmitted + 1)
                else:
                    ber = 0

                # Simulation status
                print(f"[ARGOS-3] Eb/N0={ebn0_db} dB -> Bits={total_bits_transmitted}, Erros={total_errors}, BER={ber}")

                # Store the tuple (Eb/N0, BER) in the list
                ber_results.append((ebn0_db, ber))

        return ber_results

class BERSNR_QPSK:
    def __init__(self, EbN0_values=np.arange(0, 10, 1), num_workers=8, num_bits=10_000, max_repetitions=2000, error_values=None):
        r"""
        Implements the BER vs Eb/N0 simulation for the QPSK standard.

        Args:
            EbN0_values (array-like): Values of Eb/N0 for which the simulation will be performed.
            num_workers (int): Number of threads for parallelization.
            num_bits (int): Number of bits for each simulation.
            max_repetitions (int): Maximum number of repetitions for each Eb/N0 value.
            error_values (array-like): Maximum number of errors for each Eb/N0 value.
        
        Raises:
            ValueError: If the number of errors is not the same as the number of Eb/N0 values.
        """

        if error_values is None or len(error_values) != len(EbN0_values):
            raise ValueError("error_values must have the same length as EbN0_values")

        # Variable system parameters
        self.EbN0_values = EbN0_values
        self.num_workers = num_workers
        self.num_bits = num_bits
        self.max_repetitions = max_repetitions
        self.error_values = error_values

    @staticmethod
    def simulate_qpsk(ebn0_db, num_bits=1000, bits_por_simbolo=2, rng=10):
        # Seed of the random number generator
        rng = np.random.default_rng(rng)

        # Generation of bits (I and Q independent)
        bI = rng.integers(0, 2, size=(num_bits,))
        bQ = rng.integers(0, 2, size=(num_bits,))

        # QPSK mapping
        I = (2*bI - 1) / np.sqrt(2)
        Q = (2*bQ - 1) / np.sqrt(2)

        # Complex signal
        signal = I + 1j*Q

        # Eb/N0 calculation
        ebn0_lin = 10 ** (ebn0_db / 10)
        signal_power = np.mean(np.abs(signal)**2)
        bit_energy = signal_power / bits_por_simbolo
        noise_density = bit_energy / ebn0_lin
        variance = noise_density / 2
        sigma = np.sqrt(variance)

        # AWGN channel
        noise = rng.normal(0.0, sigma, size=signal.shape) + 1j * rng.normal(0.0, sigma, size=signal.shape)
        r = signal + noise

        # Demodulation
        bI_dec = (r.real >= 0).astype(int)
        bQ_dec = (r.imag >= 0).astype(int)

        # Error count
        erros = np.count_nonzero(bI_dec != bI) + np.count_nonzero(bQ_dec != bQ)
        ber = erros / (2 * num_bits)
        return ber

    def run(self):
        r"""
        Runs the BER vs Eb/N0 simulation for the QPSK standard.

        Returns:
            ber_results (list): List of tuples (Eb/N0, BER) for each Eb/N0 value.
        """
        ber_results = []

        # Parallelize the simulations for each Eb/N0 using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for ebn0_db in self.EbN0_values:
                total_errors = 0
                repetitions = 0
                total_bits = 0

                # Monitor the simulation progress for each Eb/N0 iteration
                with tqdm(total=self.max_repetitions, desc=f"QPSK Eb/N0 = {ebn0_db} dB", ncols=100) as pbar:

                    # Execute the simulations for each worker
                    while repetitions < self.max_repetitions and total_errors < self.error_values[int(ebn0_db)]:
                        futures = [executor.submit(self.simulate_qpsk, ebn0_db, num_bits=self.num_bits, rng=np.random.default_rng())
                                   for _ in range(self.num_workers)]

                        # Wait for the completion of the simulations for each worker
                        for future in concurrent.futures.as_completed(futures):
                            ber = future.result()

                            # Error count (2 bits per symbol)
                            errors = int(ber * self.num_bits * 2) 
                            total_errors += errors
                            total_bits += self.num_bits * 2
                            repetitions += 1
                            pbar.update(1)

                        # If the error limit is reached, break the simulation
                        if total_errors >= self.error_values[int(ebn0_db)]:
                            break

                # Calculate the final BER
                if total_bits > 0:
                    ber_final = (total_errors + 1) / (total_bits + 1)
                else:
                    ber_final = 0

                # Simulation status
                print(f"[QPSK] Eb/N0={ebn0_db} dB -> Bits={total_bits}, Erros={total_errors}, BER={ber_final}")

                # Store the tuple (Eb/N0, BER) in the list
                ber_results.append((ebn0_db, ber_final))

        return ber_results

    def teorical_qpsk(self):
        r"""
        Calculates the theoretical BER vs Eb/N0 curve for QPSK, according to the expression below.

        $$
        P_b(x) = Q \left(x\right) \mapsto P_b(x) = Q\left(\sqrt{2 \cdot \frac{E_b}{N_0}}\right)
        $$

        Where:
            - $P_b(x)$: Error probability. 
            - $Q(x)$: Complementary error function.
            - $x$: Argument of the $Q(x)$ function.
            - $E_b$: Energy per bit.
            - $N_0$: Noise power. 

        Returns:
            ber_teorico (np.ndarray): Array of theoretical BER values for each Eb/N0 of the class.
        """

        ebn0_lin = 10 ** (self.EbN0_values / 10)
        
        # argument of the $Q(x)$ function
        x = np.sqrt(2 * ebn0_lin)

        # calculation of the $Q(x)$ function
        Qx = 0.5 * erfc(x / np.sqrt(2))
        return Qx

class BERSNR_QPSK_CONV:
    """
    BER vs Eb/N0 for QPSK with convolutional coding (rate 1/2).
    Maintains the loop and parallelization similar to BERSNR_QPSK.run().
    """
    def __init__(self, EbN0_values=np.arange(0, 10, 1), num_workers=8, num_bits=10_000,
                 max_repetitions=2000, error_values=None, decision="hard", G=None):
        if error_values is None or len(error_values) != len(EbN0_values):
            raise ValueError("error_values must have the same length as EbN0_values")

        self.EbN0_values = EbN0_values
        self.num_workers = num_workers
        self.num_bits = num_bits
        self.max_repetitions = max_repetitions
        self.error_values = error_values
        self.decision = decision.lower()

    @staticmethod
    def _get_rng(rng):
        # accepts seed (int/None) or a np.random.Generator
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng(rng)

    @staticmethod
    def simulate_conv(ebn0_db, num_bits=1000, rng=None, decision="hard"):
        """
        One execution: generates ut, encodes convolutionally (rate 1/2), maps to QPSK,
        passes through AWGN and does Viterbi (hard/soft). Returns number of errors (integer).
        """
        gen = BERSNR_QPSK_CONV._get_rng(rng)

        # generates input bits (ut)
        ut = gen.integers(0, 2, size=num_bits)

        # encodes convolutionally (returns vt0, vt1 of size num_bits)
        encoder = EncoderConvolutional()
        vt0, vt1 = encoder.encode(ut)

        # NRZ mapping to +/-1 and QPSK composition with normalization
        sI = 2 * vt0.astype(float) - 1.0   # +/-1
        sQ = 2 * vt1.astype(float) - 1.0   # +/-1
        signal = (sI + 1j * sQ) / np.sqrt(2)  # normalizado (energia por símbolo = 1)

        # AWGN according to Eb/N0
        ebn0_lin = 10 ** (ebn0_db / 10.0)
        signal_power = np.mean(np.abs(signal) ** 2)  # deve ser ~1
        bits_per_symbol = 2
        Eb = signal_power / bits_per_symbol
        N0 = Eb / ebn0_lin
        variance = N0 / 2.0
        sigma = np.sqrt(variance)

        noise = gen.normal(0.0, sigma, size=signal.shape) + 1j * gen.normal(0.0, sigma, size=signal.shape)
        r = signal + noise

        # Viterbi decoder
        decoder = DecoderViterbi(decision=decision)

        if decision == "hard":
            # detect bits demodulated (hard) and pass to Viterbi (integer inputs)
            bI_dec = (r.real >= 0).astype(int)
            bQ_dec = (r.imag >= 0).astype(int)
            ut_hat = decoder.decode(bI_dec, bQ_dec)  # já retorna bits (hard)
        else:
            # soft: scale r.real and r.imag to recover amplitude +/-1 before passing
            received0 = r.real * np.sqrt(2)
            received1 = r.imag * np.sqrt(2)
            llrs = decoder.decode(received0, received1)  # retorna LLRs (mm0 - mm1)
            ut_hat = (llrs >= 0).astype(int)

        L = min(len(ut_hat), len(ut))
        errors = int(np.count_nonzero(ut_hat[:L] != ut[:L]))
        return errors

    def run(self):
        ber_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for ebn0_db in self.EbN0_values:
                total_errors = 0
                repetitions = 0
                total_bits = 0

                with tqdm(total=self.max_repetitions, desc=f"CONV-{self.decision} Eb/N0 = {ebn0_db} dB", ncols=100) as pbar:
                    while (repetitions < self.max_repetitions) and (total_errors < self.error_values[int(ebn0_db)]):
                        # dispara N simulações paralelas (uma por worker)
                        futures = [executor.submit(self.simulate_conv, ebn0_db,
                                                   num_bits=self.num_bits,
                                                   rng=np.random.default_rng(),
                                                   decision=self.decision)
                                   for _ in range(self.num_workers)]

                        for future in concurrent.futures.as_completed(futures):
                            errors = future.result()
                            total_errors += errors
                            total_bits += self.num_bits
                            repetitions += 1
                            pbar.update(1)

                            if total_errors >= self.error_values[int(ebn0_db)]:
                                break

                ber = (total_errors + 1) / (total_bits + 1) if total_bits > 0 else 0.0
                print(f"[CONV-{self.decision}] Eb/N0={ebn0_db} dB -> Bits={total_bits}, Erros={total_errors}, BER={ber}")
                ber_results.append((ebn0_db, ber))
        return ber_results


if __name__ == "__main__":

    # Define the Eb/N0 values for the simulation
    EbN0_vec = np.arange(0, 9.5, 0.5)

    ref_values = [10000, 5000, 800, 200]
    ref_points = [0, 3, 6, 12]
    error_values = interpolate(len(EbN0_vec), ref_points, ref_values)

    # Print the maximum number of bits for each Eb/N0
    for ebn0, error in zip(EbN0_vec, error_values):
        print(f"Eb/N0 = {ebn0} dB: {error} erros")

    ### ARGOS-3
    reps = 1048576
    print(f"[ARGOS-3] Maximo de bits transmitidos por Eb/N0: {reps}")
    bersnr_argos = BERSNR_ARGOS(EbN0_values=EbN0_vec, error_values=error_values, num_workers=64, numblocks=1, max_repetitions=reps)

    ### QPSK
    bersnr_qpsk = BERSNR_QPSK(EbN0_values=EbN0_vec, error_values=error_values, num_workers=56, num_bits=50_000, max_repetitions=5000)
    bersnr_qpsk_hard = BERSNR_QPSK_CONV(EbN0_values=EbN0_vec, error_values=error_values, num_workers=56, num_bits=1000, max_repetitions=2000, decision="hard")
    bersnr_qpsk_soft = BERSNR_QPSK_CONV(EbN0_values=EbN0_vec, error_values=error_values, num_workers=56, num_bits=1000, max_repetitions=2000, decision="soft")

    # Simulation
    # ###############################################

    results = bersnr_argos.run()
    ExportData(results, "bersnr_argos").save()

    results_qpsk = bersnr_qpsk.run()
    ExportData(results_qpsk, "bersnr_qpsk").save()
    
    # PLOT
    # ###############################################

    # ARGOS-3
    bersnr_argos = ImportData("bersnr_argos").load()
    ber_values_argos = bersnr_argos[:, 1]

    print(ber_values_argos)

    # QPSK Ideal
    bersnr_qpsk_teorico = bersnr_qpsk.teorical_qpsk()
    
    # QPSK Simulated
    bersnr_qpsk = ImportData("bersnr_qpsk").load()
    ber_values_qpsk = bersnr_qpsk[:, 1]

    print(ber_values_qpsk)
    print(bersnr_qpsk_teorico)


    # BER vs Eb/N0
    fig, grid = create_figure(1, 1)
    BersnrPlot(fig, grid, 0,
               EbN0=EbN0_vec,
               ber_curves=[ber_values_argos, ber_values_qpsk, bersnr_qpsk_teorico],
               labels=["ARGOS-3", "QPSK Simulated", "QPSK Ideal"],
               linestyles=["-", "-", ":"],
               markers=["o", "s", "x"],
               title="BER vs $E_b/N_0$",
               ylim=(1e-5, 1),
               xlim=(-1, 10)
    ).plot()
    
    save_figure(fig, "ber_vs_ebn0.pdf")
