# """
# Implementação de simulação para curva BER vs Eb/N0. 

# Autor: Arthur Cadore
# Data: 8-09-2025
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
    Define o número de repetições em função do $Eb/N0$, usando interpolação linear entre pontos de referência, dada pela expressão abaixo.

    $$
    r = r_{i} + \frac{(EBN0 - EBN0_{i})}{(EBN0_{i+1} - EBN0_{i})} \cdot (r_{i+1} - r_{i})
    $$

    Onde:
        - $r$: Número de repetições.
        - $EBN0$: Relação $Eb/N_0$ em decibéis.
        - $r_i$ e $r_{i+1}$: Número de repetições nos pontos de referência próximos.
        - $EBN0_i$ e $EBN0_{i+1}$: Relações $Eb/N_0$ nos pontos de referência próximos.

    Args: 
        positions (int): O número total de pontos a serem gerados.
        ref_points (array-like): Pontos de referência. 
        ref_values (array-like): Valores correspondentes aos pontos de referência.
    
    Returns:
        interpolated_values (np.ndarray): Vetor de valores interpolados, arredondados para inteiros.
    """
    # Garante que as entradas são arrays numpy
    ref_points = np.array(ref_points)
    ref_values = np.array(ref_values)
    
    # Realiza a interpolação linear usando np.interp
    interpolated_values = np.interp(np.linspace(ref_points[0], ref_points[-1], positions), ref_points, ref_values)
    
    # Arredonda os valores e converte para inteiros
    interpolated_values = np.round(interpolated_values).astype(int)
    
    return interpolated_values

class BERSNR_ARGOS: 
    def __init__(self, EbN0_values=np.arange(0, 10, 1), num_workers=56, numblocks=8, max_repetitions=2000, error_values=None):
        r"""
        Implementa a simulação de BER vs Eb/N0 para o padrão ARGOS-3.

        Args:
            EbN0_values (array-like): Valores de Eb/N0 para os quais a simulação será realizada.
            num_workers (int): Número de threads para paralelização.
            numblocks (int): Número de blocos de dados para cada datagrama.
            max_repetitions (int): Número máximo de repetições para cada valor de Eb/N0.
            error_values (array-like): Número máximo de erros para cada valor de Eb/N0.
        
        Raises:
            ValueError: Se o número de erros não for o mesmo que o número de valores de Eb/N0.

        Example: 
            ![pageplot](assets/ber_vs_ebn0.svg)
        """
        if len(error_values) != len(EbN0_values):
            raise ValueError("error_values deve ter o mesmo tamanho que EbN0_values")

        # Parâmetros fixos do sistema
        self.fs = 128_000
        self.Rb = 400
        self.fc = 4000

        # Parâmetros variáveis
        self.EbN0_values = EbN0_values
        self.num_workers = num_workers
        self.numblocks = numblocks
        self.max_repetitions = max_repetitions
        self.error_values = error_values

        # Cadeia de TX
        self.datagramTX = Datagram(pcdnum=1234, numblocks=numblocks, seed=10)
        self.bitsTX = self.datagramTX.streambits
        self.bitsSent = len(self.bitsTX)

        # Gerando sinal fixo s(t)
        self.t, self.s = Transmitter(fc=self.fc, output_print=False, output_plot=False, fs=self.fs, Rb=self.Rb).transmit(datagram=self.datagramTX) 

        # Cadeia de RX
        self.receiver = Receiver(fc=self.fc, output_print=False, output_plot=False, fs=self.fs, Rb=self.Rb) 

    def simulate(self, ebn0_db):
        # Adicionando ruído ao sinal
        add_noise = NoiseEBN0(ebn0_db, fs=self.fs, Rb=self.Rb, seed=10)
        s_noisy = add_noise.add_noise(self.s)
        
        # Recebendo bits
        bitsRX = self.receiver.receive(s_noisy)

        # Contando erros entre bitsTX e bitsRX
        num_errors = sum(1 for tx, rx in zip(self.bitsTX, bitsRX) if tx != rx)
        return num_errors

    def run(self):
        r"""
        Executa a simulação de BER vs Eb/N0 para o padrão ARGOS-3.

        Returns:
            ber_results (list): Lista de tuplas (Eb/N0, BER) para cada valor de Eb/N0.
        """
        ber_results = []

        # Paralelizar as simulações para cada Eb/N0 usando Pool de Workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # monitorar o progresso da simulação em cada iteração de Eb/N0
            for ebn0_db in self.EbN0_values:
                total_errors = 0
                repetitions = 0

                with tqdm(total=self.max_repetitions, desc=f"Simulando Eb/N0 = {ebn0_db} dB", ncols=100) as pbar:
                    while repetitions < self.max_repetitions and total_errors < self.error_values[int(ebn0_db)]:

                        # Cria múltiplas simulações (uma para cada worker)
                        futures = [executor.submit(self.simulate, ebn0_db) for _ in range(self.num_workers)]  

                        for future in futures:
                            # Aguarda a conclusão da tarefa
                            num_errors = future.result() 
                            total_errors += num_errors
                            repetitions += 1
                            pbar.update(1)  

                        # Se atingiu o limite de erros, interrompe a simulação
                        if total_errors >= self.error_values[int(ebn0_db)]:
                            break

                # Calcula o número total de bits transmitidos (Repetições * Bits do datagrama)
                total_bits_transmitted = repetitions * self.bitsSent

                # Calcula a BER
                if total_bits_transmitted > 0:
                    ber = (total_errors + 1) / (total_bits_transmitted + 1)
                else:
                    ber = 0

                # Status da simulação
                print(f"[ARGOS-3] Eb/N0={ebn0_db} dB -> Bits={total_bits_transmitted}, Erros={total_errors}, BER={ber}")

                # Armazena a tupla (Eb/N0, BER) na lista
                ber_results.append((ebn0_db, ber))

        return ber_results

class BERSNR_QPSK:
    def __init__(self, EbN0_values=np.arange(0, 10, 1), num_workers=8, num_bits=10_000, max_repetitions=2000, error_values=None):
        r"""
        Implementa a simulação de BER vs Eb/N0 para o padrão QPSK.

        Args:
            EbN0_values (array-like): Valores de Eb/N0 para os quais a simulação será realizada.
            num_workers (int): Número de threads para paralelização.
            num_bits (int): Número de bits para cada simulação.
            max_repetitions (int): Número máximo de repetições para cada valor de Eb/N0.
            error_values (array-like): Número máximo de erros para cada valor de Eb/N0.
        
        Raises:
            ValueError: Se o número de erros não for o mesmo que o número de valores de Eb/N0.
        """

        if error_values is None or len(error_values) != len(EbN0_values):
            raise ValueError("error_values deve ter o mesmo tamanho que EbN0_values")

        # Parâmetros variáveis
        self.EbN0_values = EbN0_values
        self.num_workers = num_workers
        self.num_bits = num_bits
        self.max_repetitions = max_repetitions
        self.error_values = error_values

    @staticmethod
    def simulate_qpsk(ebn0_db, num_bits=1000, bits_por_simbolo=2, rng=10):
        # Seed do gerador de números aleatórios
        rng = np.random.default_rng(rng)

        # Geração dos bits (I e Q independentes)
        bI = rng.integers(0, 2, size=(num_bits,))
        bQ = rng.integers(0, 2, size=(num_bits,))

        # Mapeamento QPSK
        I = (2*bI - 1) / np.sqrt(2)
        Q = (2*bQ - 1) / np.sqrt(2)

        # Sinal complexo
        signal = I + 1j*Q

        # Cálculo do Eb/N0
        ebn0_lin = 10 ** (ebn0_db / 10)
        signal_power = np.mean(np.abs(signal)**2)
        bit_energy = signal_power / bits_por_simbolo
        noise_density = bit_energy / ebn0_lin
        variance = noise_density / 2
        sigma = np.sqrt(variance)

        # Canal AWGN
        noise = rng.normal(0.0, sigma, size=signal.shape) + 1j * rng.normal(0.0, sigma, size=signal.shape)
        r = signal + noise

        # Demodulação
        bI_dec = (r.real >= 0).astype(int)
        bQ_dec = (r.imag >= 0).astype(int)

        # Contagem de erros
        erros = np.count_nonzero(bI_dec != bI) + np.count_nonzero(bQ_dec != bQ)
        ber = erros / (2 * num_bits)
        return ber

    def run(self):
        r"""
        Executa a simulação de BER vs Eb/N0 para o padrão QPSK.

        Returns:
            ber_results (list): Lista de tuplas (Eb/N0, BER) para cada valor de Eb/N0.
        """
        ber_results = []

        # Paralelizar as simulações para cada Eb/N0 usando Pool de Workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for ebn0_db in self.EbN0_values:
                total_errors = 0
                repetitions = 0
                total_bits = 0

                # Monitorar o progresso da simulação em cada iteração de Eb/N0
                with tqdm(total=self.max_repetitions, desc=f"QPSK Eb/N0 = {ebn0_db} dB", ncols=100) as pbar:

                    # Executar as simulações para cada worker
                    while repetitions < self.max_repetitions and total_errors < self.error_values[int(ebn0_db)]:
                        futures = [executor.submit(self.simulate_qpsk, ebn0_db, num_bits=self.num_bits, rng=np.random.default_rng())
                                   for _ in range(self.num_workers)]

                        # Aguardar a conclusão das simulações para cada worker
                        for future in concurrent.futures.as_completed(futures):
                            ber = future.result()

                            # Contagem de erros (2 bits por símbolo)
                            errors = int(ber * self.num_bits * 2) 
                            total_errors += errors
                            total_bits += self.num_bits * 2
                            repetitions += 1
                            pbar.update(1)

                        # Se atingiu o limite de erros, interrompe a simulação
                        if total_errors >= self.error_values[int(ebn0_db)]:
                            break

                # Calcula a BER final
                if total_bits > 0:
                    ber_final = (total_errors + 1) / (total_bits + 1)
                else:
                    ber_final = 0

                # Status da simulação
                print(f"[QPSK] Eb/N0={ebn0_db} dB -> Bits={total_bits}, Erros={total_errors}, BER={ber_final}")

                # Armazena a tupla (Eb/N0, BER) na lista
                ber_results.append((ebn0_db, ber_final))

        return ber_results

    def teorical_qpsk(self):
        r"""
        Calcula a curva teórica de $BER$ vs $Eb/N_0$ para QPSK, segundo a expressão abaixo.

        $$
        P_b(x) = Q \left(x\right) \mapsto P_b(x) = Q\left(\sqrt{2 \cdot \frac{E_b}{N_0}}\right)
        $$

        Sendo:
            - $P_b(x)$: Probabilidade de erro. 
            - $Q(x)$: Função de erro complementar.
            - $x$: Argumento da função $Q(x)$.
            - $E_b$: Energia por bit.
            - $N_0$: Potência do ruído. 

        Returns:
            ber_teorico (np.ndarray): Vetor de valores de BER teórica para cada Eb/N0 da classe.
        """

        ebn0_lin = 10 ** (self.EbN0_values / 10)
        
        # argumento da função Q(x)
        x = np.sqrt(2 * ebn0_lin)

        # calculo da função Q(x)
        Qx = 0.5 * erfc(x / np.sqrt(2))
        return Qx

class BERSNR_QPSK_CONV:
    """
    BER vs Eb/N0 para QPSK com codificação convolucional (rate 1/2).
    Mantém o laço e paralelização parecidos com BERSNR_QPSK.run().
    """
    def __init__(self, EbN0_values=np.arange(0, 10, 1), num_workers=8, num_bits=10_000,
                 max_repetitions=2000, error_values=None, decision="hard", G=None):
        if error_values is None or len(error_values) != len(EbN0_values):
            raise ValueError("error_values deve ter o mesmo tamanho que EbN0_values")

        self.EbN0_values = EbN0_values
        self.num_workers = num_workers
        self.num_bits = num_bits
        self.max_repetitions = max_repetitions
        self.error_values = error_values
        self.decision = decision.lower()

    @staticmethod
    def _get_rng(rng):
        # aceita seed (int/None) ou um np.random.Generator
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng(rng)

    @staticmethod
    def simulate_conv(ebn0_db, num_bits=1000, rng=None, decision="hard"):
        """
        Uma execução: gera ut, codifica convolucional (rate 1/2), mapeia p/ QPSK,
        passa por AWGN e faz Viterbi (hard/soft). Retorna número de erros (inteiro).
        """
        gen = BERSNR_QPSK_CONV._get_rng(rng)

        # 1) gera bits de entrada (ut)
        ut = gen.integers(0, 2, size=num_bits)

        # 2) codifica convolucional (retorna vt0, vt1 de tamanho num_bits)
        encoder = EncoderConvolutional()
        vt0, vt1 = encoder.encode(ut)

        # 3) mapeamento NRZ para +/-1 e composição QPSK com normalização
        sI = 2 * vt0.astype(float) - 1.0   # +/-1
        sQ = 2 * vt1.astype(float) - 1.0   # +/-1
        signal = (sI + 1j * sQ) / np.sqrt(2)  # normalizado (energia por símbolo = 1)

        # 4) AWGN conforme Eb/N0
        ebn0_lin = 10 ** (ebn0_db / 10.0)
        signal_power = np.mean(np.abs(signal) ** 2)  # deve ser ~1
        bits_per_symbol = 2
        Eb = signal_power / bits_per_symbol
        N0 = Eb / ebn0_lin
        variance = N0 / 2.0
        sigma = np.sqrt(variance)

        noise = gen.normal(0.0, sigma, size=signal.shape) + 1j * gen.normal(0.0, sigma, size=signal.shape)
        r = signal + noise

        # 5) decodificador Viterbi
        decoder = DecoderViterbi(decision=decision)

        if decision == "hard":
            # detecta bits demodulados (hard) e passa pro Viterbi (entradas inteiras)
            bI_dec = (r.real >= 0).astype(int)
            bQ_dec = (r.imag >= 0).astype(int)
            ut_hat = decoder.decode(bI_dec, bQ_dec)  # já retorna bits (hard)
        else:
            # soft: escala os r.real e r.imag para recuperar amplitude +/-1 antes de passar
            # lembre: signal.real = sI / sqrt(2); multiplicando por sqrt(2) recuperamos sI + ruído escalado
            received0 = r.real * np.sqrt(2)
            received1 = r.imag * np.sqrt(2)
            llrs = decoder.decode(received0, received1)  # retorna LLRs (mm0 - mm1)
            # converter LLR -> bit: se llr >= 0 => bit=1 (ver comentário no decode)
            ut_hat = (llrs >= 0).astype(int)

        # 6) contar erros (comparando com ut original)
        # garantir compatibilidade de comprimento (por precaução)
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

    # Define os valores de Eb/N0 para a simulação
    EbN0_vec = np.arange(0, 9.5, 0.5)

    ref_values = [10000, 5000, 800, 200]
    ref_points = [0, 3, 6, 12]
    error_values = interpolate(len(EbN0_vec), ref_points, ref_values)

    # Imprime os valores de erro máximo para cada Eb/N0
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

    # Simulação
    # ###############################################

    results = bersnr_argos.run()
    ExportData(results, "bersnr_argos").save()

    results_qpsk = bersnr_qpsk.run()
    ExportData(results_qpsk, "bersnr_qpsk").save()

    results_conv_hard = bersnr_qpsk_hard.run()
    ExportData(results_conv_hard, "bersnr_conv_hard").save()

    results_conv_soft = bersnr_qpsk_soft.run()
    ExportData(results_conv_soft, "bersnr_conv_soft").save()
    
    # PLOT
    # ###############################################

    # extrair os valores de Eb/N0 e BER
    bersnr_argos = ImportData("bersnr_argos").load()
    ber_values_argos = bersnr_argos[:, 1]

    print(ber_values_argos)

    # QPSK Teorico
    bersnr_qpsk_teorico = bersnr_qpsk.teorical_qpsk()
    
    # extrair os valores de Eb/N0 e BER
    bersnr_qpsk = ImportData("bersnr_qpsk").load()
    ber_values_qpsk = bersnr_qpsk[:, 1]

    print(ber_values_qpsk)
    print(bersnr_qpsk_teorico)


    # extrair os valores de Eb/N0 e BER teórico
    fig, grid = create_figure(1, 1)
    BersnrPlot(fig, grid, 0,
               EbN0=EbN0_vec,
               ber_curves=[ber_values_argos, ber_values_qpsk, bersnr_qpsk_teorico],
               labels=["ARGOS-3", "QPSK Simulado", "QPSK Ideal"],
               linestyles=["-", "-", ":"],
               markers=["o", "s", "x"],
               title="Curva BER vs Eb/N0",
               ylim=(1e-5, 1),
               xlim=(-1, 10)
    ).plot()
    
    save_figure(fig, "ber_vs_ebn0.pdf")
