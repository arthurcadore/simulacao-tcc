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
        ref_points (array-like): Pontos de referência (por exemplo, os valores de Eb/N0).
        ref_values (array-like): Valores correspondentes aos pontos de referência (por exemplo, os valores de erro).
    
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
        self.t, self.s = Transmitter(fc=self.fc, datagram=self.datagramTX, output_print=False, output_plot=False, fs=self.fs, Rb=self.Rb).run() 

        # Cadeia de RX
        self.receiver = Receiver(fc=self.fc, output_print=False, output_plot=False, fs=self.fs, Rb=self.Rb) 

    def simulate(self, ebn0_db):
        # Adicionando ruído ao sinal
        add_noise = NoiseEBN0(ebn0_db, fs=self.fs, Rb=self.Rb, seed=10)
        s_noisy = add_noise.add_noise(self.s)
        
        # Recebendo bits
        bitsRX = self.receiver.run(s_noisy, self.t)

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
        P_b(x) = Q \left(x\right) \mapsto P_b(x) = \frac{1}{2} \operatorname{erfc} \left(\frac{x}{\sqrt{2}}\right), \text{ onde } x = \sqrt{2 \cdot \frac{E_b}{N_0}}
        $$

        Podendo simplificar como: 

        $$
        P_b(x) = \frac{1}{2} \operatorname{erfc} \left(\sqrt{\frac{E_b}{N_0}}\right)
        $$

        Sendo:
            - $P_b(x)$: Probabilidade de erro. 
            - $Q(x)$: Função de erro complementar.
            - $x$: Argumento da função $Q(x)$.
            - $E_b$: Energia por bit.
            - $N_0$: Potência do ruído. 
            - $erfc$: Função de erro complementar.

        Returns:
            ber_teorico (np.ndarray): Vetor de valores de BER teórica para cada Eb/N0 da classe.
        """

        ebn0_lin = 10 ** (self.EbN0_values / 10)
        
        # argumento da função Q(x)
        x = np.sqrt(2 * ebn0_lin)
        ber_teorico = 0.5 * erfc(x / np.sqrt(2))

        return ber_teorico



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

    # Simulação
    # ###############################################

    results = bersnr_argos.run()
    ExportData(results, "bersnr_argos").save()

    results_qpsk = bersnr_qpsk.run()
    ExportData(results_qpsk, "bersnr_qpsk").save()
    
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
