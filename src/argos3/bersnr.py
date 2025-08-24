# """
# Implementação de simulação para curva BER vs Eb/N0. 

# Autor: Arthur Cadore
# Data: 28-07-2025
# """
import numpy as np
import concurrent.futures
from tqdm import tqdm

from .datagram import Datagram
from .transmitter import Transmitter
from .receiver import Receiver
from .noise import NoiseEBN0
from .data import ExportData, ImportData
from .plotter import BersnrPlot, create_figure, save_figure

def repetitions_for_ebn0(ebn0: float) -> int:
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
        ebn0 (float): Relação $Eb/N_0$ em decibéis.

    Returns:
        int: Número de repetições, arredondado para o valor inteiro mais próximo.
    """
    ebn0_ref = [0, 1, 4, 6, 8]
    reps_ref = [2000, 4000, 20000, 40000, 60000]

    bean = np.interp(ebn0, ebn0_ref, reps_ref)

    repetitions = int(round(bean))

    print(f"Repetições para {ebn0} dB: {repetitions}")
    return repetitions


def simulate_argos(ebn0_db, numblocks=8, fs=128_000, Rb=400):
    r"""
    Simula a transmissão e recepção de um datagrama ARGOS-3, para um dado $Eb/N0$, retornando a taxa de erro de bit (BER) simulada.

    Args: 
        ebn0_db (float): Relação $Eb/N0$ em decibéis.
        numblocks (int): Número de blocos a serem transmitidos.
        fs (int): Frequência de amostragem.
        Rb (int): Taxa de bits. 

    Returns: 
        ber (float): A taxa de erro de bit (BER) simulada.
    """
    datagramTX = Datagram(pcdnum=1234, numblocks=numblocks)
    bitsTX = datagramTX.streambits

    transmitter = Transmitter(datagramTX, output_print=False, output_plot=False)
    t, s = transmitter.run()

    # Canal AWGN baseado em Eb/N0
    add_noise = NoiseEBN0(ebn0_db, fs=fs, Rb=Rb)
    s_noisy = add_noise.add_noise(s)

    receiver = Receiver(fs=fs, Rb=Rb, output_print=False, output_plot=False)
    bitsRX = receiver.run(s_noisy, t)

    # Calcula BER
    num_errors = sum(1 for tx, rx in zip(bitsTX, bitsRX) if tx != rx)
    ber = num_errors / len(bitsTX)
    return ber

# TODO: Alterar função para operar usando add_noise
def simulate_qpsk(ebn0_db, num_bits=1000, bits_por_simbolo=2, rng=None):
    r"""
    Simula a transmissão e recepção QPSK em canal AWGN para um dado $Eb/N0$, retornando a taxa de erro de bit ($BER$) simulada.

    Args:
        ebn0_db (float): Relação $Eb/N0$ em dB.
        num_bits (int): Número de bits simulados.
        bits_por_simbolo (int): Número de bits por símbolo $k$, ($QPSK = 2$).
        rng (np.random.Generator, opcional): gerador de números aleatórios.

    Returns:
        ber (float): BER simulada.
    """
    rng = rng if rng is not None else np.random.default_rng()

    # Geração dos bits (I e Q independentes)
    bI = rng.integers(0, 2, size=(num_bits,))
    bQ = rng.integers(0, 2, size=(num_bits,))

    # Mapeamento QPSK normalizado (Es=1, Gray coding)
    I = (2*bI - 1) / np.sqrt(2)
    Q = (2*bQ - 1) / np.sqrt(2)
    signal = I + 1j*Q

    # Cálculo do Eb/N0
    ebn0_lin = 10 ** (ebn0_db / 10)
    signal_power = np.mean(np.abs(signal)**2)
    bit_energy = signal_power / bits_por_simbolo
    noise_density = bit_energy / ebn0_lin
    variance = noise_density / 2
    sigma = np.sqrt(variance)

    noise = rng.normal(0.0, sigma, size=signal.shape) + 1j * rng.normal(0.0, sigma, size=signal.shape)
    r = signal + noise

    bI_dec = (r.real >= 0).astype(int)
    bQ_dec = (r.imag >= 0).astype(int)

    erros = np.count_nonzero(bI_dec != bI) + np.count_nonzero(bQ_dec != bQ)
    ber = erros / (2 * num_bits)
    return ber


def run(EbN0_values=np.arange(0, 12, 0.5), num_workers=28):
    r"""
    Executa a simulação completa de $BER$ vs $Eb/N0$ para as funções de simulação implementadas.

    Args: 
        EbN0_values (np.ndarray): Valores de $Eb/N0$ a serem simulados.
        num_workers (int): Número de processos para execução paralela.

    Returns:
        results (np.ndarray): Array de [Eb/N0, BER_ARGOS_médio, BER_QPSK_médio].

    Exemplos:
        - Argos e QPSK: ![pageplot](assets/ber_vs_ebn0.svg)
    """
    results = []
    ber_accumulator_argos = {ebn0: [] for ebn0 in EbN0_values}
    ber_accumulator_qpsk = {ebn0: [] for ebn0 in EbN0_values}

    # --- Simulação ARGOS-3 ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures_argos = {
            executor.submit(simulate_argos, ebn0): ebn0
            for ebn0 in EbN0_values
            for _ in range(repetitions_for_ebn0(ebn0))
        }
        for future in tqdm(concurrent.futures.as_completed(futures_argos),
                           total=len(futures_argos),
                           desc="Simulando ARGOS"):
            ebn0 = futures_argos[future]
            try:
                ber = future.result()
                ber_accumulator_argos[ebn0].append(ber)
            except Exception as e:
                print(f"Erro na simulação ARGOS Eb/N0={ebn0}: {e}")

    # Calcula média para ARGOS
    for ebn0 in EbN0_values:
        mean_ber_argos = np.mean(ber_accumulator_argos[ebn0])
        results.append([ebn0, mean_ber_argos, None])

    # --- Simulação QPSK ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures_qpsk = {
            executor.submit(simulate_qpsk, ebn0): ebn0
            for ebn0 in EbN0_values
            for _ in range(repetitions_for_ebn0(ebn0))
        }
        for future in tqdm(concurrent.futures.as_completed(futures_qpsk),
                           total=len(futures_qpsk),
                           desc="Simulando QPSK"):
            ebn0 = futures_qpsk[future]
            try:
                ber = future.result()
                ber_accumulator_qpsk[ebn0].append(ber)
            except Exception as e:
                print(f"Erro na simulação QPSK Eb/N0={ebn0}: {e}")

    # Calcula média para QPSK
    for i, ebn0 in enumerate(EbN0_values):
        mean_ber_qpsk = np.mean(ber_accumulator_qpsk[ebn0])
        results[i][2] = mean_ber_qpsk

    return results

if __name__ == "__main__":
    results = run()  
    results = np.array(results)

    # Salvar os resultados no formato adequado
    ExportData(results, "bersnr").save()

    print(results)

    import_data = ImportData("bersnr").load()

    ebn0_values = import_data[:, 0]  
    ber_values_argos = import_data[:, 1]   
    ber_values_qpsk = import_data[:, 2]   

    # Criando o gráfico
    fig, grid = create_figure(rows=1, cols=1)
    ber_plot = BersnrPlot(
        fig=fig,
        grid=grid,
        pos=0,
        ebn0=ebn0_values,
        ber_values=[ber_values_argos, ber_values_qpsk],
        labels=["ARGOS-3 (Default)", "QPSK"]
    )
    ber_plot.plot(ylim=(1e-6, 1))
    save_figure(fig, "ber_vs_ebn0.pdf")


