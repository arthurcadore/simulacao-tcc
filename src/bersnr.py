"""
Implementação de simulação para curva BER vs Eb/N0. 

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm
from datagram import Datagram
from transmitter import Transmitter
from receiver import Receiver
from noise import NoiseEBN0
import os

def simulate_ber(ebn0_db, numblocks=8, fs=128_000, Rb=400):
    r"""
    Simula a transmissão e recepção de um datagrama ARGOS-III, para um dado Eb/N0.

    Args: 
        ebn0_db (float): Relação Eb/N0 em decibéis.
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
    add_noise = NoiseEBN0(ebn0_db, bits_per_symbol=2, fs=fs, Rb=Rb)
    s_noisy = add_noise.add_noise(s)

    receiver = Receiver(fs=fs, Rb=Rb, output_print=False, output_plot=False)
    bitsRX = receiver.run(s_noisy, t)

    # Calcula BER
    num_errors = sum(1 for tx, rx in zip(bitsTX, bitsRX) if tx != rx)
    ber = num_errors / len(bitsTX)
    return ber

def run_simulation(EbN0_values=np.arange(0, 12, 0.5), repetitions=100, numblocks=8, num_workers=24):
    r"""
    Executa a simulação completa de BER vs Eb/N0. Retorna a tupla BER vs Eb/N0.

    Args: 
        EbN0_values (np.ndarray): Valores de Eb/N0 a serem simulados.
        repetitions (int): Número de repetições para cada valor.
        numblocks (int): Número de blocos a serem transmitidos.
        num_workers (int): Número de processos para execução paralela.

    Returns:
        list: Lista de tuplas (Eb/N0, BER_médio).
    """
    results = []
    total_tasks = len(EbN0_values) * repetitions
    ber_accumulator = {ebn0: [] for ebn0 in EbN0_values}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(simulate_ber, ebn0, numblocks): ebn0
            for ebn0 in EbN0_values
            for _ in range(repetitions)
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Simulando"):
            ebn0 = futures[future]
            try:
                ber = future.result()
                ber_accumulator[ebn0].append(ber)
            except Exception as e:
                print(f"Erro na simulação Eb/N0={ebn0}: {e}")

    # Calcula média por Eb/N0
    for ebn0 in EbN0_values:
        mean_ber = np.mean(ber_accumulator[ebn0])
        results.append((ebn0, mean_ber))

    return results

def save_results(results, filename="../out/ebn0_vs_ber.txt"):
    r"""
    Salva os resultados de simulação em um arquivo .txt 

    Args:
        results (list): Lista de tuplas (Eb/N0, BER) a serem salvas.
        filename (str): Caminho do arquivo de saída.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.normpath(os.path.join(script_dir, filename))

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("EbN0_dB\tBER\n")
        for ebn0, ber in results:
            f.write(f"{ebn0}\t{ber:.8e}\n")

def plot_from_file(filename="../out/ebn0_vs_ber.txt", out_pdf="../out/ebn0_vs_ber.pdf"):
    r"""
    Lê os resultados do arquivo TXT e gera o gráfico BER vs Eb/N0 em PDF.

    Args:
        filename (str): Caminho do arquivo de entrada.
        out_pdf (str): Caminho do arquivo PDF de saída. 
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.normpath(os.path.join(script_dir, filename))
    outpath = os.path.normpath(os.path.join(script_dir, out_pdf))

    data = np.loadtxt(filepath, skiprows=1)
    ebn0_list, ber_list = data[:, 0], data[:, 1]

    plt.figure()
    plt.semilogy(ebn0_list, ber_list, marker='o', label='BER vs Eb/N0')
    plt.grid(True, which="both", ls="--")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.xlim(0, 20)
    plt.ylim(1e-5, 1)

    leg1 = plt.legend(
            loc='upper right', frameon=True, edgecolor='black',
            facecolor='white', fontsize=12, fancybox=True
            )
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_edgecolor('black')
    leg1.get_frame().set_alpha(1.0)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()

if __name__ == "__main__":
    results = run_simulation()
    save_results(results)
    plot_from_file()
