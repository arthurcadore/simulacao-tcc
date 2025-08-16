"""
Implementação de simulação para curva BER vs SNR. 

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
from noise import Noise
import os

def simulate_ber(snr_db, numblocks=8, fs=128_000, Rb=400):
    r"""
    Simula a transmissão e recepção de um datagrama ARGOS-III, para um dado SNR.

    Args: 
        snr_db (float): Relação sinal-ruído em decibéis.
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

    add_noise = Noise(snr=snr_db)
    s_noisy = add_noise.add_noise(s)

    receiver = Receiver(fs=fs, Rb=Rb, output_print=False, output_plot=False)
    bitsRX = receiver.run(s_noisy, t)

    # Calcula BER
    num_errors = sum(1 for tx, rx in zip(bitsTX, bitsRX) if tx != rx)
    ber = num_errors / len(bitsTX)
    return ber

def run_simulation(SNR_values=np.arange(-30, 31, 1), repetitions=10, numblocks=1, num_workers=16):
    r"""
    Executa a simulação completa de BER vs SNR. Retorna a tupla BER vs SNR.

    Args: 
        SNR_values (np.ndarray): Valores de SNR a serem simulados.
        repetitions (int): Número de repetições para cada valor de SNR.
        numblocks (int): Número de blocos a serem transmitidos.
        num_workers (int): Número de trabalhadores para a execução paralela.

    Returns:
        list: Lista de tuplas (SNR, BER_médio).
    """
    results = []
    total_tasks = len(SNR_values) * repetitions
    ber_accumulator = {snr: [] for snr in SNR_values}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(simulate_ber, snr, numblocks): snr
            for snr in SNR_values
            for _ in range(repetitions)
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Simulando"):
            snr = futures[future]
            try:
                ber = future.result()
                ber_accumulator[snr].append(ber)
            except Exception as e:
                print(f"Erro na simulação SNR={snr}: {e}")

    # Calcula média por SNR
    for snr in SNR_values:
        mean_ber = np.mean(ber_accumulator[snr])
        results.append((snr, mean_ber))

    return results

# TODO: Implementar salvamento dos dados em formato binário para diminuir o tamanho do arquivo e facilitar a leitura posterior.
def save_results(results, filename="../out/snr_vs_ber.txt"):
    r"""
    Salva os resultados de simulação em um arquivo .txt 

    Args:
        results (list): Lista de tuplas (SNR, BER) a serem salvas.
        filename (str): Caminho do arquivo de saída.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.normpath(os.path.join(script_dir, filename))

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("SNR_dB\tBER\n")
        for snr, ber in results:
            f.write(f"{snr}\t{ber:.8e}\n")

def plot_from_file(filename="../out/snr_vs_ber.txt", out_pdf="../out/snr_vs_ber.pdf"):
    r"""
    Lê os resultados do arquivo TXT (relativo ao diretório do script) e gera o gráfico BER vs SNR em PDF.

    Args:
        filename (str): Caminho do arquivo de entrada.
        out_pdf (str): Caminho do arquivo PDF de saída. 

    ![bersnr](assets/snr_vs_ber.svg)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.normpath(os.path.join(script_dir, filename))
    outpath = os.path.normpath(os.path.join(script_dir, out_pdf))

    data = np.loadtxt(filepath, skiprows=1)  # ignora cabeçalho
    snr_list, ber_list = data[:, 0], data[:, 1]

    plt.figure()
    plt.semilogy(snr_list, ber_list, marker='o')
    plt.grid(True, which="both", ls="--")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("SNR vs BER")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()

if __name__ == "__main__":
    results = run_simulation()
    save_results(results)
    plot_from_file()