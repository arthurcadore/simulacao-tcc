"""
Implementação de simulação para curva BER vs Eb/N0. 

Autor: Arthur Cadore
Data: 28-07-2025
"""

import os
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm
from datagram import Datagram
from transmitter import Transmitter
from receiver import Receiver
from noise import NoiseEBN0
from data import ExportData, ImportData
from plotter import BersnrPlot, create_figure, save_figure

def simulate_argos(ebn0_db, numblocks=8, fs=128_000, Rb=400):
    r"""
    Simula a transmissão e recepção de um datagrama ARGOS-3, para um dado Eb/N0.

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

def run(EbN0_values=np.arange(0, 15, 0.5), repetitions=10, num_workers=24):
    r"""
    Executa a simulação completa de BER vs Eb/N0. Retorna a tupla BER vs Eb/N0.

    Args: 
        EbN0_values (np.ndarray): Valores de Eb/N0 a serem simulados.
        repetitions (int): Número de repetições para cada valor.
        num_workers (int): Número de processos para execução paralela.

    Returns:
        list: Lista de tuplas (Eb/N0, BER_médio).
    """
    results = []
    total_tasks = len(EbN0_values) * repetitions
    ber_accumulator = {ebn0: [] for ebn0 in EbN0_values}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(simulate_argos, ebn0): ebn0
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

if __name__ == "__main__":
    results = run()
    results = np.array(results)
    ExportData(results, "bersnr").save()

    import_data = ImportData("bersnr").load()
    ebn0_values = import_data[:, 0]  
    ber_values = import_data[:, 1]   

    fig, grid = create_figure(rows=1, cols=1)
    ber_plot = BersnrPlot(
        fig=fig,
        grid=grid,
        pos=0,
        ebn0=ebn0_values,
        ber_values=[ber_values],
        labels=["ARGOS-3 (default)"]
    )
    ber_plot.plot()
    save_figure(fig, "ber_vs_ebn0.pdf")

