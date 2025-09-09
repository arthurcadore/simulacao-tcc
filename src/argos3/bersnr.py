# """
# Implementação de simulação para curva BER vs Eb/N0. 

# Autor: Arthur Cadore
# Data: 8-09-2025
# """

import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm

from .datagram import Datagram
from .transmitter import Transmitter
from .receiver import Receiver
from .noise import NoiseEBN0
from .data import ExportData, ImportData

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
        if len(error_values) != len(EbN0_values):
            raise ValueError("error_values deve ter o mesmo tamanho que EbN0_values")

        self.fs = 128_000
        self.Rb = 400
        self.fc = 4000
        self.EbN0_values = EbN0_values
        self.num_workers = num_workers
        self.numblocks = numblocks
        self.max_repetitions = max_repetitions
        self.error_values = error_values
        self.datagramTX = Datagram(pcdnum=1234, numblocks=numblocks)
        self.bitsTX = self.datagramTX.streambits
        self.bitsSent = len(self.bitsTX)
        self.t, self.s = Transmitter(fc=self.fc, datagram=self.datagramTX, output_print=False, output_plot=False, fs=self.fs, Rb=self.Rb).run() 
        self.receiver = Receiver(fc=self.fc, output_print=False, output_plot=False, fs=self.fs, Rb=self.Rb) 

    def simulate(self, ebn0_db):
        add_noise = NoiseEBN0(ebn0_db, fs=self.fs, Rb=self.Rb)
        s_noisy = add_noise.add_noise(self.s)
        
        bitsRX = self.receiver.run(s_noisy, self.t)

        num_errors = sum(1 for tx, rx in zip(self.bitsTX, bitsRX) if tx != rx)
        return num_errors

    def run(self):
        # Lista para armazenar os valores de BER e Eb/N0
        ber_results = []

        # Paralelizar as simulações para cada Eb/N0 usando Pool de Workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Usa o tqdm para monitorar o progresso da simulação em cada iteração de Eb/N0
            for ebn0_db in self.EbN0_values:
                total_errors = 0
                repetitions = 0

                # Usa o tqdm para monitorar o progresso da simulação
                with tqdm(total=self.max_repetitions, desc=f"Simulando Eb/N0 = {ebn0_db} dB", ncols=100) as pbar:
                    # Realiza a simulação de forma paralela usando executor.map
                    while repetitions < self.max_repetitions and total_errors < self.error_values[int(ebn0_db)]:
                        futures = [executor.submit(self.simulate, ebn0_db) for _ in range(self.num_workers)]  # Cria múltiplas simulações

                        for future in futures:
                            num_errors = future.result()  # Aguarda a conclusão da tarefa
                            total_errors += num_errors
                            repetitions += 1
                            pbar.update(1)  # Atualiza o progresso do tqdm

                        # Se atingiu o limite de erros, interrompe a simulação
                        if total_errors >= self.error_values[int(ebn0_db)]:
                            break

                # Calcula o número total de bits transmitidos
                total_bits_transmitted = repetitions * self.bitsSent
                print(f"Total de bits transmitidos: {total_bits_transmitted}")
                print(f"Total de erros: {total_errors}")

                # Calcula a BER
                if total_bits_transmitted > 0:
                    ber = (total_errors + 1) / (total_bits_transmitted + 1)
                    print(f"BER: {ber}")
                else:
                    ber = 0

                # Armazena a tupla (Eb/N0, BER) na lista
                ber_results.append((ebn0_db, ber))

        # Retorna a lista de tuplas com (Eb/N0, BER)
        return ber_results


if __name__ == "__main__":
    EbN0_values = np.arange(0, 8, 0.5)

    ref_values = [20, 10, 2]
    ref_points = [0, 3, 10]

    error_values = interpolate(len(EbN0_values), ref_points, ref_values)

    # Imprime os valores de erro máximo para cada Eb/N0
    for ebn0, error in zip(EbN0_values, error_values):
        print(f"Eb/N0 = {ebn0} dB: {error} erros")

    bersnr_argos = BERSNR_ARGOS(EbN0_values=EbN0_values, error_values=error_values, num_workers=8, numblocks=1, max_repetitions=2000)
    results = bersnr_argos.run()
    
    ExportData(results, "bersnr_argos").save()

    bersnr_argos = ImportData("bersnr_argos").load()

    # extrair os valores de Eb/N0 e BER
    EbN0_values = bersnr_argos[:, 0]
    ber_values = bersnr_argos[:, 1]

    # Criar o gráfico
    plt.figure(figsize=(16, 9))
    plt.plot(EbN0_values, ber_values, marker='o', linestyle='-', color='b')

    # Adicionar título e rótulos aos eixos
    plt.title("Curva BER vs Eb/N0", fontsize=16)
    plt.xlabel("Eb/N0 (dB)", fontsize=14)
    plt.ylabel("Taxa de Erro de Bit (BER)", fontsize=14)

    # Configurar o grid para aparecer na escala logarítmica no eixo y
    plt.grid(True, which='both', axis='y', linestyle='--', color='gray')

    # Definir os limites do eixo y de 10^-10 até 10^-5
    plt.ylim(10**-5, 1)
    # Alterar escala do eixo y para ser logarítmica
    plt.yscale('log')

    # Exibir o gráfico
    plt.grid(True)
    plt.savefig("bersnr_argos.pdf")
