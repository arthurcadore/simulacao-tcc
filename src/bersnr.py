import numpy as np
import concurrent.futures
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from datagram import Datagram
from transmitter import Transmitter
from receiver import Receiver
from noise import Noise

def simulate_ber(snr_db, numblocks=8, fs=128_000, Rb=400):
    r"""
    Simula o BER para um dado SNR em dB.
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

if __name__ == "__main__":
    r"""
    Simulação de BER em função do SNR.
    """
    SNR_values = np.arange(-30, 0, 0.5)  # -30 até 0 dB
    repetitions = 16666
    numblocks = 8
    num_workers = 24  

    results = []
    total_tasks = len(SNR_values) * repetitions

    ber_accumulator = {snr: [] for snr in SNR_values}

    # Executa simulações em paralelo
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

    # Salva resultados em TXT
    with open("snr_vs_ber.txt", "w") as f:
        f.write("SNR_dB\tBER\n")
        for snr, ber in results:
            f.write(f"{snr}\t{ber:.8e}\n")

    # Gera gráfico final único
    snr_list, ber_list = zip(*results)
    fig, ax = plt.subplots()
    ax.semilogy(snr_list, ber_list, marker='o')
    ax.grid(True, which="both", ls="--")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("SNR vs BER")
    plt.savefig("snr_vs_ber.png", dpi=300)
    plt.close(fig)
