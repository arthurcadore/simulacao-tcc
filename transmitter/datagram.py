import numpy as np
import matplotlib.pyplot as plt
import json
import os
import scienceplots

# Estilo science
plt.style.use('science')
plt.rcParams["figure.figsize"] = (16, 9)
plt.rc('font', size=16)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=22)
plt.rc('legend', frameon=True, edgecolor='black', facecolor='white', fancybox=True, fontsize=12)

class Datagram: 
    def __init__(self, pcdnum, numblocks):
        if not (1 <= numblocks <= 8):
            raise ValueError("O número de blocos deve estar entre 1 e 8.")
        
        if not (0 <= pcdnum < 1048575):  # 2^20 - 1
            raise ValueError("O número PCD deve estar entre 0 e 1048575 (2^20 - 1).")

        self.pcdnum = pcdnum
        self.numblocks = numblocks
        self.blocks = self.generate_blocks()
        self.pcdid = self.calc_pcdid()
        self.tail = self.generate_tail()
        self.msglength = self.calc_msglength()
        self.bits = np.concatenate((self.msglength, self.pcdid, self.blocks, self.tail))
        self.blocks_json = self.parse_sensors_to_json()


    def generate_blocks(self):
        length = [24] + [32] * (self.numblocks - 1)
        total_length = sum(length)
        return np.random.randint(0, 2, size=total_length, dtype=np.uint8)

    def calc_pcdid(self):
        bin_str = format(self.pcdnum, '020b')
        pcd_bits = np.array([int(b) for b in bin_str], dtype=np.uint8)

        checksum_val = pcd_bits.sum() % 256
        checksum_bits = np.array([int(b) for b in format(checksum_val, '08b')], dtype=np.uint8)
        return np.concatenate((pcd_bits, checksum_bits))

    def calc_msglength(self):
        n = self.numblocks - 1
        bin_str = format(n, '03b')
        bits = np.array([int(b) for b in bin_str], dtype=np.uint8)
        paridade = bits.sum() % 2
        return np.append(bits, paridade)
    
    def generate_tail(self):
        tail_pad = [7, 8, 9]
        tail_length = tail_pad[(self.numblocks - 1) % 3]
        return np.zeros(tail_length, dtype=np.uint8)

    def parse_sensors_to_json(self):
        data = {}
        index = 0

        for i in range(self.numblocks):
            bloco_nome = f"bloco_{i+1}"

            if i == 0:
                bloco_len = 24
                num_sensores = 3
            else:
                bloco_len = 32
                num_sensores = 4

            bloco_bits = self.blocks[index:index+bloco_len]
            index += bloco_len

            sensor_len = bloco_len // num_sensores
            sensors = {}

            for j in range(num_sensores):
                s_ini = j * sensor_len
                if j == num_sensores - 1:
                    s_fim = bloco_len
                else:
                    s_fim = s_ini + sensor_len

                bits_list = bloco_bits[s_ini:s_fim].tolist()
                bin_str = "".join(str(b) for b in bits_list)
                sensors[f"sensor_{j+1}"] = int(bin_str, 2)

            data[bloco_nome] = sensors

        return json.dumps(data, indent=2)

    def plot_datagram(self, save_path=None):
        # Concatenação dos campos
        all_bits = np.concatenate((self.msglength, self.pcdid, self.blocks, self.tail))
        sections = [len(self.msglength), len(self.pcdid), len(self.blocks), len(self.tail)]
        section_names = ["Message Length", "PCD ID", "Blocks", "Tail"]
        section_colors = ["green", "orange", "red", "blue"]

        # Superamostragem para degrau
        bits_up = np.repeat(all_bits, 2)
        x = np.arange(len(bits_up))
        bit_edges = np.arange(0, len(bits_up) + 1, 2)

        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(0, len(bits_up))
        ax.set_ylim(-0.2, 1.4)
        # Grade bit a bit
        ax.grid(False)
        for pos in bit_edges:
            ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
        start_bit = 0
        legend_handles = []
        for i, (sec_len, sec_name, sec_color) in enumerate(zip(sections, section_names, section_colors)):
            bit_start = start_bit * 2
            bit_end = (start_bit + sec_len) * 2

            # Inclui o último ponto da seção anterior para manter continuidade
            if i > 0:
                bit_start -= 1

            x_sec = x[bit_start:bit_end]
            y_sec = bits_up[bit_start:bit_end]

            handle, = ax.step(x_sec, y_sec, where='post', color=sec_color, linewidth=3, label=sec_name)
            legend_handles.append(handle)

            start_bit += sec_len

        # Adiciona os valores dos bits
        for i, bit in enumerate(all_bits):
            ax.text(i * 2 + 1, 1.15, str(bit), ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_ylabel("Datagrama")
        ax.set_yticks([0, 1])
        ax.set_xlabel('Bits')
        leg = ax.legend(handles=legend_handles, loc='upper right', frameon=True, edgecolor='black', facecolor='white', fontsize=12, fancybox=True)

        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_alpha(1.0)

        # Eixo X com marcações de bits
        num_bits = len(all_bits)
        step = max(1, num_bits // 16)
        ax.set_xticks(np.arange(0, num_bits * 2, step * 2))
        ax.set_xticklabels(np.arange(0, num_bits, step))

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()


class Assembler:
    def __init__(self, datagrambits):
        self.datagrambits = datagrambits
        self.msg_length_bits = datagrambits[:4]
        self.pcdid_bits = datagrambits[4:32]
        self.numblocks = self.get_msglength()
        self.pcdnum = self.get_pcdid()
        self.databits = self.get_databits()
        self.tail = self.get_tail()
        self.sensor_data = self.parse_sensors_to_json()

    def get_msglength(self):
        # Extrai os 3 bits e calcula valor
        value_bits = self.msg_length_bits[:3]
        paridade_bit = self.msg_length_bits[3]
        valor = int("".join(map(str, value_bits)), 2)
        
        # Verifica paridade 
        if sum(value_bits) % 2 != paridade_bit:
            raise ValueError("Paridade inválida no campo de tamanho da mensagem.")

        return valor + 1 
    
    def get_pcdid(self):
        # Extrai os 20 bits do PCD ID
        pcdnum_bits = self.pcdid_bits[:20]
        checksum_bits = self.pcdid_bits[20:28]

        # Calcula o checksum
        checksum_val = sum(pcdnum_bits) % 256
        if checksum_val != int("".join(map(str, checksum_bits)), 2):
            raise ValueError("Checksum inválido no campo PCD ID.")
        
        # Converte bits para número inteiro
        pcdnum = int("".join(map(str, pcdnum_bits)), 2)
        return pcdnum

    def get_databits(self):
        # Extrai os bits de dados partindo do 32º bit até o final
        # 24 + (32 * (self.numblocks - 1)) = 24 + 32 * (n - 1)
        start = 32
        end = start + 24 + (32 * (self.numblocks - 1))
        return self.datagrambits[start:end]
    
    def get_tail(self):
        # Tail começa após o final da seção de dados
        data_end = 32 + len(self.databits)
        return self.datagrambits[data_end:]
    
    def parse_sensors_to_json(self):
        data = {}
        index = 0

        for i in range(self.numblocks):
            bloco_nome = f"bloco_{i+1}"

            if i == 0:
                bloco_len = 24
                num_sensores = 3
            else:
                bloco_len = 32
                num_sensores = 4

            bloco_bits = self.databits[index:index+bloco_len]
            index += bloco_len

            sensor_len = bloco_len // num_sensores
            sensors = {}

            for j in range(num_sensores):
                s_ini = j * sensor_len
                if j == num_sensores - 1:
                    s_fim = bloco_len
                else:
                    s_fim = s_ini + sensor_len

                bits_list = bloco_bits[s_ini:s_fim].tolist()
                bin_str = "".join(str(b) for b in bits_list)
                sensors[f"sensor_{j+1}"] = int(bin_str, 2)

            data[bloco_nome] = sensors
        return json.dumps(data, indent=2)

if __name__ == "__main__":
    datagram = Datagram(pcdnum=123456, numblocks=2)
    print("PCD ID Bits:", ''.join(str(b) for b in datagram.pcdid))
    print("Message Length Bits:", ''.join(str(b) for b in datagram.msglength))
    print("Blocks Bits:", ''.join(str(b) for b in datagram.blocks))
    print("Tail Bits:", ''.join(str(b) for b in datagram.tail))
    print(datagram.blocks_json)

    output_path = os.path.join("out", "example_datagram.pdf")
    datagram.plot_datagram(output_path)

    assambling = Assembler(datagram.bits)
    print("PCD Number (Decoded):", assambling.pcdnum)
    print("Message Length Bits (Decoded):", ''.join(str(b) for b in assambling.msg_length_bits))
    print("Data Bits:", ''.join(str(b) for b in assambling.databits))
    print("Tail Bits (Decoded):", ''.join(str(b) for b in assambling.tail))
    print(assambling.sensor_data)
    
    print("\nVerificação:")
    print("Número de blocos correto:", assambling.numblocks == datagram.numblocks)
    print("PCD Number correto:", assambling.pcdnum == datagram.pcdnum)
    print("Dados corretos:", np.array_equal(assambling.databits, datagram.blocks))
    print("Tail correto:", np.array_equal(assambling.tail, datagram.tail))
    print("Sensor data correto:", assambling.sensor_data == datagram.blocks_json)