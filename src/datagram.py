import numpy as np
import json
from plots import Plotter

class Datagram: 
    def __init__(self, pcdnum=None, numblocks=None, streambits=None):

        # construtor TX
        if pcdnum is not None and numblocks is not None:
            if not (1 <= numblocks <= 8):
                raise ValueError("O número de blocos deve estar entre 1 e 8.")
            if not (0 <= pcdnum <= 1048575):  # 2^20 - 1
                raise ValueError("O número PCD deve estar entre 0 e 1048575.")

            self.pcdnum = pcdnum
            self.numblocks = numblocks
            self.blocks = self.generate_blocks()
            self.pcdid = self.generate_pcdid()
            self.tail = self.generate_tail()
            self.msglength = self.generate_msglength()
            self.streambits = np.concatenate((self.msglength, self.pcdid, self.blocks, self.tail))
            self.blocks_json = self.parse_datagram()
        
        # construtor RX  
        elif streambits is not None:
            self.streambits = streambits
            self.blocks_json = self.parse_datagram()
        else:
            raise ValueError("Você deve fornecer ou (pcdnum e numblocks) ou streambits")

    def generate_blocks(self):
        length = [24] + [32] * (self.numblocks - 1)
        total_length = sum(length)
        return np.random.randint(0, 2, size=total_length, dtype=np.uint8)

    def generate_pcdid(self):
        bin_str = format(self.pcdnum, '020b')
        pcd_bits = np.array([int(b) for b in bin_str], dtype=np.uint8)

        checksum_val = pcd_bits.sum() % 256
        checksum_bits = np.array([int(b) for b in format(checksum_val, '08b')], dtype=np.uint8)
        return np.concatenate((pcd_bits, checksum_bits))

    def generate_msglength(self):
        n = self.numblocks - 1
        bin_str = format(n, '03b')
        bits = np.array([int(b) for b in bin_str], dtype=np.uint8)
        paridade = bits.sum() % 2
        return np.append(bits, paridade)
    
    def generate_tail(self):
        tail_pad = [7, 8, 9]
        tail_length = tail_pad[(self.numblocks - 1) % 3]
        return np.zeros(tail_length, dtype=np.uint8)

    def parse_datagram(self):
        msglength_bits = self.streambits[:4]
        value_bits = msglength_bits[:3]
        paridade_bit = msglength_bits[3]

        if paridade_bit != value_bits.sum() % 2:
            raise ValueError("Paridade inválida no campo Message Length.")

        numblocks = int("".join(map(str, value_bits)), 2) + 1
        pcdid_bits = self.streambits[4:32]

        pcdnum_bits = pcdid_bits[:20]
        checksum_bits = pcdid_bits[20:28]
        checksum_val = pcdnum_bits.sum() % 256
        if checksum_val != int("".join(map(str, checksum_bits)), 2):
            raise ValueError("Checksum inválido no campo PCD ID.")
        pcdnum = int("".join(map(str, pcdnum_bits)), 2)
        
        data_bits = self.streambits[32:32 + 24 + (32 * (numblocks - 1))]
        tail_bits = self.streambits[32 + 24 + (32 * (numblocks - 1)):]
        tail_length = len(tail_bits)
        data = {
            "msglength": numblocks,
            "pcdid": pcdnum,
            "tail": tail_length,
            "data": {}
        }

        index = 0
        for bloco in range(numblocks):
            bloco_nome = f"bloco_{bloco+1}"
            data["data"][bloco_nome] = {}
            
            num_sensores = 3 if bloco == 0 else 4
            for sensor in range(num_sensores):
                sensor_nome = f"sensor_{sensor+1}"
                sensor_bits = data_bits[index:index+8]
                sensor_valor = int("".join(map(str, sensor_bits)), 2)
                data["data"][bloco_nome][sensor_nome] = sensor_valor
                index += 8

        return json.dumps(data, indent=2)

if __name__ == "__main__":

    print("\n\nTransmissor:")
    datagram_tx = Datagram(pcdnum=123456, numblocks=2)
    print(datagram_tx.parse_datagram())
    print("Stream bits: ", ''.join(str(b) for b in datagram_tx.streambits))

    plotter = Plotter()
    plotter.plot_bits([datagram_tx.msglength, 
                       datagram_tx.pcdid, 
                       datagram_tx.blocks, 
                       datagram_tx.tail],
                       sections=[("Message Length", len(datagram_tx.msglength)),
                                 ("PCD ID", len(datagram_tx.pcdid)),
                                 ("Dados de App.", len(datagram_tx.blocks)),
                                 ("Tail", len(datagram_tx.tail))],
                       colors=["green", "orange", "red", "blue"],
                       save_path="../out/example_datagram.pdf")


    # Receptor
    bits = datagram_tx.streambits

    print("\n\nReceptor: ")
    datagram_rx = Datagram(streambits=bits)
    print(datagram_rx.parse_datagram())
    print("Stream bits: ", ''.join(str(b) for b in datagram_rx.streambits))