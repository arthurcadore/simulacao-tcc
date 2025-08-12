"""
Implementa um datagrama compatível com o padrão PPT-A3.

Referência:
    AS3-SP-516-274-CNES (seção 3.1.4)

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
import json
from plots import Plotter

class Datagram: 
    def __init__(self, pcdnum=1234, numblocks=2, streambits=None):
        r"""
        Inicializa uma instância do datagrama.
        
        Referência:
            AS3-SP-516-274-CNES (seção 3.1.4).

        Args:
            pcdnum (int): Número identificador da PCD. Necessário para o modo TX.
            numblocks (int): Quantidade de blocos de dados (1 a 8). Necessário para o modo TX.
            streambits (np.ndarray): Sequência de bits do datagrama. Necessário para o modo RX.
        
        Raises:
            ValueError: Se o número de blocos não estiver entre 1 e 8.
            ValueError: Se o número PCD não estiver entre 0 e 1048575 $(2^{20} - 1)$.
            ValueError: Se os parâmetros `pcdnum` e `numblocks` ou `streambits` não forem fornecidos.

        """

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
        r"""
        Gera os blocos de dados simulados (valores aleatórios), com base na quantidade especificada de blocos.

        Referência:
            AS3-SP-516-274-CNES (seção 3.1.4.2)

        Returns:
            blocks (np.ndarray): Vetor de bits representando os blocos de dados.
        """

        length = [24] + [32] * (self.numblocks - 1)
        total_length = sum(length)
        return np.random.randint(0, 2, size=total_length, dtype=np.uint8)

    def generate_pcdid(self):
        r"""
        Gera o campo PCD ID a partir do número PCD, incluindo um checksum de 8 bits.

        Referência:
            AS3-SP-516-274-CNES (seção 3.1.4.2)

        Returns:
            pcd_id (np.ndarray): Vetor de bits contendo o PCD ID e o checksum.
        """

        bin_str = format(self.pcdnum, '020b')
        pcd_bits = np.array([int(b) for b in bin_str], dtype=np.uint8)

        checksum_val = pcd_bits.sum() % 256
        checksum_bits = np.array([int(b) for b in format(checksum_val, '08b')], dtype=np.uint8)
        return np.concatenate((pcd_bits, checksum_bits))

    def generate_msglength(self):
        r"""
        Gera o campo Message Length com base na quantidade de blocos e calcula o bit de paridade.

        Referência:
            AS3-SP-516-274-CNES (seção 3.1.4.2)

        Returns:
           msg_length (np.ndarray): Vetor de 4 bits representando o campo Message Length.
        """

        n = self.numblocks - 1
        bin_str = format(n, '03b')
        bits = np.array([int(b) for b in bin_str], dtype=np.uint8)
        paridade = bits.sum() % 2
        return np.append(bits, paridade)
    
    def generate_tail(self):
        r"""
        Gera o campo Tail (cauda), utilizado para esvaziar o registrador do codificador convolucional.

        Referência:
            AS3-SP-516-274-CNES (seção 3.1.4.3)

        Returns:
            tail (np.ndarray): Vetor de bits zerados com comprimento variável (7, 8 ou 9 bits).
        """

        tail_pad = [7, 8, 9]
        tail_length = tail_pad[(self.numblocks - 1) % 3]
        return np.zeros(tail_length, dtype=np.uint8)

    def parse_datagram(self):
        r"""
        Faz o parsing da sequência de bits do datagrama, extraindo campos e validando integridade.
        
        Returns:
            str (json): Objeto JSON contendo a representação estruturada do datagrama.
        
        Raises:
            ValueError: Caso haja falha nas validações de paridade de Message Length.
            ValueError: Caso haja falha no checksum do campo PCD ID.
            ValueError: Se a sequência de bits não for válida ou não puder ser convertida para json.

        Examples:

                >>> datagram = Datagram(streambits=bits)
                >>> print(datagram.parse_datagram())
                    {
                      "msglength": 2,
                      "pcdid": 1234,
                      "data": {
                        "bloco_1": {
                          "sensor_1": 42,
                          "sensor_2": 147,
                          "sensor_3": 75
                        },
                        "bloco_2": {
                          "sensor_1": 138,
                          "sensor_2": 7,
                          "sensor_3": 134,
                          "sensor_4": 182
                        }
                      },
                      "tail": 8
                    }
        """

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
            "data": {},
            "tail": tail_length
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