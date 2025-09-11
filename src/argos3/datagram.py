"""
Implementa um datagrama compatível com o padrão PPT-A3.

Referência:
    AS3-SP-516-274-CNES (seção 3.1.4)

Autor: Arthur Cadore
Data: 28-07-2025
"""

import numpy as np
import json
from .plotter import BitsPlot, create_figure, save_figure

class Datagram: 
    def __init__(self, pcdnum=None, numblocks=None, streambits=None, seed=None, payload=None):
        r"""
        Gera um datagrama no padrão ARGOS-3. O formato do datagrama é ilustrado na figura abaixo.

        ![pageplot](../assets/datagrama.svg)

        Args:
            pcdnum (int): Número identificador da PCD. Necessário para o modo TX.
            numblocks (int): Quantidade de blocos de dados. Necessário para o modo TX.
            streambits (np.ndarray): Sequência de bits do datagrama. Necessário para o modo RX.
            seed (int): Seed do gerador de números aleatórios.
            payload (np.ndarray): Payload do datagrama. 

        Raises:
            ValueError: Se o número de blocos não estiver entre 1 e 8.
            ValueError: Se o número PCD não estiver entre 0 e 1048575 $(2^{20} - 1)$.
            ValueError: Se os parâmetros `pcdnum` e `numblocks` ou `streambits` não forem fornecidos.
            ValueError: Se o payload não for fornecido ou se o comprimento do payload não for o mesmo que o número de blocos.

        Exemplo: 
            ![pageplot](assets/example_datagram_time.svg)

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.1.4.2)
        </div>
        """

        # construtor TX
        if pcdnum is not None and numblocks is not None:
            if not (1 <= numblocks <= 8):
                raise ValueError("O número de blocos deve estar entre 1 e 8.")
            if not (0 <= pcdnum <= 1048575):  # 2^20 - 1
                raise ValueError("O número PCD deve estar entre 0 e 1048575.")
            if (payload is not None) and (len(payload) != (numblocks -1) * 32 + 24):
                raise ValueError("O payload deve ter o mesmo comprimento que o número de blocos.")

            self.pcdnum = pcdnum
            self.numblocks = numblocks
            self.rng = np.random.default_rng(seed)

            if payload is not None:
                self.blocks = payload
            else:
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
        
        A quantidade de blocos pode variar de 1 á 8. O primeiro bloco tem comprimento de 24bits, enquanto que todos os demais blocos tem 32bits. Dessa forma, os dados o comprimento dos dados de aplicação são dados pela expressão abaixo.

        $$
        L_{app} = 24 + 32 \cdot (n-1)
        $$

        Sendo: 
            - $L_{app}$: Comprimento do datagrama em bits 
            - $n$: Número de blocos do datagrama, podendo variar de 1 á 8. 

        Returns:
            blocks (np.ndarray): Vetor de bits representando os blocos de dados.

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.1.4.2)
        </div>
        """

        length = [24] + [32] * (self.numblocks - 1)
        total_length = sum(length)
        return self.rng.integers(0, 2, size=total_length, dtype=np.uint8)

    def generate_pcdid(self):
        r"""
        Gera o campo $PCD_{ID}$ a partir do número PCD ($PCD_{num}$), Primeiro gera-se a sequência de 20 bits correspondente ao número PCD.

        $$
          PCDnum_{10} \mapsto PCDnum_{2}  
        $$

        Sendo: 
            - $PCDnum_{10}$: Valor decimal do campo $PCD_{num}$, podendo variar de 0 á 1048575 $(2^{20} - 1)$.
            - $PCDnum_{2}$: Sequência de 20 bits correspondente ao valor de $PCD_{num}$.

        Em seguida, é calculado o checksum, $R_{PCD}$, do campo $PCD_{num}$, obtido através da soma dos bits e aplicação da operação módulo 256 ($2^8$).

        $$
        \begin{equation}
        R_{PCD} = \left( \sum_{i=0}^{19} b_i \cdot 2^i \right) \bmod 256
        \end{equation}
        $$

        Sendo: 
            - $R_{PCD}$: Sequência de 8 bits correspondente ao checksum do campo $PCD_{num}$.
            - $i$: Indice do bit do campo $PCD_{num}$.
            - $b$: Valor do bit do campo $PCD_{num}$.

        O campo $PCD_{ID}$ é gerado concatenando os parâmetros gerados, sendo $PCD_{ID} = PCD_{num} \oplus R_{PCD}$.

        Returns:
            pcd_id (np.ndarray): Vetor de bits contendo o PCD ID e o checksum.       

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.1.4.2)
        </div>
        """

        bin_str = format(self.pcdnum, '020b')
        pcd_bits = np.array([int(b) for b in bin_str], dtype=np.uint8)

        checksum_val = pcd_bits.sum() % 256
        checksum_bits = np.array([int(b) for b in format(checksum_val, '08b')], dtype=np.uint8)
        return np.concatenate((pcd_bits, checksum_bits))

    def generate_msglength(self):
        r"""
        Gera o valor do comprimento de mensagem $T_{m}$ com base na quantidade de blocos $n$. Primeiro, deve-se calcular a sequência de bits $B_m$. 
        
         $$
           Bm_{10} = (n - 1) \mapsto Bm_{2} 
         $$

        Sendo: 
            - $B_m$: Sequência de três bits correspondendo ao tamanho de mensagem. 
            - $n$: Número de blocos do datagrama, podendo variar de 1 á 8. 

        Em seguida, é calculado o quarto bit $P_m$ (bit de paridade).

        $$
        \begin{equation}
            P_m = 
            \begin{cases}
            1, & \text{se } \left[ \sum_{i=0}^{B_m} b_i = 0 \right]\mod 2  \\
            0, & \text{se } \left[ \sum_{i=0}^{B_m} b_i = 1 \right]\mod 2 
            \end{cases} \text{.}
        \end{equation}
        $$
        
        Sendo: 
            - $P_m$: Bit de paridade.
            - $i$: Indice de bit do campo $B_m$.

        O campo $T_{m}$ é gerado concatenando os parâmetros gerados, sendo $T_{m} = B_{m} \oplus P_{m}$.

        Returns:
           msg_length (np.ndarray): Vetor de 4 bits representando o campo Message Length.

        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.1.4.2)
        </div>
        """

        n = self.numblocks - 1
        bin_str = format(n, '03b')
        bits = np.array([int(b) for b in bin_str], dtype=np.uint8)
        paridade = bits.sum() % 2
        return np.append(bits, paridade)
    
    def generate_tail(self):
        r"""
        Gera a cauda do datagrama $E_m$, utilizado para limpar o registrador do codificador convolucional.

        $$
        E_m = 7 + [(n - 1) \bmod 3]
        $$

        Sendo: 
            - $E_m$: Comprimento de cauda do datagrama (zeros) adicionada ao final do datagrama. 
            - $n$: Número de blocos do datagrama.

        Returns:
            tail (np.ndarray): Vetor de bits zerados com comprimento variável (7, 8 ou 9 bits).
            
        <div class="referencia">
        <b>Referência:</b><br>
        AS3-SP-516-274-CNES (seção 3.1.4.3)
        </div>
        """

        tail_pad = [7, 8, 9]
        tail_length = tail_pad[(self.numblocks - 1) % 3]
        return np.zeros(tail_length, dtype=np.uint8)

    def parse_datagram(self):
        r"""
        Interpreta a sequência de bits do datagrama, extraindo campos e validando integridade.
        
        Returns:
            str (json): Objeto JSON contendo a representação estruturada do datagrama.
        
        Raises:
            ValueError: Caso haja falha na validação de paridade do comprimento de mensagem $T_m$.
            ValueError: Caso haja falha no checksum do campo $PCD_{ID}$. 
            ValueError: Se a sequência de bits de aplicação não corresponder ao comprimento de $T_m$.

        Exemplo:
            ```python
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
            ```
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
    datagram_tx = Datagram(pcdnum=123456, numblocks=2, seed=10)
    print(datagram_tx.parse_datagram())
    print("Stream bits: ", ''.join(str(b) for b in datagram_tx.streambits))

    fig_datagram, grid = create_figure(1, 1, figsize=(16, 5))
    
    BitsPlot(
        fig_datagram, grid, (0, 0),
        bits_list=[datagram_tx.msglength, 
                   datagram_tx.pcdid, 
                   datagram_tx.blocks, 
                   datagram_tx.tail],
        sections=[("Message Length", len(datagram_tx.msglength)),
                  ("PCD ID", len(datagram_tx.pcdid)),
                  ("Dados de App.", len(datagram_tx.blocks)),
                  ("Tail", len(datagram_tx.tail))],
        colors=["green", "orange", "red", "blue"]
    ).plot(xlabel="Index de Bit")

    fig_datagram.tight_layout()
    save_figure(fig_datagram, "example_datagram_time.pdf")

    # Receptor
    bits = datagram_tx.streambits

    print("\n\nReceptor: ")
    datagram_rx = Datagram(streambits=bits)
    print(datagram_rx.parse_datagram())
    print("Stream bits: ", ''.join(str(b) for b in datagram_rx.streambits))


    # Teste com payload:

    # Gera um vetor com 24 uns
    payload = np.ones(24, dtype=np.uint8)

    datagram_tx = Datagram(pcdnum=123456, numblocks=1, payload=payload, seed=10)
    print(datagram_tx.parse_datagram())
    print("Stream bits: ", ''.join(str(b) for b in datagram_tx.streambits))

    bits = datagram_tx.streambits

    print("\n\nReceptor: ")
    datagram_rx = Datagram(streambits=bits)
    print(datagram_rx.parse_datagram())
    print("Stream bits: ", ''.join(str(b) for b in datagram_rx.streambits))
