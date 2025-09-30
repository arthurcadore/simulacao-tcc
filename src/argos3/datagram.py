# """
# Implementation of a datagram compatible with the PPT-A3 standard.

# Author: Arthur Cadore (github.com/arthurcadore)
# Date: 28-07-2025
# """

import numpy as np
import json
from .plotter import BitsPlot, create_figure, save_figure

class Datagram: 
    def __init__(self, pcdnum=None, numblocks=None, streambits=None, seed=None, payload=None):
        r"""
        Generate a datagram in the ARGOS-3 standard. The datagram format is illustrated in the figure below.

        ![pageplot](../assets/datagram.svg)

        Args:
            pcdnum (int): PCD number. Required for TX mode.
            numblocks (int): Number of blocks. Required for TX mode.
            seed (int): Seed of the random number generator. Optional for TX mode.
            payload (np.ndarray): Payload of the datagram. Optional for TX mode.
            streambits (np.ndarray): Bitstream of the datagram. Required for RX mode.

        Raises:
            ValueError: If the number of blocks is not between 1 and 8.
            ValueError: If the PCD number is not between 0 and 1048575 $(2^{20} - 1)$.
            ValueError: If the parameters `pcdnum` and `numblocks` or `streambits` are not provided.
            ValueError: If the payload is not provided or if the length of the payload is not the same as the number of blocks.

        Example: 
            ![pageplot](assets/example_datagram_time.svg)

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (section 3.1.4.2)
        </div>
        """

        # Attributes
        self.streambits = None
        self.blocks_json = None
        
        # The constructor will be called depending on how the datagram is created (TX or RX)
        if pcdnum is not None and numblocks is not None and streambits is None:
            # TX constructor
            self._init_tx(pcdnum, numblocks, seed, payload)
        elif streambits is not None and pcdnum is None and numblocks is None:
            # RX constructor
            self._init_rx(streambits)
        else:
            raise ValueError("You must provide either (pcdnum and numblocks) or streambits")
    
    def _init_tx(self, pcdnum, numblocks, seed, payload):
        r"""
        TX constructor
        """

        if not (1 <= numblocks <= 8):
            raise ValueError("The number of blocks must be between 1 and 8.")
        if not (0 <= pcdnum <= 1048575):  # 2^20 - 1
            raise ValueError("The PCD number must be between 0 and 1048575.")
        if (payload is not None) and (len(payload) != (numblocks -1) * 32 + 24):
            raise ValueError("The payload must have the same length as the number of blocks.")
        
        self.pcdnum = pcdnum
        self.numblocks = numblocks
        self.rng = np.random.default_rng(seed)

        # If payload is not provided, generate blocks automatically
        if payload is not None:
            # calculate the number of blocks based on the length of the payload
            payload_blocks = (len(payload) + 24) // 32
            if not (1 <= payload_blocks <= 8):
                raise ValueError("The payload length should be between 24 and 248 bits.")

            if payload_blocks != numblocks:
                raise ValueError("The number of blocks must be the same as the number of blocks calculated from the payload.")
            
            self.blocks = payload
        else:
            self.blocks = self.generate_blocks()

        # Generate datagram components
        self.pcdid = self.generate_pcdid()
        self.tail = self.generate_tail()
        self.msglength = self.generate_msglength()

        # The datagram bitstream>
        self.streambits = np.concatenate((self.msglength, self.pcdid, self.blocks, self.tail))

        # Create the datagram JSON representation
        self.blocks_json = self.parse_datagram()

    def _init_rx(self, streambits):
        r"""
        RX constructor
        """

        self.streambits = streambits
        self.blocks_json = self.parse_datagram()

    def generate_blocks(self):
        r"""
        Generate simulated data blocks (random values), based on the specified number of blocks. 
        
        The number of blocks can be between 1 and 8. The first block has a length of 24 bits, and all other blocks have 32 bits. In this way, the length of the data application is given by the expression below.

        $$
        L_{app} = 24 + 32 \cdot (n-1)
        $$

        Where: 
            - $L_{app}$: Data application length in bits 
            - $n$: Number of blocks of the datagram, varying from 1 to 8. 

        Returns:
            blocks (np.ndarray): Bit array representing the data blocks.

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (section 3.1.4.2)
        </div>
        """

        length = [24] + [32] * (self.numblocks - 1)
        total_length = sum(length)
        return self.rng.integers(0, 2, size=total_length, dtype=np.uint8)

    def generate_pcdid(self):
        r"""
        Generate the PCD_ID field from the PCD number ($PCD_{num}$), First generate the sequence of 20 bits corresponding to the PCD number.

        $$
          PCDnum_{10} \mapsto PCDnum_{2}  
        $$

        Where: 
            - $PCDnum_{10}$: Decimal value of the $PCD_{num}$ field, varying from 0 to 1048575 $(2^{20} - 1)$.
            - $PCDnum_{2}$: Sequence of 20 bits corresponding to the value of $PCD_{num}$.

        Then, the checksum, $R_{PCD}$, of the $PCD_{num}$ field is calculated, obtained through the sum of the bits and application of the modulo 256 ($2^8$) operation.

        $$
        \begin{equation}
        R_{PCD} = \left( \sum_{i=0}^{19} b_i \cdot 2^i \right) \bmod 256
        \end{equation}
        $$

        Where: 
            - $R_{PCD}$: Sequence of 8 bits corresponding to the checksum of the $PCD_{num}$ field.
            - $i$: Index of the bit of the $PCD_{num}$ field.
            - $b$: Value of the bit of the $PCD_{num}$ field.

        The $PCD_{ID}$ field is generated by concatenating the generated parameters, being $PCD_{ID} = PCD_{num} \oplus R_{PCD}$.

        Returns:
            pcd_id (np.ndarray): Bit array containing the PCD ID and checksum.       

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (section 3.1.4.2)
        </div>
        """

        bin_str = format(self.pcdnum, '020b')
        pcd_bits = np.array([int(b) for b in bin_str], dtype=np.uint8)

        checksum_val = pcd_bits.sum() % 256
        checksum_bits = np.array([int(b) for b in format(checksum_val, '08b')], dtype=np.uint8)
        return np.concatenate((pcd_bits, checksum_bits))

    def generate_msglength(self):
        r"""
        Generate the value of the message length $T_{m}$ based on the number of blocks $n$. First, the sequence of bits $B_m$ must be calculated. 
         $$
           Bm_{10} = (n - 1) \mapsto Bm_{2} 
         $$

        Where: 
            - $B_m$: Sequence of three bits corresponding to the message length. 
            - $n$: Number of blocks of the datagram, varying from 1 to 8. 

        Then, the fourth bit $P_m$ (parity bit) is calculated.

        $$
        \begin{equation}
            P_m = 
            \begin{cases}
            1, & \text{se } \left[ \sum_{i=0}^{B_m} b_i = 0 \right]\mod 2  \\
            0, & \text{se } \left[ \sum_{i=0}^{B_m} b_i = 1 \right]\mod 2 
            \end{cases} \text{.}
        \end{equation}
        $$
        
        Where: 
            - $P_m$: Parity bit.
            - $i$: Index of bit of the $B_m$ field.

        The $T_{m}$ field is generated by concatenating the generated parameters, being $T_{m} = B_{m} \oplus P_{m}$.

        Returns:
           msg_length (np.ndarray): Bit array representing the Message Length field.

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (section 3.1.4.2)
        </div>
        """

        n = self.numblocks - 1
        bin_str = format(n, '03b')
        bits = np.array([int(b) for b in bin_str], dtype=np.uint8)
        paridade = bits.sum() % 2
        return np.append(bits, paridade)
    
    def generate_tail(self):
        r"""
        Generate the tail of the datagram $E_m$, used to clear the codifier's register.

        $$
        E_m = 7 + [(n - 1) \bmod 3]
        $$

        Where: 
            - $E_m$: Tail of the datagram (zeros) added to the end of the datagram. 
            - $n$: Number of blocks of the datagram.

        Returns:
            tail (np.ndarray): Bit array of zeros with variable length (7, 8 or 9 bits).
            
        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (section 3.1.4.3)
        </div>
        """

        tail_pad = [7, 8, 9]
        tail_length = tail_pad[(self.numblocks - 1) % 3]
        return np.zeros(tail_length, dtype=np.uint8)

    def parse_datagram(self):
        r"""
        Interprets the bit sequence of the datagram, extracting fields and validating integrity.
        
        Returns:
            str (json): JSON object containing the structured representation of the datagram.
        
        Raises:
            ValueError: If the parity check of the message length $T_m$ fails.
            ValueError: If the checksum of the $PCD_{ID}$ field fails.
            ValueError: If the application bit sequence does not correspond to the length of $T_m$.

        Example:
            ```python
            >>> datagram = Datagram(streambits=bits)
            >>> print(datagram.parse_datagram())
            {
              "msglength": 2,
              "pcdid": 1234,
              "data": {
                "block_1": {
                  "sensor_1": 42,
                  "sensor_2": 147,
                  "sensor_3": 75
                },
                "block_2": {
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

        # extract the message length field
        msglength = self.streambits[:4]
        value_bits = msglength[:3]
        paridade_bit = msglength[3]

        # Verify the integrity of the field
        if paridade_bit != value_bits.sum() % 2:
            raise ValueError("Parity check failed for the message length field.")
        else:
            self.msglength = msglength

        # extract the PCD ID field
        pcdid_bits = self.streambits[4:32]
        pcdnum_bits = pcdid_bits[:20]
        checksum_bits = pcdid_bits[20:28]

        # Verify the integrity of the field
        checksum_val = pcdnum_bits.sum() % 256
        if checksum_val != int("".join(map(str, checksum_bits)), 2):
            raise ValueError("Checksum check failed for the PCD ID field.")
        else:
            self.pcdid = pcdid_bits
            self.pcdnum = int("".join(map(str, pcdnum_bits)), 2)            

        
        # extract the application data field
        self.numblocks = int("".join(map(str, value_bits)), 2) + 1
        self.blocks = self.streambits[32:32 + 24 + (32 * (self.numblocks - 1))]

        # extract the final bits
        finalbits = self.streambits[32 + 24 + (32 * (self.numblocks - 1)):]

        # extract the tail
        tail_pad = [7, 8, 9]
        tail_length = tail_pad[(self.numblocks - 1) % 3]
        tail_bits = finalbits[:tail_length]

        # Verify the integrity of the tail, all bits must be 0.
        if any(int(b) != 0 for b in tail_bits):
            raise ValueError("Tail check failed.")
        else:
            self.tail = tail_bits
    
        # create the JSON object
        data = {
            "msglength": self.numblocks,
            "pcdid": self.pcdnum,
            "data": {},
            "tail": tail_length
        }

        # build the JSON object
        index = 0
        for bloco in range(self.numblocks):
            bloco_nome = f"block_{bloco+1}"
            data["data"][bloco_nome] = {}
            
            num_sensores = 3 if bloco == 0 else 4
            for sensor in range(num_sensores):
                sensor_nome = f"sensor_{sensor+1}"
                sensor_bits = self.blocks[index:index+8]
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
        colors=["green", "orange", "red", "blue"],
        xlabel="Index de Bit"
    ).plot()

    fig_datagram.tight_layout()
    save_figure(fig_datagram, "example_datagram_time.pdf")

    # Receptor
    bits = datagram_tx.streambits

    print("\n\nReceptor: ")
    datagram_rx = Datagram(streambits=bits)
    print(datagram_rx.parse_datagram())
    print("Stream bits: ", ''.join(str(b) for b in datagram_rx.streambits))


    # Teste com payload:
    numblocks = 3
    payload_length = (numblocks - 1) * 32 + 24

    # Gera um vetor com 24 uns
    payload = np.ones(payload_length, dtype=np.uint8)

    datagram_tx = Datagram(pcdnum=123456, numblocks=numblocks, payload=payload, seed=10)
    print(datagram_tx.parse_datagram())
    print("Stream bits: ", ''.join(str(b) for b in datagram_tx.streambits))

    bits = datagram_tx.streambits

    print("\n\nReceptor: ")
    datagram_rx = Datagram(streambits=bits)
    print(datagram_rx.parse_datagram())
    print("Stream bits: ", ''.join(str(b) for b in datagram_rx.streambits))
