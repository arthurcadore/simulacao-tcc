# """
# Implementation of a datagram compatible with the PPT-A3 standard.

# Author: Arthur Cadore (github.com/arthurcadore)
# Date: 28-07-2025
# """

import numpy as np
import json
from .plotter import BitsPlot, create_figure, save_figure
from .env_vars import *

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

        Examples:
            >>> import argos3
            >>> datagramTX = argos3.Datagram(pcdnum=123456, numblocks=2, seed=10)
            >>> streambits = datagramTX.streambits
            >>> print(streambits)
            [0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 1 0 0 1 0 0 0 0 0 0 1 0 1 0 0 1 1 0
             1 1 0 1 0 0 1 1 1 1 1 1 1 0 1 1 1 0 0 0 0 1 1 0 0 1 0 1 1 1 0 0 0 1 1 1 0
             1 1 0 1 1 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0]
            >>> 
            >>> datagramRX = argos3.Datagram(streambits=streambits)
            >>> print(datagramRX.parse_datagram())
            {
              "msglength": 2,
              "pcdid": 123456,
              "payload": {
                "block_1": {
                  "byte_1": 183,
                  "byte_2": 108,
                  "byte_3": 125
                },
                "block_2": {
                  "byte_1": 112,
                  "byte_2": 217,
                  "byte_3": 78,
                  "byte_4": 141
                }
              },
              "tail": 8
            }

            - Bitstream Plot Example: ![pageplot](assets/example_datagram_time.svg)

        <div class="referencia">
        <b>Reference:</b><br>
        AS3-SP-516-274-CNES (section 3.1.4.2)
        </div>
        """

        # Attributes
        self.streambits = None
        self.parsed_json = None
        
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
            
            self.payload = payload
        else:
            self.payload = self.generate_blocks()

        # Generate datagram components
        self.pcdid = self.generate_pcdid()
        self.tail = self.generate_tail()
        self.msglength = self.generate_msglength()

        # The datagram bitstream
        self.streambits = np.concatenate((self.msglength, self.pcdid, self.payload, self.tail))

        # Create the datagram JSON representation
        self.parsed_json = self.parse_datagram()

    def _init_rx(self, streambits):
        r"""
        RX constructor
        """

        self.streambits = streambits

        # Create the datagram JSON representation
        self.parsed_json = self.parse_datagram()

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

        # Generate data blocks length
        l_app = sum([24] + [32] * (self.numblocks - 1))

        # Generate random payload
        payload = self.rng.integers(0, 2, size=l_app, dtype=np.uint8)
        
        return payload

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

        # Convert PCD number to binary
        pcdnum_b = np.array([int(b) for b in format(self.pcdnum, '020b')], dtype=np.uint8)

        # Calculate checksum
        rpcd = pcdnum_b.sum() % 256
        rpcd_b = np.array([int(b) for b in format(rpcd, '08b')], dtype=np.uint8)

        # Generate PCD_ID
        pcd_id = np.concatenate((pcdnum_b, rpcd_b))
        return pcd_id

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

        # Convert number of blocks to binary
        bm = np.array([int(b) for b in format((self.numblocks - 1), '03b')], dtype=np.uint8)

        # Calculate parity bit
        pm = bm.sum() % 2

        # Generate message length
        tm = np.append(bm, pm)

        return tm
    
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
        
        # The tail can vary between 7, 8 or 9 bits, based on the number of blocks (mod 3)
        em_dict = [7, 8, 9]

        # Calculate tail length
        em_length = em_dict[(self.numblocks - 1) % 3]

        # Generate tail
        em = np.zeros(em_length, dtype=np.uint8)

        return em

    def parse_datagram(self):
        r"""
        Interprets the bit sequence of the datagram, extracting fields and validating integrity.
        
        Returns:
            str (json): JSON object containing the structured representation of the datagram.
        
        Raises:
            ValueError: If the parity check of the message length $T_m$ fails.
            ValueError: If the checksum of the $PCD_{ID}$ field fails.
            ValueError: If the application bit sequence does not correspond to the length of $T_m$.
        """

        # extract the message length field
        tm = self.streambits[:4]
        bm = tm[:3]
        pm = tm[3]

        # calculate the number of blocks
        self.numblocks = int("".join(map(str, bm)), 2) + 1

        # Verify the integrity of the field
        if pm != bm.sum() % 2:
            raise ValueError("Parity check failed for the message length field.")
        else:
            self.msglength = tm

        # extract the PCD ID field
        pcd_id = self.streambits[4:32]
        pcd_num = pcd_id[:20]
        rpcd = pcd_id[20:28]

        # Verify the integrity of the field
        rpcd_prime = pcd_num.sum() % 256
        if rpcd_prime != int("".join(map(str, rpcd)), 2):
            raise ValueError("Checksum check failed for the PCD ID field.")
        else:
            self.pcdid = pcd_id
            self.pcdnum = int("".join(map(str, pcd_num)), 2)            

        # extract the payload, any validation is done in the receiver
        data_length = 24 + (32 * (self.numblocks - 1))
        self.payload = self.streambits[32:32 + data_length]

        # calculate the tail bits based on the number of blocks
        tail_pad = [7, 8, 9]
        tail_length = tail_pad[(self.numblocks - 1) % 3]

        # extract the tail bits
        self.tail = self.streambits[32 + data_length:32 + data_length + tail_length]

        # Verify the integrity of the tail, all bits must be 0.
        if any(int(b) != 0 for b in self.tail):
            raise ValueError("Tail check failed.")

        # create the JSON object
        data = {
            "msglength": self.numblocks,
            "pcdid": self.pcdnum,
            "payload": {},
            "tail": tail_length
        }

        # build the JSON object
        index = 0
        for block in range(self.numblocks):
            block_index = f"block_{block+1}"
            data["payload"][block_index] = {}
            
            # the first block has 24 bits, other 32.
            num_bytes = 3 if block == 0 else 4

            # add the payload to the JSON object
            for byte in range(num_bytes):
                byte_index = f"byte_{byte+1}"
                byte_data = self.payload[index:index+8]
                byte_parsed = int("".join(map(str, byte_data)), 2)
                data["payload"][block_index][byte_index] = byte_parsed
                index += 8

        datagram_json = json.dumps(data, indent=2)
        return datagram_json

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
                   datagram_tx.payload, 
                   datagram_tx.tail],
        sections=[("Message Length", len(datagram_tx.msglength)),
                  ("PCD ID", len(datagram_tx.pcdid)),
                  ("Payload", len(datagram_tx.payload)),
                  ("Tail", len(datagram_tx.tail))],
        colors=[COLOR_AUX1, COLOR_AUX2, COLOR_AUX3, COLOR_AUX4],
        xlabel=BITSTREAM_X,
        ylabel=BITSTREAM_Y,
        title=DATAGRAM_STREAM_TITLE,
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
