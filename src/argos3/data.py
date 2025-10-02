# """
# Import and Export of data to numpy files.
#
# Author: Arthur Cadore
# Date: 28-07-2025
# """

import os
import numpy as np

class ExportData:
    r"""
    Instantiates an `ExportData` object, used to save vectors in binary `.npy` or text `.txt` files.

    Args:
        vector (Union[np.ndarray, List[np.ndarray]]): A single vector or list of vectors to save.
        filename (str): Output file name.
        path (str): Output directory path.
    """
    def __init__(self, vector, filename, path="../../out"):
        # Converts a single vector to a list with one element
        self.vectors = [vector] if isinstance(vector, np.ndarray) else list(vector)
        self.filename = filename
        self.path = path

    def save(self, binary=True):
        r"""
        Saves the results in a binary `.npy` or text `.txt` file.
        
        Args:
            binary (bool): If `True`, saves in binary `.npy` format.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        basepath = os.path.normpath(os.path.join(script_dir, self.path, self.filename))
        os.makedirs(os.path.dirname(basepath), exist_ok=True)

        if binary:
            # Saves in NumPy binary format
            # If there is only one vector, saves as 1D array, otherwise as 2D array
            data = self.vectors[0] if len(self.vectors) == 1 else np.array(self.vectors)
            np.save(f"{basepath}.npy", data)
        else:
            # Saves in text (less efficient, but readable)
            with open(f"{basepath}.txt", "w") as f:
                for i, vec in enumerate(self.vectors):
                    if i > 0:
                        f.write("\n--- Vector {} ---\n".format(i+1))
                    f.write(" ".join(map(str, vec)))

class ImportData:
    r"""
    Instantiates an `ImportData` object, used to load vectors from binary `.npy` or text `.txt` files.

    Args:
        filename (str): Input file name (without extension).
        path (str): Input directory path.
    """
    def __init__(self, filename, path="../../out"):
        self.filename = filename
        self.path = path

    def load(self, mode="npy", dtype=np.float64):
        r"""
        Loads the saved vector.

        Args:
            mode (str): File format: `npy` for binary files, `txt` for text files.
            dtype (np.dtype): Encoding type used, necessary for `npy`.

        Returns:
            data (np.ndarray): Loaded vector.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        basepath = os.path.normpath(os.path.join(script_dir, self.path, self.filename))

        if mode == "npy":
            return np.load(f"{basepath}.npy")

        elif mode == "bin":
            return np.fromfile(f"{basepath}.bin", dtype=dtype)

        elif mode == "txt":
            with open(f"{basepath}.txt", "r") as f:
                data = list(map(float, f.read().split()))
            return np.array(data, dtype=dtype)

        else:
            raise ValueError(f"Format '{mode}' not supported. Use 'npy', 'bin' or 'txt'.")
