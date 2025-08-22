import os
import numpy as np

class ExportData:
    r"""
    Classe para armazenar o resultado da transmissão, incluindo o sinal modulado e o vetor de tempo.
    Pode salvar um único vetor ou múltiplos vetores em um único arquivo.

    Args:
        vector (Union[np.ndarray, List[np.ndarray]]): Um único vetor ou lista de vetores para salvar.
        filename (str): Nome do arquivo de saída.
        path (str): Caminho do diretório de saída.
    """
    def __init__(self, vector, filename, path="../out"):
        # Converte um único vetor para uma lista com um elemento
        self.vectors = [vector] if isinstance(vector, np.ndarray) else list(vector)
        self.filename = filename
        self.path = path

    def save(self, binary=True):
        r"""
        Salva os resultados em arquivo binário (.npy) ou em TXT.
        
        Args:
            binary (bool): Se True, salva em formato binário (.npy).
                          Se False, salva em formato de texto (.txt).
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        basepath = os.path.normpath(os.path.join(script_dir, self.path, self.filename))
        os.makedirs(os.path.dirname(basepath), exist_ok=True)

        if binary:
            # Salva em formato binário do NumPy
            # Se houver apenas um vetor, salva como array 1D, senão como array 2D
            data = self.vectors[0] if len(self.vectors) == 1 else np.array(self.vectors)
            np.save(f"{basepath}.npy", data)
        else:
            # Salva em texto (menos eficiente, mas legível)
            with open(f"{basepath}.txt", "w") as f:
                for i, vec in enumerate(self.vectors):
                    if i > 0:
                        f.write("\n--- Vector {} ---\n".format(i+1))
                    f.write(" ".join(map(str, vec)))

class ImportData:
    r"""
    Classe para carregar um vetor salvo em arquivo.

    Args:
        filename (str): Nome do arquivo (sem extensão).
        path (str): Caminho do diretório de entrada.
    """
    def __init__(self, filename, path="../out"):
        self.filename = filename
        self.path = path

    def load(self, mode="npy", dtype=np.float64):
        r"""
        Carrega o vetor salvo.

        Args:
            mode (str): Formato do arquivo:
                - "npy" : arquivo binário com metadados do NumPy (.npy)
                - "bin" : dados crus em binário (.bin)
                - "txt" : arquivo de texto (.txt)
            dtype (np.dtype): Tipo de dado (usado apenas para "bin").

        Returns:
            np.ndarray: Vetor carregado.
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
            raise ValueError(f"Formato '{mode}' não suportado. Use 'npy', 'bin' ou 'txt'.")
