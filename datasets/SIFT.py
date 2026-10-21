import os
import urllib.request
import tarfile
import numpy as np
from typing import Tuple

class SIFT:
    """
    Classe para gerenciar o download, carregamento e pré-processamento do dataset SIFT.
    """
    def __init__(self, n_vectors: int = 10000, dim: int = 128, data_dir: str = "./data"):
        self.n_vectors = n_vectors
        self.dim = dim
        self.data_dir = data_dir
        self.vectors = None
        
        # Cria o diretório de dados se não existir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Executa o pipeline de download e pré-processamento
        self._download_and_preprocess()

    def _download_and_preprocess(self) -> None:
        """Baixa o dataset SIFT do repositório padrão e extrai os vetores."""
        sift_url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        tar_path = os.path.join(self.data_dir, "sift.tar.gz")
        extracted_folder = os.path.join(self.data_dir, "sift")
        fvecs_path = os.path.join(extracted_folder, "sift_base.fvecs")

        if not os.path.exists(fvecs_path):
            if not os.path.exists(tar_path):
                print(f"Baixando o dataset SIFT de {sift_url}...")
                try:
                    urllib.request.urlretrieve(sift_url, tar_path)
                except Exception as e:
                    print(f"Aviso no download via FTP ({e}). Gerando dados sintéticos para fallback.")
                    self._generate_synthetic_data()
                    return

            print("Extraindo o arquivo compactado...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.data_dir)

        if os.path.exists(fvecs_path):
            print(f"Processando e carregando {self.n_vectors} vetores de {self.dim} dimensões...")
            self.vectors = self._read_fvecs(fvecs_path, self.n_vectors, self.dim)
        else:
            self._generate_synthetic_data()

    def _read_fvecs(self, filename: str, max_vectors: int, expected_dim: int) -> np.ndarray:
        """Lê o formato binário padrão .fvecs do SIFT."""
        vectors = []
        with open(filename, "rb") as f:
            count = 0
            while count < max_vectors:
                dim_data = f.read(4)
                if not dim_data:
                    break
                d = np.frombuffer(dim_data, dtype=np.int32)[0]
                v = np.frombuffer(f.read(4 * d), dtype=np.float32)
                
                # Garante que respeita a dimensão especificada
                if d >= expected_dim:
                    vectors.append(v[:expected_dim])
                    count += 1
        return np.array(vectors, dtype=np.float32)

    def _generate_synthetic_data(self) -> None:
        """Gera dados sintéticos caso o download falhe."""
        print("Gerando dados sintéticos estruturados do SIFT...")
        rng = np.random.default_rng(42)
        self.vectors = rng.standard_normal(size=(self.n_vectors, self.dim), dtype=np.float32)

    def get_shape(self) -> Tuple[int, int]:
        """Retorna o formato atual dos dados (número de vetores, dimensões)."""
        return self.vectors.shape