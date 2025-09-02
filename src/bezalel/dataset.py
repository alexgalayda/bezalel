from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from loguru import logger as loguru_logger
from Bio import Align

from bezalel.utils import root_dir
from bezalel.nearest_search import Qdrant, QdrantConfig, get_general_embedding
from bezalel.embeddings import Embedding, EmbeddingModel
from bezalel.utils import load_json


def make_main_features(E: torch.Tensor, prior: torch.Tensor | None = None):
    # E: [B, L, d], prior: [L, L]
    B, L, d = E.shape
    Ei = E.unsqueeze(2).expand(B, L, L, d)   # [B,L,L,d]
    Ej = E.unsqueeze(1).expand(B, L, L, d)   # [B,L,L,d]
    prod = Ei * Ej
    if prior is not None:
        feat = torch.cat([Ei, Ej, prod], dim=-1)  # [B,L,L,3d]
    else:
        feat = torch.cat([Ei, Ej, prod, prior], dim=-1)  # [B,L,L,3d+1]
    return feat


def make_pair_features(E: torch.Tensor, D: torch.Tensor, prior: torch.Tensor):
    feat = make_main_features(E, prior)
    B, L, d = D.shape
    Di = D.unsqueeze(2).expand(B, L, L, d)
    Dj = D.unsqueeze(1).expand(B, L, L, d)
    prod = Di * Dj
    feat = torch.cat([prod, feat], dim=-1)
    return feat


class ProteinDataloaderConfig(BaseModel):
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


class ProteinDatasetConfig(BaseModel):
    preload: bool = False
    init_qdrant: bool = True
    processed_dir: Path = root_dir() / "data" / "processed"
    database_dir: Path = root_dir() / "data" / "processed"
    qdrant_config: QdrantConfig = QdrantConfig()
    embedding_model_config: EmbeddingModel = EmbeddingModel(cuda=True)


class Protein:
    def __init__(self, id: str, name: str, sequence: str, adjacency_matrix: np.ndarray | None = None, embedding: torch.Tensor | None = None):
        self.id = id
        self.name = name
        self.sequence = sequence
        self.adjacency_matrix = np.array(adjacency_matrix) if adjacency_matrix is not None else None
        self.embedding = torch.tensor(embedding) if embedding is not None else None

    def __str__(self):
        return f"Protein(id={self.id},\nname={self.name},\nsequence={self.sequence},\nadjacency_matrix={self.adjacency_matrix.shape},\nembedding={self.embedding.shape})"

    def __repr__(self):
        return self.__str__()


def align_proteins(protein: str, neighbor: str):
    aligner = Align.PairwiseAligner()
    alignments = aligner.align(protein, neighbor)
    # alignment = min(alignments, key=lambda x: len(x[0]))
    alignment = alignments[0]
    return alignment[0], alignment[1]


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, config: ProteinDatasetConfig | None = None, logger = None):
        self.config = config or ProteinDatasetConfig()
        self.processed_dir = self.config.processed_dir
        self.database_dir = self.config.database_dir
        self.logger = logger or loguru_logger
        self.data = []
        self._init_qdrant(self.config.qdrant_config)
        self._init_embedding_model(self.config.embedding_model_config)
        self._load_data()

    def _init_qdrant(self, config: QdrantConfig):
        if self.config.init_qdrant:
            self.qdrant = Qdrant(config)

    def _init_embedding_model(self, config: EmbeddingModel):
        self.embedding_model = Embedding(config)

    def _get_nearest_neighbors(self, protein: Protein, limit: int = 3):
        embedding = self.embedding_model(protein.sequence).cpu().numpy()
        embedding = get_general_embedding(embedding)
        nearest_neighbors = self.qdrant.search(embedding, limit=limit)
        for neighbor in nearest_neighbors:
            if neighbor["id"] == protein.id:
                continue
            neighbor = Protein(
                name=neighbor["name"],
                **load_json(self.database_dir / f"{neighbor['name']}.json")
            )
            break
        else:
            raise ValueError("No nearest neighbors found")
        return neighbor

    def process_protein(self, sequence: str, protein: Protein):
        adjacency_matrix_full = np.eye(len(sequence), dtype=int)
        adjacency_matrix = np.array(protein.adjacency_matrix)
        delta_i = 0
        for i, c in enumerate(sequence):
            delta_j = 0
            if c == "-":
                delta_i += 1
                continue
            for j, d in enumerate(sequence):
                if d == "-":
                    delta_j += 1
                    continue
                adjacency_matrix_full[i, j] = adjacency_matrix[i- delta_i, j- delta_j]
        return Protein(
            id=protein.id,
            name=protein.name,
            sequence=sequence,
            adjacency_matrix=adjacency_matrix_full
        )

    def _align_neighbors(self, protein: Protein, neighbor: Protein):
        protein_seq, neighbor_seq = align_proteins(protein.sequence, neighbor.sequence)
        protein_full = self.process_protein(protein_seq, protein)
        neighbor_full = self.process_protein(neighbor_seq, neighbor)
        protein_full.embedding = self.embedding_model(protein_seq)
        neighbor_full.embedding = self.embedding_model(neighbor_seq)
        return protein_full, neighbor_full

    def _process_data(self, file: Path):
        protein = Protein(name=file.stem, **load_json(file))
        neighbor = self._get_nearest_neighbors(protein)
        protein, neighbor = self._align_neighbors(protein, neighbor)
        feat = make_pair_features(protein.embedding, neighbor.embedding, neighbor.adjacency_matrix)
        target = torch.tensor(protein.adjacency_matrix).to(protein.embedding.device)
        return feat.cpu(), target.cpu().unsqueeze(0)

    def _load_data(self):
        if self.config.preload:
            for file in self.processed_dir.glob("*.json"):
                self.data.append(self._process_data(file))
        else:
            self.data = list(self.processed_dir.glob("*.json"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.config.preload:
            return self.data[index]
        else:
            return self._process_data(self.data[index])

