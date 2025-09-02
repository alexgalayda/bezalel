from pathlib import Path

from tqdm.auto import tqdm
from pydantic import BaseModel
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from loguru import logger as loguru_logger

from bezalel.utils import load_json


class QdrantConfig(BaseModel):
    storage_path: str = "./qdrant_storage"
    collection: str = "embeddings"
    vector_size: int = 320
    with_vectors: bool = True


class Qdrant:
    def __init__(self, config: QdrantConfig, logger = None):
        self.config = config
        self.client = QdrantClient(path=config.storage_path)
        self.logger = logger or loguru_logger

    def build_collection(self):
        self.client.create_collection(
            collection_name=self.config.collection,
            vectors_config=VectorParams(size=self.config.vector_size, distance=Distance.COSINE),
        )

    def upload_collection(self, embeddings: np.ndarray, ids: list[int], payload: list[dict] | None = None):
        self.client.upload_collection(
            collection_name=self.config.collection,
            vectors=embeddings,
            ids=ids,
            payload=payload
        )

    def search(self, query: np.ndarray, limit: int = 10):
        points = self.client.query_points(
            collection_name=self.config.collection,
            query=query,
            limit=limit,
            with_vectors=self.config.with_vectors
        ).points
        points = [
            {
                "id": point.id, 
                "vector": torch.tensor(point.vector) if point.vector is not None else None,
                "name": point.payload["name"]
            } for point in points]
        return points


def get_general_embedding(emb: torch.Tensor) -> torch.Tensor:
    return emb.squeeze().mean(axis=0, keepdims=False)


def create_db(qdrant_config: QdrantConfig, process_dir: Path, embedding_dir: Path):
    qdrant = Qdrant(qdrant_config)
    qdrant.build_collection()
    ids, embeddings, payloads = [], [], []
    processed_files = list(Path(process_dir).glob("*.json"))
    for file in tqdm(processed_files, desc="Processing embeddings files", total=len(processed_files)):
        data = load_json(file)
        embedding = torch.load(embedding_dir / f"{file.stem}.pt")
        embeddings.append(get_general_embedding(embedding).cpu().numpy())
        ids.append(data["id"])
        payloads.append({"name": file.stem})
    qdrant.upload_collection(embeddings, ids, payloads)
    qdrant.logger.info(f"Uploaded {len(embeddings)} embeddings to Qdrant")

if __name__ == "__main__":
    config = QdrantConfig(
        storage_path="./qdrant_storage",
        collection="embeddings",
    )
    create_db(config, Path("data/processed"), Path("data/embeddings"))
    # qdrant = Qdrant(config)
    # search_result = qdrant.search(
    #     get_general_embedding(torch.load(Path("data/embeddings") / "1A75_A.pt")).cpu().numpy(),
    #     limit=5
    # )
    # print(len(search_result))
    # print(search_result[0]["vector"].shape)
