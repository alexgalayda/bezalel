from enum import Enum
from pathlib import Path

import torch
from pydantic import BaseModel
from tqdm.auto import tqdm
from transformers import EsmModel, EsmTokenizer

from bezalel.utils import load_json


class ModelName(Enum):
    ESM2_T48_15B_UR50D = "facebook/esm2_t48_15B_UR50D"
    ESM2_T36_3B_UR50D = "facebook/esm2_t36_3B_UR50D"
    ESM2_T33_650M_UR50D = "facebook/esm2_t33_650M_UR50D"
    ESM2_T30_150M_UR50D = "facebook/esm2_t30_150M_UR50D"
    ESM2_T12_35M_UR50D = "facebook/esm2_t12_35M_UR50D"
    ESM2_T6_8M_UR50D = "facebook/esm2_t6_8M_UR50D"


class EmbeddingModel(BaseModel):
    model_name: ModelName = ModelName.ESM2_T6_8M_UR50D
    cuda: bool = False


class Embedding:
    def __init__(self, config: EmbeddingModel):
        self.config = config
        self.cuda = config.cuda
        self._init_model(config.model_name, self.cuda)

    def _init_model(self, model_name: ModelName, cuda: bool):
        self.tokenizer = EsmTokenizer.from_pretrained(model_name.value)
        self.model = EsmModel.from_pretrained(model_name.value)
        if cuda:
            self.model.to(torch.device("cuda"))

    def __call__(self, sequence: str) -> torch.FloatTensor:
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        if self.cuda:
            inputs = inputs.to(torch.device("cuda"))
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        embeddings = embeddings[:, 1:-1, :]
        return embeddings


def create_embeddings(model: EmbeddingModel, process_dir: str, save_dir: Path | None = None):
    process_dir.mkdir(parents=True, exist_ok=True)
    if save_dir is None:
        save_dir = Path(process_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for file in tqdm(Path(process_dir).glob("*.json"), desc="Processing JSON files", total=len(list(Path(process_dir).glob("*.json")))):
        data = load_json(file)
        embedding = model(data["sequence"]).cpu()
        torch.save(embedding, save_dir / f"{file.stem}.pt")


if __name__ == "__main__":
    embedding_model = Embedding(
        EmbeddingModel(
            model_name=ModelName.ESM2_T6_8M_UR50D,
            cuda=True
        )
    )
    create_embeddings(embedding_model, Path("data/processed"), save_dir=Path("data/embeddings"))
