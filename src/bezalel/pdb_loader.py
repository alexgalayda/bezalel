from enum import Enum
from pathlib import Path
from uuid import uuid4

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Model import Model as BioModel
from Bio.PDB.Residue import Residue as BioResidue
from loguru import logger as loguru_logger
from pydantic import BaseModel
from tqdm.auto import tqdm

from bezalel.pdb_convert import map_residue
from bezalel.utils import save_json


class SeveralModelsTactic(Enum):
    TAKE_FIRST = "take_first"
    TAKE_ALL = "take_all"


class SeveralChainsTactic(Enum):
    TAKE_FIRST = "take_first"
    TAKE_ALL = "take_all"


class ResiduesTactic(Enum):
    IGNORE = "ignore"
    FILL_MISSING = "fill_missing"


class Residue(BaseModel):
    three: str
    resname: str | None = None
    coord: tuple[float, float, float] | None = None

    @classmethod
    def from_residue(cls, residue: BioResidue):
        resname = map_residue(residue.get_resname())
        if resname is None:
            item = cls(three=residue.get_resname())
        else:
            item = cls(
                three=residue.get_resname(),
                resname=resname,
            )
        coord = [float(f) for f in residue["CA"].get_coord()] if "CA" in residue else None
        item.coord = coord
        return item


class PDBParserConfig(BaseModel):
    quiet: bool = False
    several_models_tactic: SeveralModelsTactic = SeveralModelsTactic.TAKE_FIRST
    several_chains_tactic: SeveralChainsTactic = SeveralChainsTactic.TAKE_ALL
    residues_tactic: ResiduesTactic = ResiduesTactic.IGNORE
    threshold: float = 8.0


class PDBBuilder:
    def __init__(self, config: PDBParserConfig, logger = None):
        self.config = config
        self.logger = logger or loguru_logger
        self._init_parser(config)

    def _init_parser(self, config: PDBParserConfig):
        self.parser = PDBParser(QUIET=config.quiet)
        self.builder = PPBuilder()

    def get_model(self, path: Path) -> BioModel | list[BioModel]:
        structure = self.parser.get_structure("structure", path)
        models = [model for model in structure]
        match self.config.several_models_tactic:
            case SeveralModelsTactic.TAKE_FIRST:
                return models[0]
            case SeveralModelsTactic.TAKE_ALL:
                return models
            case _:
                raise ValueError(f"Invalid several models tactic: {self.config.several_models_tactic}")

    def get_chains(self, model: BioModel) -> dict[str, BioChain]:
        chains = [chain for chain in model]
        match self.config.several_chains_tactic:
            case SeveralChainsTactic.TAKE_FIRST:
                return {chains[0].get_id(): chains[0]}
            case SeveralChainsTactic.TAKE_ALL:
                return {chain.get_id(): chain for chain in chains}
            case _:
                raise ValueError(f"Invalid several chains tactic: {self.config.several_chains_tactic}")

    def get_residues(self, chain: BioChain) -> list[Residue]:
        residues = []
        for residue in chain:
            residue = Residue.from_residue(residue)
            match self.config.residues_tactic:
                case ResiduesTactic.IGNORE:
                    if residue.resname is not None:
                        residues.append(residue)
                case ResiduesTactic.FILL_MISSING:
                    if residue.resname is None:
                        residue.resname = "X"
                    residues.append(residue)
                case _:
                    raise ValueError(f"Invalid residues tactic: {self.config.residues_tactic}")
        return residues

    def parse(self, path: Path):
        model = self.get_model(path)
        chains = self.get_chains(model)
        residues = {}
        for chain_id, chain in chains.items():
            residues[chain_id] = self.get_residues(chain)
        return residues

    def create_adjacency_matrix(self, chain: list[Residue], threshold: float = 8.0) -> np.ndarray:
        adjacency_matrix = np.eye(len(chain), dtype=int)
        for i in range(len(chain)):
            residue_i = chain[i]
            if residue_i.coord is None:
                continue
            for j in range(i + 1, len(chain)):
                residue_j = chain[j]
                if residue_j.coord is None:
                    continue
                distance = np.linalg.norm(np.array(residue_i.coord) - np.array(residue_j.coord))
                if distance < threshold:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
        return adjacency_matrix

    def create_sequence(self, chain: list[Residue]) -> str:
        sequence = ""
        for residue in chain:
            if residue.resname is not None:
                sequence += residue.resname
        return sequence

    def __call__(self, path: Path, save_dir: Path | None = None) -> list[dict]:
        residues = self.parse(path)
        chains = []
        for chain_id, chain in residues.items():
            chain = {
                "id": str(uuid4()),
                "sequence": self.create_sequence(chain),
                "adjacency_matrix": self.create_adjacency_matrix(chain).tolist(),
            }
            if chain["sequence"] == "":
                continue
            if save_dir is not None:
                save_json(chain, save_dir / f"{path.stem}_{chain_id}.json")
            chains.append(chain)
        return chains

    def process_pdb(self, path: Path, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        for pdb_file in tqdm(path.glob("*.pdb"), desc="Processing PDB files", total=len(list(path.glob("*.pdb")))):
            self(pdb_file, save_dir=save_dir)


if __name__ == "__main__":
    config = PDBParserConfig(
        several_models_tactic=SeveralModelsTactic.TAKE_FIRST,
        several_chains_tactic=SeveralChainsTactic.TAKE_ALL,
        residues_tactic=ResiduesTactic.IGNORE,
        quiet=True)
    pp_builder = PDBBuilder(config)
    pp_builder.process_pdb(Path("data/train"), save_dir=Path("data/processed"))
    config = PDBParserConfig(
        several_models_tactic=SeveralModelsTactic.TAKE_FIRST,
        several_chains_tactic=SeveralChainsTactic.TAKE_ALL,
        residues_tactic=ResiduesTactic.IGNORE,
        quiet=True)
    pp_builder = PDBBuilder(config)
    pp_builder.process_pdb(Path("data/test"), save_dir=Path("data/processed_test"))
