from __future__ import annotations
import json
import pathlib
import random
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path("./data/iwildcam_full")  # base data directory
JPG_DIR = ROOT / "jpg"                     # where extractor put class folders
SPECIES_MAP = JPG_DIR / "species_map.json"  # optional mapping class_id -> species name
IMG_SIZE = (32, 32)

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def _load_species_map() -> List[str]:
    """
    Load species names sorted by class_id. If mapping file is missing,
    fall back to folder names.
    """
    if SPECIES_MAP.exists():
        mapping = json.loads(SPECIES_MAP.read_text())
        # ensure order by numeric key
        return [mapping[str(i)] for i in range(len(mapping))]
    # fallback: use folder names (0,1,...)
    dirs = [d.name for d in JPG_DIR.iterdir() if d.is_dir()]
    # sort by integer value
    dirs = sorted(dirs, key=lambda x: int(x))
    return dirs

# ─────────────────────────────────────────────────────────────────────
# DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────
def _prepare_dataset() -> Tuple[List[Tuple[pathlib.Path,int]], List[str]]:
    """
    Scan JPG_DIR for subfolders and build (filepath, label) pairs and class list.
    """
    species = _load_species_map()
    samples: List[Tuple[pathlib.Path,int]] = []
    for label, sp in enumerate(species):
        folder = JPG_DIR / str(label)
        if not folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {folder}")
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                samples.append((img_path, label))
    if not samples:
        raise RuntimeError(f"No images found under {JPG_DIR}")
    return samples, species

# ─────────────────────────────────────────────────────────────────────
# TORCH DATASET
# ─────────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

class BrazilTop10Dataset(Dataset):
    def __init__(self, samples: List[Tuple[pathlib.Path,int]]):
        self.samples = samples
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return _transform(img), label

# ─────────────────────────────────────────────────────────────────────
# PUBLIC DATA LOADER
# ─────────────────────────────────────────────────────────────────────
class DatasetLoader:
    def __init__(self, num_clients: int):
        if num_clients < 1:
            raise ValueError("num_clients must be >= 1")
        self.num_clients = num_clients

        samples, self.classes = _prepare_dataset()
        full_dataset = BrazilTop10Dataset(samples)

        # train/test split (80/20)
        total = len(full_dataset)
        n_train = int(0.8 * total)
        train_ds, test_ds = random_split(full_dataset, [n_train, total - n_train])
        self.testset = test_ds

        # split train IID
        idxs = np.random.permutation(len(train_ds))
        shards = np.array_split(idxs, num_clients)
        self.client_datasets = [Subset(train_ds, shard.tolist()) for shard in shards]

    def loader(self,
               client_id: int,
               batch_size: int = 32,
               shuffle: bool = False,
               num_workers: int = 0,
               pin_memory: bool = None) -> DataLoader:
        if not (0 <= client_id < self.num_clients):
            raise ValueError(f"client_id must be between 0 and {self.num_clients-1}")
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        return DataLoader(
            dataset=self.client_datasets[client_id],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def test_loader(self,
                    batch_size: int = 32,
                    shuffle: bool = False,
                    num_workers: int = 0) -> DataLoader:
        return DataLoader(
            dataset=self.testset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_client_data_size(self, client_id: int) -> int:
        if not (0 <= client_id < self.num_clients):
            raise ValueError(f"client_id must be between 0 and {self.num_clients-1}")
        return len(self.client_datasets[client_id])

    @property
    def num_classes(self) -> int:
        return len(self.classes)
