"""Frozen index-CXR encoder trained only on D_pred patients."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from scpcp.config import DataConfig


class _CXRDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, paths: list[str], labels: Tensor, indices: Tensor) -> None:
        from torchvision.transforms import v2

        self.paths = paths
        self.labels = labels
        self.indices = indices.tolist()
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((224, 224), antialias=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        from PIL import Image

        row = self.indices[index]
        with Image.open(self.paths[row]) as image:
            return self.transform(image.convert("RGB")), self.labels[row]


class _DenseNetCXR(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        from torchvision.models import DenseNet121_Weights, densenet121

        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.projection = nn.Linear(1024, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 14)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.projection(self.backbone(images))
        return features, self.classifier(features)


def index_cxr_embeddings(
    paths: list[str],
    labels: Tensor,
    training_rows: Tensor,
    *,
    config: DataConfig,
    device: str | torch.device,
    seed: int,
) -> Tensor:
    """Fine-tune DenseNet-121 on D_pred labels, freeze, and encode every index CXR.

    No COT, certification, or evaluation patient is used to train the encoder.
    """

    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing index CXR image: {missing[0]}")
    resolved = torch.device(device)
    torch.manual_seed(seed)
    model = _DenseNetCXR(config.cxr_embedding_dim).to(resolved)
    train_dataset = _CXRDataset(paths, labels, training_rows.cpu())
    loader = DataLoader(train_dataset, batch_size=config.cxr_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    for _ in range(config.cxr_epochs):
        for images, targets in loader:
            _, logits = model(images.to(resolved, non_blocking=True))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets.to(resolved, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    all_rows = torch.arange(len(paths))
    encode_loader = DataLoader(
        _CXRDataset(paths, labels, all_rows),
        batch_size=config.cxr_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    pieces = []
    with torch.no_grad():
        for images, _ in encode_loader:
            features, _ = model(images.to(resolved, non_blocking=True))
            pieces.append(features.cpu())
    return torch.cat(pieces)
