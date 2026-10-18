"""
adr.models.cnn – CNN một nhánh trên character-tokenized SMILES.

Architecture:
  input_ids [B, L]  (int64)
  → Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
  → [B, L, embedding_dim]
  → transpose → [B, embedding_dim, L]
  → Conv1d(embedding_dim, channels, kernel_size) → ReLU
  → [B, channels, L']
  → Masked mean pooling (dùng attention_mask)
  → [B, channels]
  → Dropout → Linear(channels, n_labels)
  → logits [B, 27]

Đảm bảo:
  - PAD (padding_idx=0) không đóng góp vào embedding.
  - Masked pooling: activation ở vùng PAD không ảnh hưởng đến output.
  - CNN output bất biến khi chỉ thêm PAD cuối chuỗi (kiểm tra bằng test).
  - Không sigmoid trong forward.

Yêu cầu: torch
"""

from __future__ import annotations

import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from adr.schemas import PAD_TOKEN_ID


def _require_torch() -> None:
    if not _TORCH_OK:
        raise ImportError(
            "PyTorch chưa được cài.\n"
            "  CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


if _TORCH_OK:
    class SmilesCharCNN(nn.Module):
        """
        CNN một nhánh trên ký tự SMILES.

        Args:
            vocab_size: Kích thước vocabulary (từ CharTokenizer).
            embedding_dim: Chiều embedding (64).
            channels: Số kênh conv (128).
            kernel_size: Kích thước kernel (5).
            n_labels: Số nhãn (27).
            dropout: Tỷ lệ dropout (0.2).
        """

        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 64,
            channels: int = 128,
            kernel_size: int = 5,
            n_labels: int = 27,
            dropout: float = 0.2,
            pad_token_id: int = PAD_TOKEN_ID,
        ) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.channels = channels
            self.kernel_size = kernel_size
            self.n_labels = n_labels
            self.dropout_p = dropout
            self.pad_token_id = pad_token_id

            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=pad_token_id
            )
            self.conv = nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # same padding
            )
            self.dropout = nn.Dropout(p=dropout)
            self.head = nn.Linear(channels, n_labels)

        def forward(
            self,
            input_ids: "torch.Tensor",
            attention_mask: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            """
            Args:
                input_ids: int64 [B, L]
                attention_mask: bool [B, L] – True nơi token hợp lệ.
                                Nếu None, coi tất cả là hợp lệ.
            Returns:
                logits: float [B, n_labels]  (không sigmoid)
            """
            # Embedding: [B, L, E]
            emb = self.embedding(input_ids)

            # Conv1d cần [B, C, L]: transpose
            x = emb.transpose(1, 2)  # [B, E, L]
            x = F.relu(self.conv(x))  # [B, channels, L']

            # Masked mean pooling
            # x: [B, channels, L']  attention_mask: [B, L]
            L_prime = x.shape[2]
            if attention_mask is not None:
                # Resize mask nếu same-padding làm L' = L
                mask = attention_mask[:, :L_prime].float()  # [B, L']
                # Broadcast: [B, 1, L'] nhân với [B, channels, L']
                mask_expanded = mask.unsqueeze(1)  # [B, 1, L']
                x = x * mask_expanded
                # Sum / count hợp lệ
                pool = x.sum(dim=2) / (mask_expanded.sum(dim=2).clamp(min=1))  # [B, channels]
            else:
                pool = x.mean(dim=2)  # [B, channels]

            pool = self.dropout(pool)
            logits = self.head(pool)  # [B, n_labels]
            return logits

        def get_config(self) -> dict:
            return {
                "type": "cnn",
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "n_labels": self.n_labels,
                "dropout": self.dropout_p,
                "pad_token_id": self.pad_token_id,
            }

        def save(self, run_dir) -> None:
            """Lưu state_dict và config vào run_dir/model.pt."""
            from pathlib import Path
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": self.state_dict(), "config": self.get_config()},
                run_dir / "model.pt",
            )

        @classmethod
        def load(cls, run_dir) -> "SmilesCharCNN":
            """Load từ run_dir/model.pt."""
            from pathlib import Path
            run_dir = Path(run_dir)
            pt_path = run_dir / "model.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"model.pt không tìm thấy: {pt_path}")
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            cfg = data["config"]
            model = cls(
                vocab_size=cfg["vocab_size"],
                embedding_dim=cfg["embedding_dim"],
                channels=cfg["channels"],
                kernel_size=cfg["kernel_size"],
                n_labels=cfg["n_labels"],
                dropout=cfg["dropout"],
                pad_token_id=cfg.get("pad_token_id", PAD_TOKEN_ID),
            )
            model.load_state_dict(data["state_dict"])
            return model

else:
    class SmilesCharCNN:  # type: ignore
        def __init__(self, *args, **kwargs):
            _require_torch()
