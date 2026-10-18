"""
adr.models.mlp – MLP trên Morgan fingerprint.

Architecture:
  fingerprint [B, 2048]
  → Linear(2048, 256) → ReLU → Dropout(p)
  → Linear(256, 27)   → logits [B, 27]

Không có sigmoid trong forward (BCEWithLogitsLoss xử lý).
Inference: sigmoid áp sau khi gọi model ở eval mode.

Yêu cầu: torch
"""

from __future__ import annotations

import sys

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def _require_torch() -> None:
    if not _TORCH_OK:
        raise ImportError(
            "PyTorch chưa được cài.\n"
            "  CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  Hoặc: conda install pytorch cpuonly -c pytorch"
        )


if _TORCH_OK:
    class MLP(nn.Module):
        """
        MLP đơn giản: fingerprint → logits.

        Args:
            input_dim: Số chiều fingerprint (2048).
            hidden_dim: Số nơ-ron ẩn (256).
            n_labels: Số nhãn đầu ra (27).
            dropout: Tỷ lệ dropout (0.2).
        """

        def __init__(
            self,
            input_dim: int = 2048,
            hidden_dim: int = 256,
            n_labels: int = 27,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, n_labels),
            )
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_labels = n_labels
            self.dropout_p = dropout

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: float tensor [B, input_dim]
            Returns:
                logits: float tensor [B, n_labels]  (không sigmoid)
            """
            return self.net(x)

        def get_config(self) -> dict:
            return {
                "type": "mlp",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "n_labels": self.n_labels,
                "dropout": self.dropout_p,
            }

        def save(self, run_dir) -> None:
            """Lưu state_dict và config vào run_dir/model.pt."""
            import json
            from pathlib import Path
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": self.state_dict(), "config": self.get_config()},
                run_dir / "model.pt",
            )

        @classmethod
        def load(cls, run_dir) -> "MLP":
            """Load từ run_dir/model.pt."""
            from pathlib import Path
            run_dir = Path(run_dir)
            pt_path = run_dir / "model.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"model.pt không tìm thấy: {pt_path}")
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            cfg = data["config"]
            model = cls(
                input_dim=cfg["input_dim"],
                hidden_dim=cfg["hidden_dim"],
                n_labels=cfg["n_labels"],
                dropout=cfg["dropout"],
            )
            model.load_state_dict(data["state_dict"])
            return model

else:
    # Stub khi không có torch
    class MLP:  # type: ignore
        def __init__(self, *args, **kwargs):
            _require_torch()
