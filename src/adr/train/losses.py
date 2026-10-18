"""
adr.train.losses – Masked BCE loss cho multi-label với missing labels.

BCEWithLogitsLoss(reduction='none') → nhân mask → chuẩn hóa.
Nếu batch không có nhãn quan sát (toàn mask=0): raise RuntimeError
theo chính sách "báo lỗi" (không bỏ lặng).

Yêu cầu: torch
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def masked_bce_loss(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    mask: "torch.Tensor",
    empty_batch_policy: str = "error",
) -> "torch.Tensor":
    """
    Masked BCE with logits loss.

    Args:
        logits: float [B, L] – raw logits (không sigmoid).
        targets: float [B, L] – nhãn 0/1 (giá trị padding không quan trọng nếu mask=0).
        mask: bool hoặc float [B, L] – True/1 nếu nhãn quan sát được.
        empty_batch_policy:
          "error"  → raise RuntimeError nếu toàn batch không có nhãn quan sát.
          "zero"   → trả tensor(0.0) (log warning).

    Returns:
        Scalar loss tensor.
    """
    if not _TORCH_OK:
        raise ImportError("PyTorch chưa được cài.")

    mask = mask.float()
    element_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    # Chỉ tính loss tại vị trí mask=1
    masked_loss = element_loss * mask
    n_observed = mask.sum()

    if n_observed == 0:
        if empty_batch_policy == "error":
            raise RuntimeError(
                "Batch không có nhãn quan sát nào (toàn mask=0).\n"
                "  Kiểm tra DataLoader hoặc split. "
                "  Đặt empty_batch_policy='zero' để bỏ qua (không khuyến nghị)."
            )
        else:
            import warnings
            warnings.warn("Batch toàn mask=0, trả loss=0.", RuntimeWarning)
            return torch.tensor(0.0, requires_grad=True)

    return masked_loss.sum() / n_observed
