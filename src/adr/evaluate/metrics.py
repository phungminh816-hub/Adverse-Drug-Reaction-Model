"""Tính AP, ROC-AUC, precision, recall và F1 trên nhãn quan sát được."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


@dataclass
class LabelMetrics:
    """Metrics cho một nhãn."""
    label_name: str
    label_idx: int
    n_positive: int
    n_negative: int
    n_missing: int
    ap: float | None
    roc_auc: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    valid: bool
    reason: str = ""


@dataclass
class EvalResult:
    """Kết quả đánh giá toàn bộ model trên một split."""
    split: str
    model_id: str
    threshold: float
    per_label: list[LabelMetrics]
    macro_ap: float | None
    macro_roc_auc: float | None
    micro_precision: float | None
    micro_recall: float | None
    micro_f1: float | None
    n_valid_ap: int
    n_valid_roc_auc: int
    n_excluded: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False, default=str)


def evaluate(
    scores: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    label_names: list[str],
    threshold: float = 0.5,
    split: str = "val",
    model_id: str = "unknown",
) -> EvalResult:
    """
    Tính metric phân loại đa nhãn tại các vị trí mask=True.

    Args:
        scores: Sigmoid scores [N, 27] ∈ [0, 1].
        targets: Binary labels [N, 27].
        mask: Boolean mask [N, 27] – True nếu nhãn quan sát được.
        label_names: 27 tên nhãn theo thứ tự cột.
        threshold: Ngưỡng phân loại (mặc định 0.5).
        split: Tên split ("train"|"val"|"test").
        model_id: ID run artifact.

    Returns:
        EvalResult với metrics đầy đủ.
    """
    scores, targets, mask = np.asarray(scores), np.asarray(targets), np.asarray(mask)
    n_labels = len(label_names)
    if scores.ndim != 2 or scores.shape != targets.shape or mask.shape != targets.shape or scores.shape[1] != n_labels:
        raise ValueError("Scores, nhãn và mask phải cùng kích thước [N, len(label_names)].")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask chỉ được chứa 0 hoặc 1.")
    mask = mask.astype(bool)
    if not np.isin(targets[mask], [0, 1]).all():
        raise ValueError("Nhãn quan sát được phải là 0 hoặc 1.")
    observed_scores = scores[mask]
    if not np.isfinite(observed_scores).all() or np.any((observed_scores < 0) | (observed_scores > 1)):
        raise ValueError("Scores quan sát được phải hữu hạn và nằm trong [0, 1].")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold phải nằm trong [0, 1].")

    per_label: list[LabelMetrics] = []
    valid_aps: list[float] = []
    valid_roc_aucs: list[float] = []
    warnings: list[str] = []

    preds = (scores >= threshold).astype(int)

    all_targets_obs = []
    all_preds_obs = []

    for j in range(n_labels):
        obs = mask[:, j]
        n_obs = obs.sum()
        n_missing = (~obs).sum()
        n_pos = int(targets[obs, j].sum()) if n_obs > 0 else 0
        n_neg = int(n_obs) - n_pos

        if n_obs > 0:
            all_targets_obs.extend(targets[obs, j].tolist())
            all_preds_obs.extend(preds[obs, j].tolist())

        if n_obs == 0:
            lm = LabelMetrics(
                label_name=label_names[j], label_idx=j,
                n_positive=0, n_negative=0, n_missing=int(n_missing),
                ap=None, roc_auc=None, precision=None, recall=None, f1=None,
                valid=False, reason="Không có nhãn quan sát trong split này"
            )
            per_label.append(lm)
            continue

        ap = None
        ap_reason = ""
        if n_pos == 0:
            ap_reason = "Không có positive trong split (AP không xác định)"
        else:
            try:
                ap = float(average_precision_score(targets[obs, j], scores[obs, j]))
            except Exception as e:
                ap_reason = f"average_precision_score lỗi: {e}"

        roc = None
        roc_reason = ""
        if n_pos == 0 or n_neg == 0:
            roc_reason = f"Chỉ một lớp trong split (pos={n_pos}, neg={n_neg})"
        else:
            try:
                roc = float(roc_auc_score(targets[obs, j], scores[obs, j]))
            except Exception as e:
                roc_reason = f"roc_auc_score lỗi: {e}"

        prec = rec = f1 = None
        try:
            prec = float(precision_score(targets[obs, j], preds[obs, j], zero_division=0))
            rec = float(recall_score(targets[obs, j], preds[obs, j], zero_division=0))
            f1 = float(f1_score(targets[obs, j], preds[obs, j], zero_division=0))
        except Exception as e:
            warnings.append(f"Nhãn '{label_names[j]}': precision/recall/f1 lỗi: {e}")

        valid = True
        reason = ""
        if ap is None:
            reason += ap_reason + " | "
        if roc is None:
            reason += roc_reason

        lm = LabelMetrics(
            label_name=label_names[j], label_idx=j,
            n_positive=n_pos, n_negative=n_neg, n_missing=int(n_missing),
            ap=ap, roc_auc=roc,
            precision=prec, recall=rec, f1=f1,
            valid=valid, reason=reason.strip(" |"),
        )
        per_label.append(lm)

        if ap is not None:
            valid_aps.append(ap)
        if roc is not None:
            valid_roc_aucs.append(roc)

    macro_ap = float(np.mean(valid_aps)) if valid_aps else None
    macro_roc_auc = float(np.mean(valid_roc_aucs)) if valid_roc_aucs else None

    micro_prec = micro_rec = micro_f1 = None
    if all_targets_obs:
        try:
            micro_prec = float(precision_score(all_targets_obs, all_preds_obs, zero_division=0))
            micro_rec = float(recall_score(all_targets_obs, all_preds_obs, zero_division=0))
            micro_f1 = float(f1_score(all_targets_obs, all_preds_obs, zero_division=0))
        except Exception as e:
            warnings.append(f"Micro metrics lỗi: {e}")

    n_excluded = sum(1 for lm in per_label if not lm.valid)

    return EvalResult(
        split=split,
        model_id=model_id,
        threshold=threshold,
        per_label=per_label,
        macro_ap=macro_ap,
        macro_roc_auc=macro_roc_auc,
        micro_precision=micro_prec,
        micro_recall=micro_rec,
        micro_f1=micro_f1,
        n_valid_ap=len(valid_aps),
        n_valid_roc_auc=len(valid_roc_aucs),
        n_excluded=n_excluded,
        warnings=warnings,
    )


def print_eval_summary(result: EvalResult) -> None:
    """In tóm tắt kết quả đánh giá."""
    print(f"\n{'='*60}")
    print(f"Evaluation: {result.model_id}  split={result.split}  threshold={result.threshold}")
    print(f"{'='*60}")
    print(f"  Macro AP      : {result.macro_ap:.4f}" if result.macro_ap is not None else "  Macro AP      : N/A")
    print(f"  Macro ROC-AUC : {result.macro_roc_auc:.4f}" if result.macro_roc_auc is not None else "  Macro ROC-AUC : N/A")
    print(f"  Micro Precision: {result.micro_precision:.4f}" if result.micro_precision is not None else "  Micro Precision: N/A")
    print(f"  Micro Recall   : {result.micro_recall:.4f}" if result.micro_recall is not None else "  Micro Recall   : N/A")
    print(f"  Micro F1       : {result.micro_f1:.4f}" if result.micro_f1 is not None else "  Micro F1       : N/A")
    print(f"  Nhãn hợp lệ AP: {result.n_valid_ap}/{len(result.per_label)}")
    print(f"  Nhãn loại     : {result.n_excluded}")
    if result.warnings:
        for w in result.warnings:
            print(f"  ! {w}")
    print()

    print(f"  {'Nhãn':<50} {'AP':>7} {'ROC-AUC':>8} {'F1':>6} {'N+':>5}")
    print(f"  {'-'*80}")
    for lm in result.per_label:
        ap_str = f"{lm.ap:.3f}" if lm.ap is not None else " N/A "
        roc_str = f"{lm.roc_auc:.3f}" if lm.roc_auc is not None else "  N/A "
        f1_str = f"{lm.f1:.3f}" if lm.f1 is not None else " N/A"
        print(f"  {lm.label_name[:50]:<50} {ap_str:>7} {roc_str:>8} {f1_str:>6} {lm.n_positive:>5}")
