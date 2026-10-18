"""
adr.features.tokenizer – Character-level SMILES tokenizer cho CNN.

API:
  tok = CharTokenizer()
  tok.fit(train_smiles)              # fit vocabulary từ train
  ids, mask = tok.transform(smiles)  # encode + pad
  tok.save(path)
  tok2 = CharTokenizer.load(path)   # load không fit lại

Quy ước:
  PAD_ID = 0 (không đóng góp vào pooling)
  UNK_ID = 1 (ký tự lạ)

Đảm bảo:
  - Tokenizer không fit lại ở validation/inference.
  - Ký tự lạ → UNK; báo tỷ lệ UNK mỗi lần transform.
  - max_length fit từ train (max độ dài); mọi cap khác phải là config.
  - CNN mask padding: attention_mask=True nơi token hợp lệ.
  - CNN output bất biến khi chỉ thêm PAD đuôi chuỗi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PAD_ID = 0
UNK_ID = 1


class CharTokenizer:
    """
    Tokenizer ký tự cho SMILES.

    Attributes:
        vocab: dict char → int (PAD=0, UNK=1, ký tự thật bắt đầu từ 2).
        max_length: Độ dài tối đa sau fit (từ train).
        fitted: True sau khi fit được gọi.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"[PAD]": PAD_ID, "[UNK]": UNK_ID}
        self.max_length: int | None = None
        self.fitted: bool = False

    def fit(self, smiles_list: list[str]) -> "CharTokenizer":
        """
        Fit vocabulary từ danh sách SMILES (chỉ dùng train).

        Args:
            smiles_list: Danh sách canonical SMILES của tập train.

        Returns:
            self (để chain)
        """
        chars: set[str] = set()
        max_len = 0
        for smi in smiles_list:
            smi = str(smi)
            chars.update(smi)
            max_len = max(max_len, len(smi))

        # Gán ID theo thứ tự sorted để ổn định
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        self.max_length = max_len
        self.fitted = True
        return self

    def encode(self, smi: str, max_length: int | None = None) -> tuple[list[int], list[int]]:
        """
        Encode một SMILES thành danh sách ID.

        Args:
            smi: SMILES string.
            max_length: Độ dài tối đa (mặc định self.max_length).

        Returns:
            (input_ids, attention_mask) cùng độ dài max_length.
            attention_mask: 1 = token hợp lệ, 0 = PAD.
        """
        if not self.fitted:
            raise RuntimeError("Tokenizer chưa được fit. Gọi fit(train_smiles) trước.")

        L = max_length or self.max_length
        if L is None:
            raise ValueError("max_length chưa được set. Gọi fit() trước.")

        ids = [self.vocab.get(ch, UNK_ID) for ch in str(smi)]
        # Truncate
        ids = ids[:L]
        mask = [1] * len(ids)
        # Pad
        pad_len = L - len(ids)
        ids += [PAD_ID] * pad_len
        mask += [0] * pad_len
        return ids, mask

    def transform(
        self,
        smiles_list: list[str],
        max_length: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Encode danh sách SMILES.

        Returns:
            input_ids: int64 [N, L]
            attention_mask: bool [N, L]
            stats: dict với n_unk, n_truncated, unk_rate
        """
        if not self.fitted:
            raise RuntimeError("Tokenizer chưa được fit.")

        L = max_length or self.max_length
        all_ids = []
        all_masks = []
        n_unk = 0
        n_truncated = 0

        for smi in smiles_list:
            smi = str(smi)
            # Đếm UNK trước truncate
            raw_ids = [self.vocab.get(ch, UNK_ID) for ch in smi]
            n_unk += sum(1 for x in raw_ids if x == UNK_ID)
            if len(smi) > L:
                n_truncated += 1
            ids, mask = self.encode(smi, L)
            all_ids.append(ids)
            all_masks.append(mask)

        input_ids = np.array(all_ids, dtype=np.int64)
        attention_mask = np.array(all_masks, dtype=bool)
        total_tokens = sum(len(str(s)) for s in smiles_list)
        stats = {
            "n_unk": n_unk,
            "n_truncated": n_truncated,
            "unk_rate": n_unk / max(total_tokens, 1),
            "truncation_rate": n_truncated / max(len(smiles_list), 1),
        }
        return input_ids, attention_mask, stats

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: Path) -> None:
        """Lưu tokenizer vào JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "max_length": self.max_length,
            "fitted": self.fitted,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        """Load tokenizer từ JSON; không fit lại."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer không tìm thấy: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.max_length = data["max_length"]
        tok.fitted = data["fitted"]
        return tok

    def check_config(self, expected: dict) -> list[str]:
        """Kiểm tra vocab_size và max_length khớp expected."""
        errors = []
        if "vocab_size" in expected and self.vocab_size != expected["vocab_size"]:
            errors.append(f"vocab_size: {self.vocab_size} != {expected['vocab_size']}")
        if "max_length" in expected and self.max_length != expected["max_length"]:
            errors.append(f"max_length: {self.max_length} != {expected['max_length']}")
        return errors
