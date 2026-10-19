"""Tạo Morgan fingerprint từ SMILES và lưu cấu hình đặc trưng."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False


def _require_rdkit() -> None:
    if not _RDKIT_OK:
        raise ImportError(
            "RDKit chưa được cài. Cài bằng:\n"
            "  conda install -c conda-forge rdkit\n"
            "  hoặc: pip install rdkit"
        )


class MorganFeaturizer:
    """
    Tạo Morgan fingerprint (ECFP) từ SMILES.

    Args:
        radius: Bán kính lân cận nguyên tử (ECFP4 = radius 2).
        n_bits: Số bit (mặc định 2048).
        use_chirality: Giữ thông tin lập thể.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = True,
    ) -> None:
        _require_rdkit()
        self.radius = radius
        self.n_bits = n_bits
        self.use_chirality = use_chirality
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=use_chirality,
        )

    def _smiles_to_fp(self, smi: str) -> tuple[np.ndarray, str | None]:
        """
        Chuyển một SMILES thành fingerprint vector.

        Returns:
            (fp_array [n_bits], error_message | None)
        """
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            return np.zeros(self.n_bits, dtype=np.float32), f"RDKit parse lỗi: {smi[:60]}"
        arr = self._generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        return arr, None

    def transform(
        self,
        smiles_list: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Chuyển danh sách SMILES thành ma trận feature.

        Args:
            smiles_list: Danh sách canonical SMILES.

        Returns:
            X: np.ndarray [N, n_bits] float32
            errors: list[str] - SMILES gây lỗi; rỗng nếu tất cả hợp lệ.
        """
        rows = []
        errors = []
        for smi in smiles_list:
            fp, err = self._smiles_to_fp(str(smi).strip())
            rows.append(fp)
            if err:
                errors.append(err)
        X = np.stack(rows, axis=0) if rows else np.empty((0, self.n_bits), dtype=np.float32)
        return X, errors

    def get_config(self) -> dict:
        return {
            "type": "morgan",
            "radius": self.radius,
            "n_bits": self.n_bits,
            "use_chirality": self.use_chirality,
        }

    def save(self, path: Path) -> None:
        """Lưu config fingerprint vào JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.get_config(), fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MorganFeaturizer":
        """Load từ config JSON; kiểm tra type khớp."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fingerprint config không tìm thấy: {path}")
        with path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg.get("type") != "morgan":
            raise ValueError(f"Loại fingerprint không khớp: {cfg.get('type')} != morgan")
        return cls(
            radius=cfg["radius"],
            n_bits=cfg["n_bits"],
            use_chirality=cfg["use_chirality"],
        )

    def check_config(self, expected: dict) -> list[str]:
        """Kiểm tra config khớp expected; trả list sai biệt."""
        errors = []
        for key in ("radius", "n_bits", "use_chirality"):
            if self.get_config().get(key) != expected.get(key):
                errors.append(
                    f"fingerprint.{key}: artifact={self.get_config()[key]}, expected={expected.get(key)}"
                )
        return errors
