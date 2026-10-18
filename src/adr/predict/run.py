"""
adr.predict.run – CLI dự đoán ADR cho SMILES đầu vào.

CLI:
  python -m adr.predict.run --run-dir artifacts/<run_id> --smiles "CCO"
  python -m adr.predict.run --run-dir artifacts/<run_id> --smiles-file my_smiles.txt

Cảnh báo giới hạn diễn giải (in cùng kết quả):
  - Điểm sigmoid chưa hiệu chỉnh, không phải xác suất bệnh nhân gặp ADR.
  - Nhãn 0 không chứng minh thuốc an toàn.
  - Nhãn là nhóm rộng MedDRA, không phải triệu chứng cụ thể.
  - CCO (ethanol) chỉ là input kiểm tra kỹ thuật, không gán tên thuốc y khoa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adr.predict.predictor import Predictor


_INTERPRETATION_WARNING = """
⚠  Giới hạn diễn giải:
   - Điểm này là output của model học máy, KHÔNG phải xác suất bệnh nhân gặp ADR.
   - Nhãn 0 / score thấp không chứng minh thuốc an toàn.
   - Các nhãn là nhóm rộng MedDRA; model học liên hệ thuốc–nhãn trong benchmark SIDER.
   - Không dùng kết quả này cho quyết định lâm sàng.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dự đoán ADR scores cho SMILES từ model artifact",
        epilog=(
            "Ví dụ:\n"
            "  python -m adr.predict.run --run-dir artifacts/logreg_xxx --smiles 'CCO'\n"
            "  python -m adr.predict.run --run-dir artifacts/logreg_xxx --smiles-file smiles.txt\n"
            "\nCCO là input kiểm tra kỹ thuật, không gán tên thuốc hay suy diễn y khoa.\n"
        ),
    )
    parser.add_argument("--run-dir", required=True, help="Path đến run artifact directory")
    parser.add_argument("--smiles", default=None, help="SMILES string để predict")
    parser.add_argument("--smiles-file", default=None, help="File txt, mỗi dòng một SMILES")
    parser.add_argument("--top-k", type=int, default=5, help="Số nhãn top-k hiển thị (mặc định 5)")
    parser.add_argument("--output", default=None, help="Lưu kết quả JSON vào file")
    parser.add_argument("--all-labels", action="store_true", help="Hiển thị tất cả 27 nhãn")
    args = parser.parse_args(argv)

    if args.smiles is None and args.smiles_file is None:
        parser.error("Cần truyền --smiles hoặc --smiles-file")

    try:
        pred = Predictor(args.run_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[predict] LỖI: {e}", file=sys.stderr)
        sys.exit(1)

    smiles_list = []
    if args.smiles:
        smiles_list.append(args.smiles)
    if args.smiles_file:
        fp = Path(args.smiles_file)
        if not fp.exists():
            print(f"[predict] File không tồn tại: {fp}", file=sys.stderr)
            sys.exit(1)
        smiles_list.extend(line.strip() for line in fp.read_text(encoding="utf-8").splitlines() if line.strip())

    results = []
    for smi in smiles_list:
        result = pred.predict(smi)
        results.append(result)

        print(f"\nSMILES: {smi}")
        if result.canonical_smiles:
            print(f"Canonical: {result.canonical_smiles}")
        else:
            print("⚠  SMILES không hợp lệ.")
        if result.warnings:
            for w in result.warnings:
                print(f"  ! {w}")

        if result.canonical_smiles is not None:
            print(f"\nTop-{args.top_k} nhóm ADR:")
            for label, score in result.top_k(args.top_k):
                flag = "✓" if score >= result.threshold else " "
                print(f"  [{flag}] {label[:60]:<60} {score:.4f}")

            if args.all_labels:
                print(f"\nTất cả 27 nhóm:")
                for j, (label, score) in enumerate(zip(result.label_names, result.scores)):
                    flag = "✓" if score >= result.threshold else " "
                    print(f"  [{flag}] {j+1:2d}. {label[:55]:<55} {score:.4f}")

        print(_INTERPRETATION_WARNING)

    # Lưu JSON nếu yêu cầu
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in results], fh, indent=2, ensure_ascii=False, default=str)
        print(f"[predict] Kết quả lưu: {out_path}")


if __name__ == "__main__":
    main()
