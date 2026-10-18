"""
adr.train.neural – Training loop cho MLP và CNN.

TODO (Tuần 5-7):
  Input:
    - processed.csv + split_manifest + features (NPZ fingerprint hoặc tokenizer artifact)
    - config đã resolve (mlp.yaml hoặc cnn.yaml)
  Output:
    - Checkpoint PyTorch tốt nhất theo validation macro AP
    - Log epoch (train loss, val macro AP, val loss)
    - metadata.json với config, seed, label_names, library versions

  Bước cần thực hiện:
    1. DataLoader cho fingerprint (MLP) hoặc token IDs (CNN)
    2. Train loop: forward → masked_bce_loss → backward → optimizer step
    3. Val loop: eval mode + no_grad → compute macro AP
    4. Early stopping: patience=8 theo val macro AP
    5. Checkpoint: lưu state_dict khi val macro AP cải thiện
    6. Khởi tạo lại từ checkpoint (resume) nếu được truyền --resume
    7. Log epoch vào file log và stdout

  Điều kiện hoàn thành:
    - Chạy full 50 epoch trên SIDER train không lỗi
    - Val macro AP báo đúng
    - Test KHÔNG được evaluate tự động (cần flag --evaluate-test với metadata locked)

CLI (TODO):
  python -m adr.train.neural --config configs/mlp.yaml
  python -m adr.train.neural --config configs/cnn.yaml --resume artifacts/<run_id>
"""

import sys


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="[TODO] Training loop cho neural model (MLP/CNN). Chưa triển khai.",
        epilog=(
            "Trạng thái: TODO (Tuần 5-7 theo roadmap)\n"
            "  Dùng: python -m adr.train.baseline --config configs/logreg.yaml\n"
            "  để train baseline hiện có.\n"
        ),
    )
    parser.add_argument("--config", required=True, help="Path đến mlp.yaml hoặc cnn.yaml")
    parser.add_argument("--resume", default=None, help="[TODO] Path đến run_dir để resume")
    parser.add_argument("--base", default=None, help="Path đến base.yaml")
    parser.parse_args(argv)

    print(
        "[neural] LỖI: Training neural (MLP/CNN) chưa được triển khai.\n"
        "  Xem TODO trong adr/train/neural.py.\n"
        "  Roadmap: Tuần 5 (MLP), Tuần 6-7 (CNN).\n"
        "  Hiện tại dùng: python -m adr.train.baseline --config configs/logreg.yaml",
        file=sys.stderr,
    )
    sys.exit(10)  # exit code ≠ 0 để script caller phát hiện


if __name__ == "__main__":
    main()
