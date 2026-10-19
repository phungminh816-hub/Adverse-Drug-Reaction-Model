"""Training MLP/CNN: masked BCE, validation AP, early stopping và resume."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from adr.config import load_config, resolve_paths, save_config
from adr.data.load import load_split
from adr.evaluate.metrics import evaluate
from adr.features.fingerprints import MorganFeaturizer
from adr.features.tokenizer import CharTokenizer
from adr.models.cnn import SmilesCharCNN
from adr.models.mlp import MLP
from adr.schemas import ArtifactMetadata
from adr.features.tokenizer import PAD_ID, UNK_ID
from adr.train.losses import masked_bce_loss
from adr.utils.paths import get_library_versions, get_run_dir, get_run_id, sha256_file
from adr.utils.seed import set_seed


def _encode(smiles, model_type, encoder):
    """Chuyển SMILES thành tensor fingerprint hoặc cặp token/attention mask."""
    if model_type == "mlp":
        values, errors = encoder.transform(smiles)
        if errors:
            raise ValueError(f"SMILES lỗi khi tạo fingerprint: {errors[:3]}")
        return [torch.from_numpy(values)]
    ids, attention, _ = encoder.transform(smiles)
    return [torch.from_numpy(ids), torch.from_numpy(attention)]


def _epoch(model, loader, device, optimizer=None):
    """Chạy một epoch; cập nhật trọng số khi có optimizer, trả loss và điểm validation."""
    model.train(optimizer is not None)
    total_loss, n_observed = 0.0, 0
    scores = []
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            *inputs, targets, mask = [x.to(device) for x in batch]
            if not mask.any():
                raise ValueError("Batch không có nhãn quan sát.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = masked_bce_loss(logits, targets, mask)
            if not torch.isfinite(loss):
                raise ValueError("Loss không hữu hạn; kiểm tra dữ liệu và learning rate.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count = int(mask.sum().item())
            total_loss += loss.item() * count
            n_observed += count
            if optimizer is None:
                scores.append(logits.sigmoid().cpu().numpy())
    return total_loss / n_observed, np.concatenate(scores) if scores else None


def train_neural(cfg: dict, resume: str | Path | None = None, device: str = "auto") -> Path:
    """Train hoặc resume MLP/CNN, lưu checkpoint tốt nhất theo validation macro AP."""
    cfg = resolve_paths(cfg)
    mt = cfg.get("model", {}).get("type")
    if mt not in {"mlp", "cnn"}:
        raise ValueError("Neural trainer chỉ hỗ trợ mlp hoặc cnn.")
    tc = cfg["training"]
    if tc.get("optimizer", "adam") != "adam" or tc.get("loss", "bce_with_logits") != "bce_with_logits":
        raise ValueError("Trainer hỗ trợ optimizer=adam và loss=bce_with_logits.")
    if tc.get("checkpoint_metric", "macro_ap") != "macro_ap":
        raise ValueError("Checkpoint phải được chọn theo validation macro_ap.")
    if min(tc["max_epochs"], tc["batch_size"], tc["early_stopping_patience"], tc["learning_rate"]) <= 0:
        raise ValueError("Epoch, batch size, patience và learning rate phải dương.")
    seed = cfg["project"]["seed"]
    set_seed(seed)
    torch.set_num_threads(int(tc.get("num_threads", 4)))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA chưa sẵn sàng; dùng --device cpu.")
    processed = cfg["paths"]["data_processed"] / cfg["data"]["processed_filename"]
    splits = cfg["paths"]["data_splits"]
    labels = json.loads((cfg["paths"]["data_processed"] / cfg["data"]["label_names_file"]).read_text(encoding="utf-8"))
    manifest_filename = cfg["data"].get("split_manifest", "split_manifest.csv")
    _, train_smiles, y_train, m_train = load_split(processed, splits, labels, "train", manifest_filename)
    _, val_smiles, y_val, m_val = load_split(processed, splits, labels, "val", manifest_filename)
    if not m_train.any(axis=0).all():
        raise ValueError("Có nhãn không có sample quan sát trong train.")
    if not np.any((y_val == 1) & m_val):
        raise ValueError("Validation không có nhãn dương để chọn checkpoint theo AP.")
    checksums = (sha256_file(processed), sha256_file(splits / manifest_filename), sha256_file(processed.parent / "label_mask.npy"))
    run_dir = Path(resume) if resume else get_run_dir(cfg["paths"]["artifacts"], get_run_id(mt, seed))
    if resume:
        meta = ArtifactMetadata.load(run_dir / "metadata.json")
        if (meta.dataset_checksum, meta.split_manifest_checksum, meta.label_mask_checksum) != checksums:
            raise ValueError("Không thể resume: dữ liệu, mask hoặc split đã thay đổi.")
        meta.check_label_names(labels)
        if meta.model_type != mt or meta.seed != seed:
            raise ValueError("Không thể resume: model type hoặc seed khác run gốc.")
        for section in ("model", "features"):
            if meta.config_resolved.get(section) != cfg.get(section):
                raise ValueError(f"Không thể resume: cấu hình {section} đã thay đổi.")
        for key in ("batch_size", "learning_rate", "optimizer", "loss", "early_stopping_patience"):
            if meta.config_resolved["training"].get(key) != tc.get(key):
                raise ValueError(f"Không thể resume: training.{key} đã thay đổi.")
    mc = cfg["model"]
    if mt == "mlp":
        fp = cfg["features"]["fingerprint"]
        encoder = MorganFeaturizer(radius=fp["radius"], n_bits=fp["n_bits"], use_chirality=fp["use_chirality"])
        encoder.save(run_dir / "fingerprint_config.json")
        model = MLP(input_dim=fp["n_bits"], hidden_dim=mc.get("hidden_dim", 256), n_labels=len(labels), dropout=mc.get("dropout", 0.2))
        feature_config = encoder.get_config()
    else:
        tokenizer_cfg = cfg["features"]["tokenizer"]
        if tokenizer_cfg.get("pad_token_id", PAD_ID) != PAD_ID or tokenizer_cfg.get("unk_token_id", UNK_ID) != UNK_ID:
            raise ValueError("Tokenizer sử dụng pad_token_id=0 và unk_token_id=1.")
        encoder = CharTokenizer.load(run_dir / "tokenizer.json") if resume else CharTokenizer().fit(train_smiles)
        cap = cfg["features"]["tokenizer"].get("max_length")
        if cap is not None:
            if cap <= 0:
                raise ValueError("tokenizer.max_length phải dương.")
            encoder.max_length = min(encoder.max_length, cap)
        encoder.save(run_dir / "tokenizer.json")
        feature_config = {"type": "char", "vocab_size": encoder.vocab_size, "max_length": encoder.max_length}
        model = SmilesCharCNN(vocab_size=encoder.vocab_size, embedding_dim=mc.get("embedding_dim", 64), channels=mc.get("channels", 128), kernel_size=mc.get("kernel_size", 5), n_labels=len(labels), dropout=mc.get("dropout", 0.2))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    train_ds = TensorDataset(*_encode(train_smiles, mt, encoder), torch.from_numpy(y_train), torch.from_numpy(m_train))
    val_ds = TensorDataset(*_encode(val_smiles, mt, encoder), torch.from_numpy(y_val), torch.from_numpy(m_val))
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"])
    history, start_epoch, best_ap, best_epoch, bad_epochs = [], 0, -float("inf"), 0, 0
    if resume:
        state = torch.load(run_dir / "training_state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        start_epoch, best_ap, best_epoch = state["epoch"], state["best_ap"], state["best_epoch"]
        bad_epochs, history = state["bad_epochs"], state["history"]
        torch.set_rng_state(state["rng"])
        if device == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    meta = ArtifactMetadata(
        model_type=mt, run_id=run_dir.name, seed=seed, label_names=labels,
        dataset_checksum=checksums[0], split_manifest_checksum=checksums[1], label_mask_checksum=checksums[2],
        feature_config=feature_config, library_versions=get_library_versions(),
        threshold=cfg.get("inference", {}).get("threshold", 0.5),
        eval_label_policy="observed labels; AP null without positives; ROC-AUC null with one class",
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_resolved=cfg, tokenizer_path="tokenizer.json" if mt == "cnn" else None,
    )
    meta.save(run_dir / "metadata.json")
    save_config(cfg, run_dir)
    print(f"[neural] {mt} device={device}, train={len(train_ds)}, val={len(val_ds)}", flush=True)
    for epoch in range(start_epoch + 1, tc["max_epochs"] + 1):
        if bad_epochs >= tc["early_stopping_patience"]:
            break

        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, generator=generator)
        train_loss, _ = _epoch(model, loader, device, optimizer)
        val_loss, scores = _epoch(model, val_loader, device)
        ap = evaluate(scores, y_val, m_val, labels).macro_ap
        if ap is None or not np.isfinite(ap):
            raise ValueError("Validation macro AP không xác định.")
        if ap > best_ap:
            best_ap, best_epoch, bad_epochs = ap, epoch, 0
            model.save(run_dir)
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_ap": ap})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "best_ap": best_ap, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "history": history,
                 "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []}
        torch.save(state, run_dir / "training_state.tmp")
        (run_dir / "training_state.tmp").replace(run_dir / "training_state.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[neural] epoch={epoch} loss={train_loss:.4f} val_loss={val_loss:.4f} val_AP={ap:.4f}", flush=True)
    (run_dir / "training_summary.json").write_text(json.dumps({"best_epoch": best_epoch, "best_val_macro_ap": best_ap, "epochs_completed": len(history), "device": device}, indent=2), encoding="utf-8")
    print(f"[neural] Artifact: {run_dir}", flush=True)
    return run_dir


def main(argv=None):
    """Đọc tham số CLI và khởi chạy neural trainer."""
    parser = argparse.ArgumentParser(description="Train MLP/CNN, chọn checkpoint bằng validation macro AP")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", default=None)
    parser.add_argument("--resume", default=None, help="Run directory chứa training_state.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Tổng số epoch tối đa, kể cả epoch đã resume")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, base=args.base)
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    train_neural(cfg, resume=args.resume, device=args.device)


if __name__ == "__main__":
    main()
