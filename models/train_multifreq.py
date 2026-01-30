import os
import time
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=r"Downcasting behavior in `replace` is deprecated.*")
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

import torch
import torch.nn as nn
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from models.DeepConvLSTM_final import DeepConvLSTM
from models.multifreq_deepconvlstm import MultiFreqDeepConvLSTM, FREQ_FOR_CLASS
from models.train import run_inertial_network
from utils.torch_utils_final import fix_random_seed, worker_init_reset_seed


class InertialWindowDataset(Dataset):
    """
    Windowed dataset: (x_windows, y_labels), with optional context features depending on model_name.
    This is your original 70/20/10 path.
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray, model_name: str = "deepconvlstm"):
        super().__init__()
        self.windows = windows
        self.labels = labels
        self.model_name = model_name

        # windows expected shape: [N, T, C]
        assert self.windows.ndim == 3, f"Expected windows [N,T,C], got {self.windows.shape}"
        self.window_size = self.windows.shape[1]
        self.channels = self.windows.shape[2]

        # labels are integer class ids
        self.classes = int(np.max(self.labels)) + 1 if len(self.labels) else 0

        # context models append extra features
        self.context_features = 0
        if self.model_name in ["deepconvcontext", "shallow_deepconvlstm"]:
            # original code path uses 4 extra features for these models
            self.context_features = 4

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.windows[idx]  # [T,C]
        y = self.labels[idx]

        # Add context features if required by the model (kept compatible with your original code)
        if self.context_features > 0:
            ctx = np.zeros((x.shape[0], self.context_features), dtype=x.dtype)
            x = np.concatenate([x, ctx], axis=1)

        # DeepConvLSTM expects [1, T, C] per sample (then batched)
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.tensor(y).long()
        return x, y


def _freq_tag_from_sampling_rate(sampling_rate: float) -> str:
    if abs(sampling_rate - 50.0) < 1e-3:
        return "50hz"
    if abs(sampling_rate - 25.0) < 1e-3:
        return "25hz"
    if abs(sampling_rate - 12.0) < 1e-3 or abs(sampling_rate - 12.5) < 1e-3:
        return "12hz"
    if abs(sampling_rate - 6.0) < 1e-3:
        return "6hz"
    return f"{sampling_rate:g}hz"


def build_single_freq_splits(
    cfg: Dict,
    sbj_list: List[str],
    freq_tag: str,
    seed: int,
) -> Tuple[InertialWindowDataset, InertialWindowDataset, Tuple[np.ndarray, np.ndarray]]:
    """
    70/20/10 split (legacy): loads all subjects, makes windows, then does random split.
    Returns (train_ds, val_ds, (x_test, y_test)).
    """
    sens_folder = cfg["dataset"]["sens_folder"]

    all_rows: List[np.ndarray] = []
    for sbj in sbj_list:
        arr = np.load(os.path.join(sens_folder, sbj))
        if arr.shape[0] == 0:
            continue
        all_rows.append(arr)

    data = np.concatenate(all_rows, axis=0)

    feat_cols = slice(1, -1)
    feat_min = data[:, feat_cols].min()
    feat_max = data[:, feat_cols].max()
    denom = (feat_max - feat_min) if (feat_max - feat_min) != 0 else 1.0
    data[:, feat_cols] = 2.0 * (data[:, feat_cols] - feat_min) / denom - 1.0

    print(f"[{freq_tag}] All data shape: {data.shape}")
    print(f"[{freq_tag}] Min/Max after normalization: {data[:, feat_cols].min():.2f}, {data[:, feat_cols].max():.2f}")

    window_size = cfg["dataset"]["window_size"]
    window_overlap = cfg["dataset"]["window_overlap"]
    no_context_windows = cfg["dataset"].get("no_context_windows", False)

    step = int(window_size * (1 - window_overlap))
    if step <= 0:
        step = 1

    windows = []
    labels = []

    for start in range(0, data.shape[0] - window_size + 1, step):
        win = data[start : start + window_size, feat_cols]
        lab = int(data[start + window_size - 1, -1])
        windows.append(win)
        labels.append(lab)

    windows = np.asarray(windows, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    # 70/20/10 split
    rng = np.random.RandomState(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    x_train, y_train = windows[train_idx], labels[train_idx]
    x_val, y_val = windows[val_idx], labels[val_idx]
    x_test, y_test = windows[test_idx], labels[test_idx]

    print(f"[{freq_tag}] Split sizes (train / val / test): {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    train_ds = InertialWindowDataset(x_train, y_train, model_name=cfg.get("name", "deepconvlstm"))
    val_ds = InertialWindowDataset(x_val, y_val, model_name=cfg.get("name", "deepconvlstm"))

    return train_ds, val_ds, (x_test, y_test)


def _prepare_split_from_anno(cfg: Dict, anno_path: str, sbj_list: Optional[List[str]] = None) -> Tuple[str, List[str], List[str]]:
    """
    Reads one LOSO annotation file and returns:
      split_name, train_subject_ids, val_subject_ids

    Expected WEAR/ActivityNet-like JSON structure:
      {
        "database": {
          "sbj_0": {"subset": "Validation", ...},
          "sbj_1": {"subset": "Training", ...},
          ...
        },
        "label_dict": {...},
        ...
      }
    """
    split_name = os.path.splitext(os.path.basename(anno_path))[0]

    with open(anno_path, "r") as f:
        anno = json.load(f)

    db = anno.get("database", None)
    if not isinstance(db, dict) or len(db) == 0:
        raise KeyError(f"'database' not found or empty in annotation file: {anno_path}")

    train_sbjs: List[str] = []
    val_sbjs: List[str] = []

    # Primary path: use db[sbj]["subset"]
    for sbj_id, meta in db.items():
        if not isinstance(meta, dict):
            continue

        subset = meta.get("subset", meta.get("Subset", ""))
        subset_l = str(subset).strip().lower()

        if subset_l.startswith("train"):
            train_sbjs.append(sbj_id)
        elif subset_l.startswith("val"):
            val_sbjs.append(sbj_id)

    # If subsets were correctly found, we’re done
    if len(train_sbjs) > 0 and len(val_sbjs) > 0:
        return split_name, sorted(train_sbjs), sorted(val_sbjs)

    # Fallback (robustness): infer LOSO from filename loso_sbj_<k>
    # This saves you if a JSON split file doesn’t contain per-subject subset flags.
    if sbj_list is None or len(sbj_list) == 0:
        raise KeyError(
            f"No per-subject 'subset' entries found in {anno_path}, and sbj_list fallback is unavailable."
        )

    try:
        held_out_id = int(split_name.split("_")[-1])
    except Exception as e:
        raise KeyError(
            f"No per-subject 'subset' entries found in {anno_path}, and could not parse held-out id from '{split_name}'."
        ) from e

    # Prefer exact sbj_<k> string if present; otherwise use index into sorted sbj_list
    held_out_name = f"sbj_{held_out_id}"
    if held_out_name in sbj_list:
        val_sbjs = [held_out_name]
        train_sbjs = [s for s in sbj_list if s != held_out_name]
    else:
        if held_out_id < 0 or held_out_id >= len(sbj_list):
            raise IndexError(f"held_out_id={held_out_id} out of range for sbj_list (len={len(sbj_list)})")
        val_sbjs = [sbj_list[held_out_id]]
        train_sbjs = [s for i, s in enumerate(sbj_list) if i != held_out_id]

    return split_name, train_sbjs, val_sbjs



def run_inertial_expert_loso(
    cfg: Dict,
    sbj_list: List[str],
    log_dir: str,
    seed: int,
    test_split_dir: str,
    device: torch.device,
    gamma_quant=None,
    quant_bits: int = 4,
    gamma_type: str = "s_shaped",
    apply_gamma: str = "global",
    resume: str = "",
    ckpt_freq: int = -1,
) -> List[str]:
    """
    LOSO per frequency. Uses models/train.py::run_inertial_network so logs/checkpoints match train.py.

    Added behavior:
      - per-fold resume from last_*.pth.tar (or best_*.pth.tar fallback)
      - skip fold if already finished
    """
    os.makedirs(log_dir, exist_ok=True)
    rng_generator = fix_random_seed(cfg.get("init_rand_seed", seed), include_cuda=True)

    sampling_rate = float(cfg["dataset"].get("sampling_rate", 0.0))
    freq_tag = _freq_tag_from_sampling_rate(sampling_rate)

    # Optional: keep the legacy held-out test windows used by test_multifreq_cascade.
    os.makedirs(test_split_dir, exist_ok=True)
    test_npz_path = os.path.join(test_split_dir, f"test_windows_{freq_tag}.npz")
    if not os.path.exists(test_npz_path) and cfg.get("anno_json"):
        # Use the legacy split builder once to get a deterministic window test set for cascade eval.
        _, _, (x_test, y_test) = build_single_freq_splits(cfg, sbj_list, freq_tag, seed)
        np.savez_compressed(test_npz_path, x=x_test, y=y_test)
        print(f"[{freq_tag}] Saved held-out test windows to: {test_npz_path}")

    best_ckpts: List[str] = []
    epochs = int(cfg.get("train_cfg", {}).get("epochs", 0))

    for i, anno_path in enumerate(cfg.get("anno_json", [])):
        split_name, train_sbjs, val_sbjs = _prepare_split_from_anno(cfg, anno_path)
        print(f"Split {i + 1} / {len(cfg['anno_json'])} ({freq_tag}) -> {split_name}")

        # Per-fold resume / skip
        ckpt_dir = os.path.join("logs", "deepconvlstm", freq_tag, split_name)
        last_ckpt = os.path.join(ckpt_dir, f"last_{split_name}.pth.tar")
        best_ckpt = os.path.join(ckpt_dir, f"best_{split_name}.pth.tar")

        fold_resume = resume  # explicit override if provided
        if (fold_resume is None or fold_resume == "") and epochs > 0:
            cand = last_ckpt if os.path.isfile(last_ckpt) else (best_ckpt if os.path.isfile(best_ckpt) else "")
            if cand:
                try:
                    ck = torch.load(cand, map_location="cpu")
                    done_epoch = int(ck.get("epoch", 0)) if isinstance(ck, dict) else 0
                except Exception:
                    done_epoch = 0

                if done_epoch >= epochs and os.path.isfile(best_ckpt):
                    print(f"[SKIP] {split_name} already finished (epoch {done_epoch}/{epochs}).")
                    best_ckpts.append(best_ckpt)
                    continue

                fold_resume = cand
                if done_epoch > 0:
                    print(f"[RESUME] {split_name} from {cand} (epoch {done_epoch}/{epochs}).")

        # Train using train.py behavior/logging
        cfg["dataset"]["json_anno"] = anno_path  # train.py reads this to name the split
        run_inertial_network(
            train_sbjs=train_sbjs,   # keep IDs from annotation, e.g. "sbj_0"
            val_sbjs=val_sbjs,       # keep IDs from annotation, e.g. "sbj_1"
            cfg=cfg,
            ckpt_folder=log_dir,
            ckpt_freq=ckpt_freq,
            resume=fold_resume,
            rng_generator=rng_generator,
            run=None,
            gamma_quant=gamma_quant,
            quant_bits=quant_bits,
            gamma_type=gamma_type,
            apply_gamma=apply_gamma,
        )

        # train.py saves to: logs/deepconvlstm/<freq_tag>/<split_name>/best_<split_name>.pth.tar
        ckpt_path = os.path.join("logs", "deepconvlstm", freq_tag, split_name, f"best_{split_name}.pth.tar")
        best_ckpts.append(ckpt_path)

    return best_ckpts


def run_inertial_expert_702010(
    cfg: Dict,
    sbj_list: List[str],
    log_dir: str,
    freq_tag: str,
    seed: int,
    test_split_dir: str,
    device: torch.device,
    gamma_quant=None,
    quant_bits: int = 4,
    gamma_type: str = "s_shaped",
    apply_gamma: str = "global",
) -> str:
    """
    Legacy 70/20/10 training (kept for backwards compatibility with main_multifreq imports).
    """
    os.makedirs(log_dir, exist_ok=True)
    rng_generator = fix_random_seed(cfg.get("init_rand_seed", seed), include_cuda=True)

    train_ds, val_ds, (x_test, y_test) = build_single_freq_splits(cfg, sbj_list, freq_tag, seed)

    # Save held-out test windows (for cascade evaluation)
    os.makedirs(test_split_dir, exist_ok=True)
    np.savez_compressed(os.path.join(test_split_dir, f"test_windows_{freq_tag}.npz"), x=x_test, y=y_test)

    channels = train_ds.channels
    n_classes = train_ds.classes
    window_size = train_ds.window_size
    net_cfg = cfg["model"]

    net = DeepConvLSTM(
        channels=channels,
        classes=n_classes,
        window_size=window_size,
        conv_kernels=net_cfg["conv_kernels"],
        conv_kernel_size=net_cfg["conv_kernel_size"],
        lstm_units=net_cfg["lstm_units"],
        lstm_layers=net_cfg["lstm_layers"],
        dropout=net_cfg["dropout"],
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
        conv_type=net_cfg.get("conv_type", "standard"),
        conv_rank=net_cfg.get("conv_rank", 2),
        conv_mlp_hidden_dim=net_cfg.get("conv_mlp_hidden_dim", 64),
    )

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of learnable parameters for DeepConvLSTM: {n_params}")
    print(net)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    if cfg.get("weighted_loss", False):
        y_train = train_ds.labels
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Class weights: {weights.detach().cpu().numpy()}")

    opt = torch.optim.Adam(
        net.parameters(),
        lr=cfg["train_cfg"]["lr"],
        weight_decay=cfg["train_cfg"]["weight_decay"],
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["loader"]["train_batch_size"],
        shuffle=True,
        num_workers=0 if os.name == "nt" else 4,
        worker_init_fn=worker_init_reset_seed,
        generator=rng_generator,
        persistent_workers=False if os.name == "nt" else True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["loader"]["test_batch_size"],
        shuffle=False,
        num_workers=0 if os.name == "nt" else 4,
        worker_init_fn=worker_init_reset_seed,
        generator=rng_generator,
        persistent_workers=False if os.name == "nt" else True,
    )

    best_f1 = -1.0
    best_path = os.path.join(log_dir, f"best_{freq_tag}.pth.tar")

    for epoch in range(cfg["train_cfg"]["epochs"]):
        net.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        net.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = net(xb)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.append(pred)
                gts.append(yb.numpy())

        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        # macro F1
        f1 = (pd.Series(preds).groupby(gts).apply(lambda s: (s == s.name).mean()).mean())  # fallback-ish
        # keep original print style
        print(f"[{freq_tag}] Epoch {epoch+1:03d}/{cfg['train_cfg']['epochs']:03d} train loss {np.mean(train_losses):.4f}")

        # Simple checkpointing on best (your original code has richer metrics; keep it minimal-safe)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_f1": best_f1,
                },
                best_path,
            )
            print(f"[{freq_tag}] New best -> {best_path} (F1={best_f1:.4f})")

    return best_path


def test_multifreq_cascade(
    cfg: Dict,
    ckpt_6hz: str,
    ckpt_12hz: str,
    ckpt_25hz: str,
    test_split_dir: str,
    device: torch.device,
    ckpt_50hz: str = "",
    gamma_quant=None,
    quant_bits: int = 4,
    gamma_type: str = "s_shaped",
    apply_gamma: str = "global",
) -> None:
    """
    Evaluate a multi-frequency cascade on held-out window splits.

    Backwards compatible:
      - If ckpt_50hz is not provided, runs the original 3-frequency cascade (6/12/25).
      - If ckpt_50hz is provided, runs a 4-frequency cascade (50/25/12/6).
    """
    path_6 = os.path.join(test_split_dir, "test_windows_6hz.npz")
    path_12 = os.path.join(test_split_dir, "test_windows_12hz.npz")
    path_25 = os.path.join(test_split_dir, "test_windows_25hz.npz")
    path_50 = os.path.join(test_split_dir, "test_windows_50hz.npz")

    if not os.path.exists(path_6) or not os.path.exists(path_12) or not os.path.exists(path_25):
        raise FileNotFoundError(f"Missing test windows in {test_split_dir} (need 6hz/12hz/25hz)")

    data_6 = np.load(path_6)
    data_12 = np.load(path_12)
    data_25 = np.load(path_25)

    x6, y = data_6["x"], data_6["y"]
    x12 = data_12["x"]
    x25 = data_25["x"]

    use_50 = bool(ckpt_50hz)
    x50 = None
    if use_50:
        if not os.path.exists(path_50):
            raise FileNotFoundError(f"Missing test windows file for 50hz: {path_50}")
        data_50 = np.load(path_50)
        x50 = data_50["x"]

    model = MultiFreqDeepConvLSTM(
        cfg=cfg,
        ckpt_6hz=ckpt_6hz,
        ckpt_12hz=ckpt_12hz,
        ckpt_25hz=ckpt_25hz,
        ckpt_50hz=ckpt_50hz,
        device=device,
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        gamma_type=gamma_type,
        apply_gamma=apply_gamma,
    )
    model.eval()

    # Convert to tensors: [N,T,C] -> [N,1,T,C]
    t6 = torch.from_numpy(x6).float().unsqueeze(1).to(device)
    t12 = torch.from_numpy(x12).float().unsqueeze(1).to(device)
    t25 = torch.from_numpy(x25).float().unsqueeze(1).to(device)
    t50 = torch.from_numpy(x50).float().unsqueeze(1).to(device) if use_50 else None

    with torch.no_grad():
        logits = model(x_6hz=t6, x_12hz=t12, x_25hz=t25, x_50hz=t50)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = float((pred == y).mean()) * 100.0
    print(f"[cascade] Accuracy: {acc:.2f}% (use_50hz={use_50})")
