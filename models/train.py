import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import convert_samples_to_segments, unwindow_inertial_data
from utils.torch_utils_final import (
    init_weights,
    save_checkpoint,
    worker_init_reset_seed,
    InertialDataset,
)
from utils.os_utils import mkdir_if_missing
from models.DeepConvLSTM_final import DeepConvLSTM
from models.map_metric import ANETdetection


def _pretty_freq_tag_from_len(t: int) -> str:
    if int(t) == 50:
        return "50hz"
    if int(t) == 25:
        return "25hz"
    if int(t) == 12:
        return "12hz"
    if int(t) == 6:
        return "6hz"
    return f"{int(t)}t"


def _infer_multirate_lengths_from_model(net: nn.Module):
    if hasattr(net, "branches"):
        b = getattr(net, "branches")
        if isinstance(b, nn.ModuleDict):
            out = []
            for k in b.keys():
                try:
                    out.append(int(k))
                except Exception:
                    pass
            out = sorted(set(out), reverse=True)
            return out
    return []


def _resample_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    x = x.contiguous()
    target_len = int(target_len)
    if x.shape[1] == target_len:
        return x
    x_t = x.transpose(1, 2)
    x_rs = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
    return x_rs.transpose(1, 2).contiguous()


def validate_one_epoch_resampled(loader, network, criterion, target_len: int, gpu=None):
    network.eval()
    losses = []
    preds = np.array([], dtype=np.int64)
    gt = np.array([], dtype=np.int64)

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(loader):
            if gpu is not None:
                inputs = inputs.to(gpu)
                targets = targets.to(gpu)

            inputs = _resample_time(inputs, target_len)

            outputs = network(inputs)
            batch_loss = criterion(outputs, targets.long())
            losses.append(batch_loss.item())

            batch_preds = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
            batch_gt = targets.cpu().numpy().flatten()

            preds = np.concatenate((preds, batch_preds))
            gt = np.concatenate((gt, batch_gt))

    return losses, preds, gt


def run_inertial_network(
    train_sbjs,
    val_sbjs,
    cfg,
    ckpt_folder,
    ckpt_freq,
    resume,
    rng_generator,
    run,
    gamma_quant=None,
    quant_bits=4,
    gamma_type="id",
    apply_gamma="global",
):
    split_name = os.path.splitext(os.path.basename(cfg["dataset"]["json_anno"]))[0]
    sampling_rate = float(cfg["dataset"]["sampling_rate"])

    if abs(sampling_rate - 50.0) < 1e-3:
        freq_tag = "50hz"
    elif abs(sampling_rate - 25.0) < 1e-3:
        freq_tag = "25hz"
    elif abs(sampling_rate - 12.0) < 1e-3 or abs(sampling_rate - 12.5) < 1e-3:
        freq_tag = "12hz"
    elif abs(sampling_rate - 6.0) < 1e-3:
        freq_tag = "6hz"
    else:
        freq_tag = f"{sampling_rate:g}hz"

    ckpt_folder = os.path.join("logs", "deepconvlstm", freq_tag, split_name)
    mkdir_if_missing(ckpt_folder)

    last_ckpt_path = os.path.join(ckpt_folder, f"last_{split_name}.pth.tar")

    auto_resume = cfg.get("train_cfg", {}).get("auto_resume", True)
    if (resume is None or resume == "") and auto_resume and os.path.isfile(last_ckpt_path):
        resume = last_ckpt_path
        print(f"Auto-resume: using {resume}")

    train_data = np.empty((0, cfg["dataset"]["input_dim"] + 2))
    val_data = np.empty((0, cfg["dataset"]["input_dim"] + 2))

    for t_sbj in train_sbjs:
        t_path = os.path.join(cfg["dataset"]["sens_folder"], t_sbj + ".csv")
        t_df = pd.read_csv(t_path, index_col=False, low_memory=False)
        t_df = t_df.replace({"label": cfg["label_dict"]}).fillna(0)
        t_arr = t_df.to_numpy()
        train_data = np.append(train_data, t_arr, axis=0)

    for v_sbj in val_sbjs:
        v_path = os.path.join(cfg["dataset"]["sens_folder"], v_sbj + ".csv")
        v_df = pd.read_csv(v_path, index_col=False, low_memory=False)
        v_df = v_df.replace({"label": cfg["label_dict"]}).fillna(0)
        v_arr = v_df.to_numpy()
        val_data = np.append(val_data, v_arr, axis=0)

    min_train = np.min(train_data[:, 1:-1])
    max_train = np.max(train_data[:, 1:-1])

    train_data[:, 1:-1] = 2 * (train_data[:, 1:-1] - min_train) / (max_train - min_train) - 1
    val_data[:, 1:-1] = 2 * (val_data[:, 1:-1] - min_train) / (max_train - min_train) - 1

    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print("Min/ Max values in training data: [{:.2f}, {:.2f}]".format(
        np.min(train_data[:, 1:-1]), np.max(train_data[:, 1:-1])
    ))
    print("Min/ Max values in validation data: [{:.2f}, {:.2f}]".format(
        np.min(val_data[:, 1:-1]), np.max(val_data[:, 1:-1])
    ))

    train_dataset = InertialDataset(
        train_data,
        cfg["dataset"]["window_size"],
        cfg["dataset"]["window_overlap"],
        model=cfg["name"],
    )
    test_dataset = InertialDataset(
        val_data,
        cfg["dataset"]["window_size"],
        cfg["dataset"]["window_overlap"],
        model=cfg["name"],
    )

    label_dict = cfg.get("label_dict", None)
    if label_dict is not None:
        class_id2name = {v: k for k, v in label_dict.items()}
    else:
        class_id2name = None

    num_workers = int(cfg.get("loader", {}).get("num_workers", 4))
    if os.name == "nt":
        num_workers = 0
    persistent_workers = (num_workers > 0) and (os.name != "nt")

    if cfg["name"] in ["shallow_deepconvlstm", "deepconvcontext"]:
        print("DID NOT SHUFFLE (model {})".format(cfg["name"]))
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["loader"]["train_batch_size"],
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_reset_seed,
            generator=rng_generator,
            persistent_workers=persistent_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["loader"]["train_batch_size"],
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_reset_seed,
            generator=rng_generator,
            persistent_workers=persistent_workers,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["loader"]["test_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_reset_seed,
        generator=rng_generator,
        persistent_workers=persistent_workers,
    )

    if cfg["name"] != "deepconvlstm":
        raise ValueError(f"Unsupported model name '{cfg['name']}'")

    conv_type = cfg["model"].get("conv_type", "standard")
    conv_rank = cfg["model"].get("conv_rank", 8)
    conv_mlp_hidden_dim = cfg["model"].get("conv_mlp_hidden_dim", 32)

    net = DeepConvLSTM(
        train_dataset.channels,
        train_dataset.classes,
        train_dataset.window_size,
        cfg["model"]["conv_kernels"],
        cfg["model"]["conv_kernel_size"],
        cfg["model"]["lstm_units"],
        cfg["model"]["lstm_layers"],
        cfg["model"]["dropout"],
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        apply_gamma=apply_gamma,
        gamma_type=gamma_type,
        conv_type=conv_type,
        conv_rank=conv_rank,
        conv_mlp_hidden_dim=conv_mlp_hidden_dim,
    )

    print(
        "Number of learnable parameters for DeepConvLSTM: {}".format(
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        )
    )
    print(net)

    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    if gpu is not None:
        net = net.to(gpu)

    criterion = nn.CrossEntropyLoss()

    if cfg["train_cfg"].get("weighted_loss", False):
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(train_dataset.labels), y=train_dataset.labels
        )
        print("Class weights: {}".format(class_weights))
        w_tensor = torch.FloatTensor(class_weights)
        if gpu is not None:
            w_tensor = w_tensor.to(gpu)
        criterion = nn.CrossEntropyLoss(weight=w_tensor)

    opt = torch.optim.Adam(
        net.parameters(),
        lr=cfg["train_cfg"]["lr"],
        weight_decay=cfg["train_cfg"]["weight_decay"],
    )
    scheduler = None
    if cfg["train_cfg"]["lr_step"] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=cfg["train_cfg"]["lr_step"],
            gamma=cfg["train_cfg"]["lr_decay"],
        )

    start_epoch = 0
    best_f1 = -1.0
    best_epoch = -1
    best_v_mAP_vec = None

    if resume is not None and resume != "":
        if os.path.isfile(resume):
            print(f"Loading checkpoint '{resume}'")
            checkpoint = torch.load(resume, map_location="cpu")
            start_epoch = checkpoint.get("epoch", 0)
            net.load_state_dict(checkpoint["state_dict"])
            opt.load_state_dict(checkpoint["optimizer"])
            best_f1 = checkpoint.get("best_f1", -1.0)
            print(f"Loaded checkpoint (epoch {start_epoch}, best_f1 {best_f1:.2f})")
        else:
            print(f"No checkpoint found at '{resume}'")
    else:
        net = init_weights(net, cfg["train_cfg"]["weight_init"])

    det_eval = ANETdetection(
        cfg["dataset"]["json_anno"],
        "validation",
        tiou_thresholds=cfg["dataset"]["tiou_thresholds"],
    )

    t_losses = []
    v_losses = []
    v_mAP_all = []

    v_preds_out = None
    v_gt_out = None

    mr_cfg = cfg.get("train_cfg", {}).get("multirate_val_lengths", None)
    if mr_cfg is not None:
        multirate_lengths = [int(x) for x in mr_cfg]
    else:
        multirate_lengths = _infer_multirate_lengths_from_model(net)

    base_len = int(cfg["dataset"]["window_size"])
    multirate_lengths = [int(x) for x in multirate_lengths if int(x) > 0]
    multirate_lengths = sorted(set(multirate_lengths), reverse=True)

    save_per_class_multirate = bool(cfg.get("train_cfg", {}).get("multirate_save_per_class", True))

    for epoch in range(start_epoch, cfg["train_cfg"]["epochs"]):
        net.train()
        train_epoch_losses = []

        # -------------------------------
        # NEW: init per-epoch branch usage
        # -------------------------------
        model_ref = net.module if hasattr(net, "module") else net
        train_rate_counts = {}
        if hasattr(model_ref, "branches"):
            try:
                train_rate_counts = {int(k): 0 for k in model_ref.branches.keys()}
            except Exception:
                train_rate_counts = {}
        elif hasattr(model_ref, "supported_sample_rates"):
            try:
                train_rate_counts = {int(k): 0 for k in model_ref.supported_sample_rates}
            except Exception:
                train_rate_counts = {}

        for _, (inputs, targets) in enumerate(train_loader):
            opt.zero_grad()
            if gpu is not None:
                inputs = inputs.to(gpu)
                targets = targets.to(gpu)

            outputs = net(inputs)

            # -----------------------------------------
            # NEW: count which branch this batch used
            # -----------------------------------------
            if train_rate_counts:
                hz = getattr(model_ref, "last_sample_rate", None)
                if hz in train_rate_counts:
                    train_rate_counts[hz] += int(inputs.size(0))

            loss = criterion(outputs, targets.long())
            loss.backward()
            opt.step()

            train_epoch_losses.append(loss.item())

        t_losses.append(float(np.nanmean(train_epoch_losses)))

        # Create a pretty usage string once per epoch
        train_branch_usage_msg = None
        if train_rate_counts:
            total = sum(train_rate_counts.values())
            if total > 0:
                order = [50, 25, 12, 6]
                order = [h for h in order if h in train_rate_counts] + [h for h in sorted(train_rate_counts) if h not in order]
                train_branch_usage_msg = "TRAIN BRANCH USAGE: " + ", ".join(
                    f"{h}hz {100.0 * train_rate_counts[h] / total:.1f}% ({train_rate_counts[h]})"
                    for h in order
                )

        # -------------------------
        # Base validation (UNCHANGED)
        # -------------------------
        net.eval()
        with torch.no_grad():
            val_epoch_losses = []
            preds = np.array([], dtype=np.int64)
            gt = np.array([], dtype=np.int64)

            for _, (inputs, targets) in enumerate(test_loader):
                if gpu is not None:
                    inputs = inputs.to(gpu)
                    targets = targets.to(gpu)

                outputs = net(inputs)
                loss = criterion(outputs, targets.long())
                val_epoch_losses.append(loss.item())

                batch_preds = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
                batch_gt = targets.cpu().numpy().flatten()

                preds = np.concatenate((preds, batch_preds))
                gt = np.concatenate((gt, batch_gt))

        v_losses.append(float(np.nanmean(val_epoch_losses)))
        v_preds_out = preds
        v_gt_out = gt

        conf_mat = confusion_matrix(v_gt_out, v_preds_out, normalize="true")
        v_acc = conf_mat.diagonal()

        v_prec = precision_score(v_gt_out, v_preds_out, average=None, zero_division=1)
        v_rec = recall_score(v_gt_out, v_preds_out, average=None, zero_division=1)
        v_f1 = f1_score(v_gt_out, v_preds_out, average=None, zero_division=1)

        avg_acc = float(np.nanmean(v_acc) * 100.0)
        avg_prec = float(np.nanmean(v_prec) * 100.0)
        avg_rec = float(np.nanmean(v_rec) * 100.0)
        avg_f1 = float(np.nanmean(v_f1) * 100.0)

        unseg_preds, _ = unwindow_inertial_data(
            val_data,
            test_dataset.ids,
            v_preds_out,
            cfg["dataset"]["window_size"],
            cfg["dataset"]["window_overlap"],
        )
        v_segments = convert_samples_to_segments(
            val_data[:, 0], unseg_preds, cfg["dataset"]["sampling_rate"]
        )

        if epoch == (cfg["train_cfg"]["epochs"] - 1):
            v_results = pd.DataFrame(
                {
                    "video_id": v_segments["video-id"],
                    "t_start": v_segments["t-start"].tolist(),
                    "t_end": v_segments["t-end"].tolist(),
                    "label": v_segments["label"].tolist(),
                    "score": v_segments["score"].tolist(),
                }
            )
            unproc_dir = os.path.join(ckpt_folder, "unprocessed_results")
            mkdir_if_missing(unproc_dir)
            np.save(os.path.join(unproc_dir, f"v_preds_{split_name}.npy"), v_preds_out)
            np.save(os.path.join(unproc_dir, f"v_gt_{split_name}.npy"), v_gt_out)
            v_results.to_csv(
                os.path.join(unproc_dir, f"v_seg_{split_name}.csv"),
                index=False,
            )

        v_mAP_vec, _ = det_eval.evaluate(v_segments)
        v_mAP_all.append(v_mAP_vec)

        block1 = "Epoch: [{:03d}/{:03d}]".format(
            epoch + 1, cfg["train_cfg"]["epochs"]
        )
        block2 = "TRAINING:\tavg. loss {:4.4f}".format(t_losses[-1])
        block3 = "VALIDATION:\tavg. loss {:4.4f}".format(v_losses[-1])

        if v_mAP_vec is None or len(v_mAP_vec) == 0 or np.all(np.isnan(v_mAP_vec)):
            block4 = "\t\tAvg. mAP    n/a"
        else:
            avg_map = float(np.nanmean(v_mAP_vec) * 100.0)
            block4 = "\t\tAvg. mAP {:4.2f} (%)".format(avg_map)
            for tiou, tiou_mAP in zip(
                cfg["dataset"]["tiou_thresholds"], v_mAP_vec
            ):
                block4 += " mAP@{:.1f} {:4.2f} (%)".format(
                    float(tiou), float(tiou_mAP * 100.0)
                )

        block4 += " Acc {:4.2f} (%) Prec {:4.2f} (%) Rec {:4.2f} (%) F1 {:4.2f} (%)".format(
            avg_acc, avg_prec, avg_rec, avg_f1
        )

        print(block1)
        print(block2)
        if train_branch_usage_msg is not None:
            print(train_branch_usage_msg)
        print(block3)
        print(block4)

        if run is not None:
            run["train/loss"].log(t_losses[-1])
            run["val/loss"].log(v_losses[-1])
            if v_mAP_vec is not None and len(v_mAP_vec) > 0 and not np.all(
                np.isnan(v_mAP_vec)
            ):
                run["val/mAP"].log(float(np.nanmean(v_mAP_vec) * 100.0))
            run["val/acc_mean"].log(avg_acc)
            run["val/prec_mean"].log(avg_prec)
            run["val/rec_mean"].log(avg_rec)
            run["val/f1_mean"].log(avg_f1)

        # ------------------------------------------------------------
        # EXTRA: multi-rate validation metrics (25/12/6) using resampling
        # ------------------------------------------------------------
        if multirate_lengths:
            for L in multirate_lengths:
                L = int(L)
                if L == base_len:
                    continue

                v_losses_L, v_preds_L, v_gt_L = validate_one_epoch_resampled(
                    test_loader, net, criterion, target_len=L, gpu=gpu
                )

                conf_L = confusion_matrix(v_gt_L, v_preds_L, normalize="true")
                v_acc_L = conf_L.diagonal()
                v_prec_L = precision_score(v_gt_L, v_preds_L, average=None, zero_division=1)
                v_rec_L = recall_score(v_gt_L, v_preds_L, average=None, zero_division=1)
                v_f1_L = f1_score(v_gt_L, v_preds_L, average=None, zero_division=1)

                avg_acc_L = float(np.nanmean(v_acc_L) * 100.0)
                avg_prec_L = float(np.nanmean(v_prec_L) * 100.0)
                avg_rec_L = float(np.nanmean(v_rec_L) * 100.0)
                avg_f1_L = float(np.nanmean(v_f1_L) * 100.0)

                tagL = _pretty_freq_tag_from_len(L)
                print(f"VALIDATION@{tagL}:\tavg. loss {float(np.nanmean(v_losses_L)):.4f}")
                print(
                    f"\t\tAcc {avg_acc_L:4.2f} (%) Prec {avg_prec_L:4.2f} (%) "
                    f"Rec {avg_rec_L:4.2f} (%) F1 {avg_f1_L:4.2f} (%)"
                )

                if run is not None:
                    run[f"val_{tagL}/loss"].log(float(np.nanmean(v_losses_L)))
                    run[f"val_{tagL}/acc_mean"].log(avg_acc_L)
                    run[f"val_{tagL}/prec_mean"].log(avg_prec_L)
                    run[f"val_{tagL}/rec_mean"].log(avg_rec_L)
                    run[f"val_{tagL}/f1_mean"].log(avg_f1_L)

                if save_per_class_multirate:
                    num_classes_L = len(v_acc_L)
                    if class_id2name is not None and len(class_id2name) == num_classes_L:
                        class_names_L = [class_id2name[i] for i in range(num_classes_L)]
                    else:
                        class_names_L = [f"class_{i}" for i in range(num_classes_L)]

                    per_class_df_L = pd.DataFrame(
                        {
                            "split": [split_name] * num_classes_L,
                            "sampling_rate_cfg": [sampling_rate] * num_classes_L,
                            "eval_len": [L] * num_classes_L,
                            "eval_tag": [tagL] * num_classes_L,
                            "epoch": [epoch + 1] * num_classes_L,
                            "class_id": list(range(num_classes_L)),
                            "class_name": class_names_L,
                            "acc": v_acc_L,
                            "prec": v_prec_L,
                            "rec": v_rec_L,
                            "f1": v_f1_L,
                        }
                    )
                    per_class_path_L = os.path.join(
                        ckpt_folder, f"per_class_metrics_{split_name}_{tagL}.csv"
                    )
                    per_class_df_L.to_csv(per_class_path_L, index=False)

        # Save BEST (unchanged: still based on base avg_f1)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch = epoch + 1
            best_v_mAP_vec = v_mAP_vec

            ckpt_name = f"best_{split_name}.pth.tar"
            print(
                f"New best F1 {best_f1:.2f}% at epoch {best_epoch}, "
                f"saving checkpoint to {os.path.join(ckpt_folder, ckpt_name)}"
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_f1": best_f1,
                },
                False,
                file_folder=ckpt_folder,
                file_name=ckpt_name,
            )

            num_classes = len(v_acc)
            if class_id2name is not None and len(class_id2name) == num_classes:
                class_names = [class_id2name[i] for i in range(num_classes)]
            else:
                class_names = [f"class_{i}" for i in range(num_classes)]

            per_class_df = pd.DataFrame(
                {
                    "split": [split_name] * num_classes,
                    "sampling_rate": [sampling_rate] * num_classes,
                    "freq_tag": [freq_tag] * num_classes,
                    "epoch": [epoch + 1] * num_classes,
                    "class_id": list(range(num_classes)),
                    "class_name": class_names,
                    "acc": v_acc,
                    "prec": v_prec,
                    "rec": v_rec,
                    "f1": v_f1,
                }
            )
            per_class_path = os.path.join(
                ckpt_folder, f"per_class_metrics_{split_name}.csv"
            )
            per_class_df.to_csv(per_class_path, index=False)

        # Save LAST every epoch (for resume)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": opt.state_dict(),
                "best_f1": best_f1,
            },
            False,
            file_folder=ckpt_folder,
            file_name=f"last_{split_name}.pth.tar",
        )

        if scheduler is not None:
            scheduler.step()

    if best_v_mAP_vec is None:
        v_mAP_best = np.array([], dtype=float)
    else:
        v_mAP_best = np.array(best_v_mAP_vec, dtype=float)

    return t_losses, v_losses, v_mAP_best, v_preds_out, v_gt_out, net


def validate_one_epoch(loader, network, criterion, gpu=None):
    network.eval()
    losses = []
    preds = np.array([], dtype=np.int64)
    gt = np.array([], dtype=np.int64)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if gpu is not None:
                inputs = inputs.to(gpu)
                targets = targets.to(gpu)

            outputs = network(inputs)
            batch_loss = criterion(outputs, targets.long())
            losses.append(batch_loss.item())

            batch_preds = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
            batch_gt = targets.cpu().numpy().flatten()

            preds = np.concatenate((preds, batch_preds))
            gt = np.concatenate((gt, batch_gt))

    return losses, preds, gt
