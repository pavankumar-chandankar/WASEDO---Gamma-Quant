#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"

SEED="${SEED:-1}"
TEST_SPLIT_DIR="${TEST_SPLIT_DIR:-data/wear/multifreq_test}"

GAMMA_QUANT="${GAMMA_QUANT:-quantize_and_gamma}"
QUANT_BITS="${QUANT_BITS:-4}"
GAMMA_TYPE="${GAMMA_TYPE:-s_shaped}"
APPLY_GAMMA="${APPLY_GAMMA:-global}"

OUT_DIR="checkpoints/continuous_conv"
mkdir -p "$OUT_DIR" configs/_generated

make_cfg() {
  local base_cfg="$1"
  local freq="$2"
  local out_cfg="configs/_generated/${freq}_continuous.yaml"

  python - <<PY
import yaml
base="${base_cfg}"
out="${out_cfg}"
with open(base,"r") as f:
    cfg=yaml.safe_load(f)

# Force continuous conv in BOTH possible locations
cfg.setdefault("net_cfg", {})
cfg["net_cfg"]["conv_type"] = "continuous"

if isinstance(cfg.get("model"), dict):
    cfg["model"]["conv_type"] = "continuous"

with open(out,"w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo "$out_cfg"
}

latest_best_ckpt() {
  local freq="$1"
  ls -t "logs/deepconvlstm/${freq}_702010"/*/checkpoints/best_${freq}.pth.tar | head -n 1
}

verify_continuous_ckpt() {
  local ckpt="$1"
  python - <<PY
import torch, sys
p="${ckpt}"
sd=torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
has_std = "conv1.weight" in sd
has_cont = "conv1.channel_weights" in sd
if (not has_cont) or has_std:
    print(f"[ERROR] Expected CONTINUOUS conv ckpt with conv1.channel_weights. Found conv1.weight={has_std}, conv1.channel_weights={has_cont}")
    sys.exit(1)
print("[OK] Verified CONTINUOUS conv checkpoint:", p)
PY
}

train_one() {
  local freq="$1"
  local base_cfg="$2"

  echo "============================================================"
  echo "[TRAIN] ContinuousConv (CONTINUOUS conv_type) expert: ${freq}"
  echo "Base config: ${base_cfg}"
  echo "Test split dir: ${TEST_SPLIT_DIR}"
  echo "============================================================"

  local cfg_path
  cfg_path="$(make_cfg "$base_cfg" "$freq")"

  python main_multifreq.py \
    --config "$cfg_path" \
    --mode train_expert \
    --freq-tag "$freq" \
    --test-split-dir "$TEST_SPLIT_DIR" \
    --gamma_quant "$GAMMA_QUANT" \
    --quant_bits "$QUANT_BITS" \
    --gamma_type "$GAMMA_TYPE" \
    --apply_gamma "$APPLY_GAMMA" \
    --seed "$SEED"

  local best
  best="$(latest_best_ckpt "$freq")"
  echo "[copy] ${best} -> ${OUT_DIR}/best_${freq}.pth.tar"
  cp -f "$best" "${OUT_DIR}/best_${freq}.pth.tar"

  verify_continuous_ckpt "${OUT_DIR}/best_${freq}.pth.tar"
}

train_one "25hz" "./configs/deepconvlstm/wear_loso_25hz.yaml"
train_one "12hz" "./configs/deepconvlstm/wear_loso_12hz.yaml"
train_one "6hz"  "./configs/deepconvlstm/wear_loso_6hz.yaml"

echo "============================================================"
echo "[DONE] CONTINUOUS experts saved in: ${OUT_DIR}"
echo "============================================================"
