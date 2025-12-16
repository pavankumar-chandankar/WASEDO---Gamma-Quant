set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"

TEST_SPLIT_DIR="${TEST_SPLIT_DIR:-data/wear/multifreq_test}"

python main_multifreq.py \
  --config ./configs/deepconvlstm/wear_loso_25hz.yaml \
  --mode test_multifreq \
  --test-split-dir "$TEST_SPLIT_DIR" \
  --ckpt-6hz  checkpoints/deepconvlstm/best_6hz.pth.tar \
  --ckpt-12hz checkpoints/deepconvlstm/best_12hz.pth.tar \
  --ckpt-25hz checkpoints/deepconvlstm/best_25hz.pth.tar \
  --gamma_quant quantize_and_gamma \
  --quant_bits 4 \
  --gamma_type s_shaped \
  --apply_gamma global
