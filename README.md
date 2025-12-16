# WASEDO — Gamma-Quant (Multi-Frequency ContConv-LSTM)

This repository contains:
- Multi-frequency DeepConvLSTM experts (6 Hz / 12 Hz / 25 Hz)
- Cascade evaluation across frequencies
- Gamma-Quant inference (quantization + gamma)

## Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./scripts/deepconvlstm.sh

./scripts/continuous_conv.sh

./scripts/test_deepconvlstm.sh

./scripts/test_continuous_conv.sh

