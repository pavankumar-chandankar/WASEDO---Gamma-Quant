import torch
from models.multi_freq_deepconvlstm import build_multifreq_from_single_cfg
from models.multi_freq_deepconvlstm import MultiFreqDeepConvLSTM

from your_config_loader import load_config
from my_freq_mapping import FREQ_FOR_CLASS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg_25 = load_config("./configs/deepconvlstm/wear_70_20_10_25hz.yaml")

multi_model: MultiFreqDeepConvLSTM = build_multifreq_from_single_cfg(
    cfg_25,
    freq_for_class=FREQ_FOR_CLASS,
    conv_type="continuous",
    conv_rank=8,
)

multi_model.to(device)

ckpt_25 = torch.load("logs/deepconvlstm/25hz/best_70_20_10.pth.tar", map_location=device)
ckpt_12 = torch.load("logs/deepconvlstm/12hz/best_70_20_10.pth.tar", map_location=device)
ckpt_6  = torch.load("logs/deepconvlstm/6hz/best_70_20_10.pth.tar",  map_location=device)

multi_model.model_25.load_state_dict(ckpt_25["state_dict"])
multi_model.model_12.load_state_dict(ckpt_12["state_dict"])
multi_model.model_6.load_state_dict(ckpt_6["state_dict"])

multi_model.eval()
multi_model.reset_history()
