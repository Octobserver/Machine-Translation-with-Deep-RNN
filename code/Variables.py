import torch

SOS_token = "<SOS>"
EOS_token = "<EOS>"
PADD_token = "<PAD>"
SOS_index = 0
EOS_index = 1
PADD_index = 2
MAX_LENGTH = 31

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")