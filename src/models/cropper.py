import torch
import torch.nn as nn

class Cropper(nn.Module):
    def __init__(self, embed_size):
        super(Cropper, self).__init__()
        self.image_mlp = nn.Sequential(
            nn.AdpativeMeanPool(1,1)
            nn.Linear()
        )
