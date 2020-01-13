#from .images import build_image_encoder
#from .texts import build_text_encoder
from .texts.gru_backbone import BiGRUBackbone as CaptionBackbone
from .images.image_backbones import ImageBackbone
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.ReLU(),
            nn.Linear(n_in, n_out)
        )
    def forward(self, x):
        return self.mlp(x)
    
class Model(nn.Module):
    def __init__(self, embed_size, image_opt="resnet50", caption_opt="bigru", cap_embed_type="sent", img_num_cut=1, regional_embed_size=256):
        super(Model, self).__init__()
        self.img_backbone = ImageBackbone(embed_size, regional_embed_size=256, num_cut=img_num_cut, model=image_opt)
        self.cap_backbone = CaptionBackbone(embed_size=embed_size, caption_opt=caption_opt,cap_embed_type=cap_embed_type)
        
    def forward(self,x):
        if len(x.size()) == 4:
            x = self.img_backbone(x)
        elif len(x.size()) == 2 or len(x.size()) == 3:
            x = self.cap_backbone(x)
        else:
            assert False
        return x      
      
    
    def img_frezee_layer(self, num_layer_to_frezee):
        self.img_backbone.melt_layer(num_layer_to_frezee)
    


        

    
    
            


