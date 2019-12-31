#from .images import build_image_encoder
#from .texts import build_text_encoder
from .texts.gru_backbone import BiGRUBackbone as CaptionBackbone
from .images.image_backbones import ImageBackbone
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, embed_size, image_opt="resnet50", caption_opt="bigru", cap_embed_type="sent"):
        super(Model, self).__init__()
        self.img_backbone = ImageBackbone(embed_size, image_opt)
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
    
class BaseDualEncoder(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super(BaseDualEncoder, self).__init__()
        self.img_encoder = image_encoder
        self.cap_encoder = text_encoder

    def encode(self, x):
        """
        input:
            - x: x can be either images or captions
        """
        dim = len(x.size())
        if dim == 4:
            return self.img_encoder.encode(x)
        elif dim == 3:
            return self.cap_encoder.encode(x)
        else:
            assert False, "Not Implemented"
    
    def crop(self, z, mask=None):
        """
        for image, crop the ROI in the fetaure map and output feature vector 
        for caption, crop the interested NPs
        * if not mask given, return global feature
        """
        dim = len(z.size())
        if dim == 4:
            return self.img_encoder.crop(z, mask)
        elif dim == 3:
            return self.cap_encoder.crop(z, mask)
        else:
            assert False, "Not Implemented"
    
    def forward(self, x, mask=None):
        """
        return global features and local features
        """
        feature_maps = self.encode(x)
        global_features = self.crop(feature_maps)
        local_features = self.crop(feature_maps, mask)
        return global_features, local_features


def build_dual_encoder(opt):
    image_encoder = build_image_encoder(opt)
    text_encoder = build_text_encoder(opt)
    return BaseDualEncoder(image_encoder=image_encoder, text_encoder=text_encoder)




        

    
    
            


