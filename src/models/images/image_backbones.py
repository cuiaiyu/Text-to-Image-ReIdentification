# import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F 

class ImageBackbone(nn.Module):
    def __init__(self,embed_size, regional_embed_size=256, num_cut=1, model='resnet50',coco=False):
        super(ImageBackbone,self).__init__()
        self.num_cut = num_cut
        self.hidden_size = 512 if model =='resnet18' else 2048
        if model == 'resnet101' or coco:
            self.img_backbone=models.resnet101(pretrained=True)
        elif model == 'resnet50':
            self.img_backbone=models.resnet50(pretrained=True)
        elif model =='resnet18':
            self.img_backbone=models.resnet18(pretrained=True)
        else:
            assert False
        self.img_backbone.fc=nn.Linear(self.hidden_size,embed_size)
        self.melt_layer()
        
        if num_cut > 1:
            self.conv_regional = nn.Conv2d(self.hidden_size, regional_embed_size, 1)
        
    def extract_feature_map(self, x):
        out = self.img_backbone.conv1(x)
        out = self.img_backbone.bn1(out)
        out = self.img_backbone.relu(out)
        out = self.img_backbone.maxpool(out)
        out = self.img_backbone.layer1(out)
        out = self.img_backbone.layer2(out)
        out = self.img_backbone.layer3(out)
        out = self.img_backbone.layer4(out)
        return out
    
    def mlp(self, x):
        x = self.img_backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.img_backbone.fc(x)
        return x
    
    
    def regional_extract(self,x):
        x = self.conv_regional(x)
        x = self.img_backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
    def forward(self,x):
        if self.num_cut == 1:
            return self.img_backbone(x)
        else:
            out = self.extract_feature_map(x)
            global_features = self.mlp(out)
            
            H = out.size(2)
            h = max(H // self.num_cut, 1)
            cuts = []
            for start_h in range(0, H, h):
                end_h = start_h + h
                cuts.append(self.regional_extract(out[:,:,start_h:end_h])[:,None,:])
            regional_features = torch.cat(cuts,1)
            return global_features, regional_features
            
        
    
    def melt_layer(self,forzen_util=10):
        ct = 0
        for child in self.img_backbone.children():
            ct += 1
            if ct < forzen_util:
                for param in child.parameters():
                    param.requires_grad = False
            elif ct == 10 and forzen_util < 10:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
                
            