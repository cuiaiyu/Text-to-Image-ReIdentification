import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F 

class ImageBackbone(nn.Module):
    def __init__(self,embed_size,model='resnet50',coco=False):
        super(ImageBackbone,self).__init__()
        if model == 'resnet101' or coco:
            self.img_backbone=models.resnet101(pretrained=True)
            self.img_backbone.fc=nn.Linear(2048,embed_size)
        elif model == 'resnet50':
            self.img_backbone=models.resnet50(pretrained=True)
            self.img_backbone.fc=nn.Linear(2048,embed_size)
            
        elif model =='resnet18':
            self.img_backbone=models.resnet18(pretrained=True)
            self.img_backbone.fc=nn.Linear(512,embed_size)
        else:
            assert False
        self.melt_layer()
        
    def extract_feature_map(self, x):
        out = self.img_backbone.conv1(x)
        out = self.img_backbone.bn1(out)
        out = self.img_backbone.relu(out)
        out = self.img_backbone.maxpool(out)
        out = self.img_backbone.layer1(out)
        out = self.img_backbone.layer2(out)
        out = self.img_backbone.layer3(out)
        out = self.img_backbone.layer4(out)
        
    def forward(self,x):
        return self.img_backbone(x)
    
    def melt_layer(self,forzen_util=7):
        ct = 0
        for child in self.img_backbone.children():
            ct += 1
            if ct < forzen_util:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True