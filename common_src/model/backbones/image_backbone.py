import torch
import torch.nn as nn
import torchvision.models as models

class ImageBackbone(nn.Module):
    def __init__(self, out_channels=128, depth_bins=80):
        super().__init__()
        
        base = models.resnet18(pretrained=False)
        
        self.feature_extractor = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3, 
        )
        print(self.feature_extractor)
        
        self.conv_out = nn.Conv2d(256, out_channels, kernel_size=1)
        self.depth_head = nn.Conv2d(out_channels, depth_bins, kernel_size=1)

    def forward(self, x):
        feats = self.feature_extractor(x)       
        feats = self.conv_out(feats)          
        depth_logits = self.depth_head(feats) 
        depth_prob = torch.softmax(depth_logits, dim=1)
        return feats, depth_prob                 
