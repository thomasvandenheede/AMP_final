import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class SimpleUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=4):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128) # 128 (from upconv) + 128 (from enc2 skip)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)   # 64 (from upconv) + 64 (from enc1 skip)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip1 = self.enc1(x)
        x = self.pool1(skip1)
        skip2 = self.enc2(x)
        x = self.pool2(skip2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1) # Skip connection
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1) # Skip connection
        x = self.dec1(x)
        
        return self.final_conv(x) # Output raw logits