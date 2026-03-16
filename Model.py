import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.step = nn.Sequential(nn.Conv2d(in_channel,out_channel,3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channel,out_channel,3, padding=1),
                                  nn.ReLU())
    def forward(self,x):
        return self.step(x)

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.layer1= DoubleConv(1,64)
        self.layer2= DoubleConv(64,128)
        self.layer3= DoubleConv(128,256)
        self.layer4= DoubleConv(256,512)

        # Decoder

        self.layer5 = DoubleConv(256+512,256)
        self.layer6 = DoubleConv(256+128,128)
        self.layer7 = DoubleConv(64+128,64)
        self.layer8 = nn.Conv2d(64,1,1)
        
        self.maxpool = nn.MaxPool2d(2)

    def forward(self,X):
        # Encoder Part
        x1= self.layer1(X)
        x1m= self.maxpool(x1)

        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
                
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)

        x4 = self.layer4(x3m)
        x4m = self.maxpool(x4)

        # Decoder Part

        x5 = nn.Upsample(scale_factor=2, mode= 'bilinear')(x4)
        x5 = torch.cat([x5,x3], dim=1)
        x5 = self.layer5(x5)

        x6 = nn.Upsample(scale_factor=2, mode= 'bilinear')(x5)
        x6 = torch.cat([x6,x2], dim=1)
        x6 = self.layer6(x6)

        x7= nn.Upsample(scale_factor=2, mode='bilinear')(x6)
        x7 = torch.cat([x7,x1],dim= 1)
        x7=self.layer7(x7)
        ret = self.layer8(x7)
        return ret
      