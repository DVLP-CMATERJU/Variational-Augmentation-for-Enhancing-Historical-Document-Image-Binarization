import torch
import torch.nn as nn
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(device, n_input=3, n_output=1, size=256):
    device = device
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-4)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

#gen = build_res_unet('cpu')

#x = torch.randn((1,3,256,256))
#y = gen(x)
#print(y.shape)
#print(gen)
'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_act else nn.Identity(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
        self.gate = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x_conv = self.conv(x)
        x_skip = self.gate(x)
        return x_conv + x_skip

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResUnet(nn.Module):
    def __init__(self, in_channels, features, out_channels, **kwargs):
        super().__init__()
        self.d0 = ResBlock(in_channels, features)
        self.d1 = ResBlock(features, features*2)
        self.d2 = ResBlock(features*2, features*4)
        self.d3 = ResBlock(features*4, features*8)
        self.d4 = ResBlock(features*8, features*16)

        self.down = nn.MaxPool2d((2,2))
        self.up0 = nn.ConvTranspose2d(features*16, features*8, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(features*8, features*4, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(features*4, features*2, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(features*2, features, 4, 2, 1)

        self.u0 = ResBlock(features*16, features*8)
        self.u1 = ResBlock(features*8, features*4)
        self.u2 = ResBlock(features*4, features*2)
        self.u3 = ResBlock(features*2, features)

        self.final = nn.Conv2d(features, out_channels, 3, 1, 1)

    def forward(self, x):
        x0 = self.d0(x)
        x1 = self.d1(self.down(x0))
        x2 = self.d2(self.down(x1))
        x3 = self.d3(self.down(x2))
        x4 = self.d4(self.down(x3))

        y1 = self.u0(torch.cat([self.up0(x4), x3], 1))
        y2 = self.u1(torch.cat([self.up1(y1), x2], 1))
        y3 = self.u2(torch.cat([self.up2(y2), x1], 1))
        y4 = self.u3(torch.cat([self.up3(y3), x0], 1))

        out = self.final(y4)

        return out
'''
