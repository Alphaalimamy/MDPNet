import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

from modules import DPFocus
class ResNet(nn.Module):
    def __init__(self, weights=ResNet34_Weights.DEFAULT):
        super(ResNet, self).__init__()

        resnet = resnet34(weights=weights)
        self.layer_0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # Initial layers
        self.layer_1 = resnet.layer1  # Feature maps: 64 channels
        self.layer_2 = resnet.layer2  # Feature maps: 128 channels
        self.layer_3 = resnet.layer3  # Feature maps: 256 channels
        self.layer_4 = resnet.layer4  # Feature maps: 512 channels

    def forward(self, x):
        x = self.layer_0(x)  # Downscale and process input with 64 channels
        layer_1 = self.layer_1(x)  # Output: 64 channels
        layer_2 = self.layer_2(layer_1)  # Output: 128 channels
        layer_3 = self.layer_3(layer_2)  # Output: 256 channels
        layer_4 = self.layer_4(layer_3)  # Output: 512 channels

        return layer_1, layer_2, layer_3, layer_4

class MDPNet(nn.Module):
    def __init__(self, out_channels: int = 1):
        super(MDPNet, self).__init__()

        # Initialize the ResNet backbone for feature extraction
        self.backbone = ResNet()

        self.dpfocu_1 = DPFocus(68, 32)
        self.dpfocu_2 = DPFocus(136, 64)
        self.dpfocu_3 = DPFocus(272, 128)
        self.dpfocu_4 = DPFocus(512, 256)

        self.final_conv = nn.Sequential(
            nn.Conv2d(30, 16, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.final = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, groups=1, bias=False))

    def forward(self, x):
        # Extract feature maps from the backbone
        """
        layer_1: torch.Size([1, 64, 128, 128])
        layer_2: torch.Size([1, 128, 64, 64])
        layer_3: torch.Size([1, 256, 32, 32])
        layer_4: torch.Size([1, 512, 16, 16])

        Ratio = 16
        oam_1: torch.Size([1, 2, 128, 128])
        oam_2: torch.Size([1, 4, 64, 64])
        oam_3: torch.Size([1, 8, 32, 32])
        oam_4: torch.Size([1, 16, 16, 16])

        oam_4_up: torch.Size([1, 16, 32, 32])
        """
        # RESNET 34 LAYERS
        layer_1, layer_2, layer_3, layer_4 = self.backbone(x)

        # DPFocus LAYER 4
        dpfocu_4 = self.dpfocu_4(layer_4)
        dpfocu_4_up = F.interpolate(dpfocu_4, scale_factor=2, mode='bilinear', align_corners=True)
        layer_3_up_4 = torch.cat((layer_3, dpfocu_4_up), dim=1)

        # DPFocus LAYER 3
        dpfocu_3 = self.dpfocu_3(layer_3_up_4)
        dpfocu_3_up = F.interpolate(dpfocu_3, scale_factor=2, mode='bilinear', align_corners=True)
        layer_2_up_3 = torch.cat((layer_2, dpfocu_3_up), dim=1)  # [1, 136, 64, 64]

        # DPFocus LAYER 2
        dpfocu_2 = self.dpfocu_2(layer_2_up_3)
        dpfocu_2_up = F.interpolate(dpfocu_2, scale_factor=2, mode='bilinear', align_corners=True)
        layer_1_up_2 = torch.cat((layer_1, dpfocu_2_up), dim=1)  # [1, 68, 128, 128]

        # DPFocus LAYER 1
        dpfocu_1 = self.dpfocu_1(layer_1_up_2)

        dpfocu_1_up = F.interpolate(dpfocu_1, scale_factor=2, mode='bilinear', align_corners=True)
        dpfocu_2_up = F.interpolate(dpfocu_2, scale_factor=4, mode='bilinear', align_corners=True)
        dpfocu_3_up = F.interpolate(dpfocu_3, scale_factor=8, mode='bilinear', align_corners=True)
        dpfocu_4_up = F.interpolate(dpfocu_4, scale_factor=16, mode='bilinear', align_corners=True)  # [1, 16, 256, 256]
        output = torch.cat((dpfocu_1_up, dpfocu_2_up, dpfocu_3_up, dpfocu_4_up), dim=1)

        # PASS THROUGH FINAL CONVOLUTION LAYER Conv -> BN -> ReLU
        final_conv = self.final_conv(output) + dpfocu_4_up

        final = self.final(final_conv)
        return final


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model = MDPNet(1)
    img = torch.randn(1, 3, 256, 256)
    print(f"Count Parameters: {count_parameters(model)}")

    # Forward pass
    output = model(img)
    print(f"Output shape: {output.shape}")

    from thop import profile
    flops, params = profile(model, inputs=(img,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs, Parameters: {params / 1e6:.2f} M")

