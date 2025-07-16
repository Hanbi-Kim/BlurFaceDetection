import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# ✅ Edge Attention이 포함된 Residual Block (edge 입력 X)
class EdgeAttentionResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttentionResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        # Edge Attention: Grayscale → Conv → Sigmoid
        self.edge_attention = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # 🔍 Grayscale 변환 (채널 가중합 방식)
        with torch.no_grad():
            gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
            edge = gray  # 여기선 간단히 gray 사용, 원하면 Sobel 등도 가능

        attn = self.edge_attention(edge)  # shape: [B, in_channels, H, W]

        out = self.relu(self.bn1(self.conv1(x)))
        out = out * attn
        out = self.bn2(self.conv2(out))

        out += identity
        out = self.relu(out)
        return out


# ✅ 전체 EdgeResNet18 정의 (입력: x only)
class EdgeResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(EdgeResNet18, self).__init__()
        base = resnet18(pretrained=False)

        # stem: conv1 ~ maxpool
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        # Residual + Edge Attention blocks
        self.layer1 = EdgeAttentionResidualBlock(64)
        self.layer2 = EdgeAttentionResidualBlock(128)
        self.layer3 = EdgeAttentionResidualBlock(256)
        self.layer4 = EdgeAttentionResidualBlock(512)

        # 채널 확장 및 다운샘플링
        self.down1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.down2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        self.down3 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.down1(x)

        x = self.layer2(x)
        x = self.down2(x)

        x = self.layer3(x)
        x = self.down3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
