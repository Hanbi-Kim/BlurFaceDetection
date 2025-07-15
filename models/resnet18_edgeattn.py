import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EdgeAwareChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels + 1, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, edge):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)          # [B,C]
        edge_avg = edge.mean(dim=[2,3])                # [B,1]
        combined = torch.cat([avg_out, edge_avg], dim=1)
        attn = self.fc(combined).view(b, c, 1, 1)
        return x * attn

class ResNet18_EdgeAttention(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.attn = EdgeAwareChannelAttention(64)
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

        # conv1을 수정하지 않음 (입력은 3채널 그대로)

    def forward(self, x):
        edge = self.extract_edge(x)
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # bn1
        x = self.backbone[2](x)  # relu
        x = self.backbone[3](x)  # maxpool
        x = self.attn(x, edge)   # edge-aware attention
        x = self.backbone[4](x)  # layer1
        x = self.backbone[5](x)  # layer2
        x = self.backbone[6](x)  # layer3
        x = self.backbone[7](x)  # layer4
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def extract_edge(self, x):
        # Laplacian edge on grayscale image
        gray = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
        lap_filter = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=x.device)
        edge = F.conv2d(gray, lap_filter, padding=1)
        return edge
