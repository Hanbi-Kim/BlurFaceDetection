# models/resnet18_4ch_cbam.py

import torch
import torch.nn as nn
import torchvision.models as models
from models.cbam import CBAM

class ResNet18_4ch_CBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18_4ch_CBAM, self).__init__()
        original = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 4채널 입력
        self.bn1 = original.bn1
        self.relu = original.relu
        self.maxpool = original.maxpool

        # 각 레이어에 CBAM 추가
        self.layer1 = nn.Sequential(original.layer1, CBAM(64))
        self.layer2 = nn.Sequential(original.layer2, CBAM(128))
        self.layer3 = nn.Sequential(original.layer3, CBAM(256))
        self.layer4 = nn.Sequential(original.layer4, CBAM(512))

        self.avgpool = original.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
