from torchvision.models import resnet18
import torch
import torch.nn as nn

class ResNet18_4ch(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = resnet18(pretrained=True)

        # 🔹 conv1 수정: 입력 채널 수 3 → 4
        pretrained_weight = self.model.conv1.weight.data
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv1.weight[:, :3] = pretrained_weight  # 기존 RGB 가중치 복사
            new_conv1.weight[:, 3] = pretrained_weight[:, 0]  # 4채널은 red 채널로 초기화

        self.model.conv1 = new_conv1

        # 🔹 분류기 수정
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
