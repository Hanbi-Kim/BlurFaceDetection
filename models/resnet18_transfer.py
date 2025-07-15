import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Transfer(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ResNet18Transfer, self).__init__()
        self.model = resnet18(pretrained=True)

        # 전체 fine-tuning 허용
        for param in self.model.parameters():
            param.requires_grad = True

        # FC 레이어 교체 + Dropout 조정
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
