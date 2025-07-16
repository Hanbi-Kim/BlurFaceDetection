import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from datetime import datetime

# 모델 불러오기
from dataset import BlurredImageDataset
from models import SimpleCNN, ShallowCNN, VGG16, ResNet18Transfer, ResNet18_4ch, ResNet18_4ch_CBAM, ResNet18_EdgeAttention, EdgeResNet18
from config import *

def get_model(model_name):
    if model_name == 'simple':
        return SimpleCNN(len(CLASS_NAMES))
    elif model_name == 'shallow':
        return ShallowCNN(len(CLASS_NAMES))
    elif model_name == 'vgg16':
        return VGG16(len(CLASS_NAMES))
    elif model_name == 'resnet18':
        return ResNet18Transfer(len(CLASS_NAMES))
    elif model_name == 'resnet18_4ch':
        return ResNet18_4ch(len(CLASS_NAMES))
    elif model_name == 'resnet18_4ch_cbam':
        return ResNet18_4ch_CBAM(len(CLASS_NAMES))
    elif model_name == 'resnet18_edgeattn':
        return ResNet18_EdgeAttention(len(CLASS_NAMES))
    elif model_name == 'edge_resnet':
        return EdgeResNet18(len(CLASS_NAMES))
    else:
        raise ValueError("Unknown model name")


def train(model_name='simple'):
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    # Data Loaders (edge 사용 X)
    train_loader = DataLoader(BlurredImageDataset('datasets/train', augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BlurredImageDataset('datasets/val', augment=False), batch_size=BATCH_SIZE)

    # Model, criterion, optimizer
    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Logging setup
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_filename, "w") as f:
        f.write("epoch,loss,train_acc,val_acc\n")

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # ✅ 단일 입력
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100

        # Logging
        print(f"[{model_name}] Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        with open(log_filename, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f},{train_acc:.2f},{val_acc:.2f}\n")

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"models/weights/best_{model_name}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simple', choices=[
        'simple', 'shallow', 'vgg16', 'resnet18',
        'resnet18_4ch', 'resnet18_4ch_cbam', 'resnet18_edgeattn', 'edge_resnet'
    ])
    args = parser.parse_args()

    train(args.model)
