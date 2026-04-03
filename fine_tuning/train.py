# fine_tuning/train.py
import os
import sys
sys.path.append('../src')
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import timm
import argparse
from dataclasses import dataclass, field
from typing import List
import onnx
import onnxruntime as ort
from utils import set_seed

@dataclass
class Config:
    # Путь к данным
    data_dir: str = '../data/raw'
    
    # Параметры модели
    model_name: str = 'efficientnet_b0'
    num_classes: int = 3
    
    # Гиперпараметры обучения
    epochs: int = 10
    batch_size: int = 16
    lr: float = 0.001
    weight_decay: float = 0.0
    
    # Настройки
    img_size: int = 224
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    
    # Сохранение
    output_dir: str = './models'
    onnx_path: str = './models/best_model.onnx'
    checkpoint_path: str = './models/best_model.pth'

def get_data_loaders(cfg: Config):
    transform = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = ImageFolder(cfg.data_dir, transform=transform)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(cfg.seed))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, full_dataset.classes

def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    train_loader, val_loader, class_names = get_data_loaders(cfg)
    
    # Модель
    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)
    model = model.to(cfg.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    best_acc = 0.0
    
    print(f"Starting training {cfg.model_name}...")
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.checkpoint_path)
            print(f"Saved best model with accuracy {best_acc:.2f}%")

    print("Training complete.")
    export_to_onnx(cfg, class_names)

def export_to_onnx(cfg: Config, class_names: List[str]):
    # Загружаем лучшую модель
    model = timm.create_model(cfg.model_name, num_classes=cfg.num_classes)
    model.load_state_dict(torch.load(cfg.checkpoint_path))
    model.to(cfg.device)
    model.eval()
    
    # Создаем dummy input
    dummy_input = torch.randn(1, 3, cfg.img_size, cfg.img_size).to(cfg.device)
    
    # Экспорт
    torch.onnx.export(
        model, 
        dummy_input, 
        cfg.onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Проверка модели
    onnx_model = onnx.load(cfg.onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Model exported to {cfg.onnx_path}")
    
    # Сохраняем классы для использования в приложении
    with open(os.path.join(cfg.output_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(class_names))

if __name__ == "__main__":
    # Можно парсить аргументы командной строки или создать инстанс конфига напрямую
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='Model name from timm')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Инициализация конфига
    cfg = Config(
        model_name=args.model,
        epochs=args.epochs
    )
    
    train(cfg)