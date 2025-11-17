"""
Script de entrenamiento para modelos de reconocimiento de acciones
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
import sys
from tqdm import tqdm
import json
from datetime import datetime

# Agregar paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from utils.data_loader import get_data_loaders
from models.baseline_lstm import BaselineLSTM
from models.improved_lstm import ImprovedLSTM


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entrenar una época"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Actualizar barra de progreso
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validar modelo"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train(model_name='baseline'):
    """Función principal de entrenamiento"""
    print(f"Entrenando modelo: {model_name}")
    print(f"Device: {config.TRAIN_CONFIG['device']}")
    
    # Cargar datos
    print("Cargando datos...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(augment_train=True)
    
    if train_loader is None:
        print("Error al cargar datos. Por favor ejecuta primero la preparación de datos.")
        return
    
    print(f"Número de clases: {len(class_names)}")
    print(f"Clases: {class_names}")
    
    # Calcular input size (V*C = 17*2 = 34)
    input_size = config.NUM_KEYPOINTS * config.KEYPOINT_DIM
    num_classes = len(class_names)
    
    # Crear modelo
    if model_name == 'baseline':
        model = BaselineLSTM(input_size, num_classes)
    elif model_name == 'improved':
        model = ImprovedLSTM(input_size, num_classes)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    
    model = model.to(config.TRAIN_CONFIG['device'])
    
    # Loss y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.TRAIN_CONFIG['learning_rate'],
        weight_decay=config.TRAIN_CONFIG['weight_decay']
    )
    
    # Scheduler (solo para modelo mejorado)
    if model_name == 'improved':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None
    
    # Early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Directorio para checkpoints
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Entrenamiento
    print("\nIniciando entrenamiento...")
    for epoch in range(config.TRAIN_CONFIG['num_epochs']):
        print(f"\nÉpoca {epoch + 1}/{config.TRAIN_CONFIG['num_epochs']}")
        
        # Entrenar
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.TRAIN_CONFIG['device'])
        
        # Validar
        val_loss, val_acc = validate(model, val_loader, criterion, config.TRAIN_CONFIG['device'])
        
        # Actualizar scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Guardar historial
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names,
                'config': config.TRAIN_CONFIG
            }, checkpoint_path)
            print(f"[OK] Mejor modelo guardado (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.TRAIN_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping después de {epoch + 1} épocas")
            break
        
        # Guardar checkpoint periódico
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names,
            }, checkpoint_path)
    
    # Guardar historial
    history_path = os.path.join(config.LOG_DIR, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"\n[OK] Entrenamiento completado")
    print(f"Mejor Val Acc: {best_val_acc:.2f}%")
    print(f"Historial guardado en: {history_path}")
    print(f"Mejor modelo guardado en: {checkpoint_dir}/{model_name}_best.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo de reconocimiento de acciones')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'improved'],
                        help='Modelo a entrenar (baseline o improved)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Número de épocas (usa config por defecto si no se especifica)')
    
    args = parser.parse_args()
    
    if args.epochs:
        config.TRAIN_CONFIG['num_epochs'] = args.epochs
    
    train(args.model)

