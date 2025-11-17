"""
Script de evaluación para modelos de reconocimiento de acciones
"""
import torch
import torch.nn as nn
import argparse
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from utils.data_loader import get_data_loaders
from models.baseline_lstm import BaselineLSTM
from models.improved_lstm import ImprovedLSTM


def evaluate(model_name='baseline', checkpoint_path=None):
    """Evaluar modelo en conjunto de test"""
    print(f"Evaluando modelo: {model_name}")
    
    # Cargar datos
    print("Cargando datos...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(augment_train=False)
    
    if test_loader is None:
        print("Error al cargar datos.")
        return
    
    # Calcular input size
    input_size = config.NUM_KEYPOINTS * config.KEYPOINT_DIM
    num_classes = len(class_names)
    
    # Crear modelo
    if model_name == 'baseline':
        model = BaselineLSTM(input_size, num_classes)
    elif model_name == 'improved':
        model = ImprovedLSTM(input_size, num_classes)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    
    # Cargar checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, model_name, f'{model_name}_best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No se encontró el checkpoint en {checkpoint_path}")
        return
    
    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.TRAIN_CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.TRAIN_CONFIG['device'])
    model.eval()
    
    # Evaluar en test
    print("\nEvaluando en conjunto de test...")
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.TRAIN_CONFIG['device'])
            labels = labels.to(config.TRAIN_CONFIG['device'])
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    print(f"\n{'='*50}")
    print(f"Resultados en Test:")
    print(f"{'='*50}")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"{'='*50}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Visualizar confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name.upper()} Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(config.RESULTS_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\nConfusion matrix guardada en: {cm_path}")
    
    # Guardar resultados
    results = {
        'model': model_name,
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'classification_report': classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    import json
    results_path = os.path.join(config.RESULTS_DIR, f'{model_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Resultados guardados en: {results_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluar modelo de reconocimiento de acciones')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'improved'],
                        help='Modelo a evaluar')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Ruta al checkpoint (usa el mejor por defecto)')
    
    args = parser.parse_args()
    evaluate(args.model, args.checkpoint)

