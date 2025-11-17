"""
Utilidades para visualización de resultados
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_training_history(history, model_name, save_path=None):
    """Visualizar historial de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name.upper()} - Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name.upper()} - Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_keypoints(keypoints, title="Keypoints Visualization"):
    """Visualizar keypoints de un frame"""
    # keypoints shape: [V, C] donde V es número de keypoints y C es coordenadas (x, y)
    if len(keypoints.shape) == 3:
        # Si es [T, V, C], tomar el primer frame
        keypoints = keypoints[0]
    
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords, s=50, c='red', marker='o')
    
    # Conectar keypoints (estructura básica de esqueleto humano)
    # Esto es una simplificación - ajustar según el formato de keypoints
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Cabeza y hombros
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Brazos
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Piernas
    ]
    
    for start, end in connections:
        if start < len(x_coords) and end < len(x_coords):
            plt.plot([x_coords[start], x_coords[end]], 
                    [y_coords[start], y_coords[end]], 'b-', linewidth=2)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Invertir Y para que coincida con coordenadas de imagen
    plt.grid(True)
    plt.tight_layout()
    plt.show()

