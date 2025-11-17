"""
Script para comparar modelos baseline y mejorado
"""
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# Agregar paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def compare_models():
    """Comparar resultados de modelos baseline y improved"""
    print("Comparando modelos...")
    
    # Cargar historiales de entrenamiento
    baseline_history_path = os.path.join(config.LOG_DIR, 'baseline_history.json')
    improved_history_path = os.path.join(config.LOG_DIR, 'improved_history.json')
    
    baseline_history = None
    improved_history = None
    
    if os.path.exists(baseline_history_path):
        with open(baseline_history_path, 'r') as f:
            baseline_history = json.load(f)
    
    if os.path.exists(improved_history_path):
        with open(improved_history_path, 'r') as f:
            improved_history = json.load(f)
    
    if baseline_history is None and improved_history is None:
        print("Error: No se encontraron historiales de entrenamiento.")
        print("Por favor entrena los modelos primero.")
        return
    
    # Cargar resultados de evaluación
    baseline_results_path = os.path.join(config.RESULTS_DIR, 'baseline_results.json')
    improved_results_path = os.path.join(config.RESULTS_DIR, 'improved_results.json')
    
    baseline_results = None
    improved_results = None
    
    if os.path.exists(baseline_results_path):
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
    
    if os.path.exists(improved_results_path):
        with open(improved_results_path, 'r') as f:
            improved_results = json.load(f)
    
    # Comparar resultados de test
    print("\n" + "="*60)
    print("COMPARACIÓN DE MODELOS")
    print("="*60)
    
    if baseline_results:
        print(f"\nBaseline Model:")
        print(f"  Test Accuracy: {baseline_results['test_acc']:.2f}%")
        print(f"  Test Loss: {baseline_results['test_loss']:.4f}")
    
    if improved_results:
        print(f"\nImproved Model:")
        print(f"  Test Accuracy: {improved_results['test_acc']:.2f}%")
        print(f"  Test Loss: {improved_results['test_loss']:.4f}")
    
    if baseline_results and improved_results:
        improvement = improved_results['test_acc'] - baseline_results['test_acc']
        print(f"\nMejora del modelo mejorado: {improvement:+.2f}%")
    
    print("="*60)
    
    # Visualizar curvas de entrenamiento
    if baseline_history or improved_history:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax = axes[0, 0]
        if baseline_history:
            ax.plot(baseline_history['train_loss'], label='Baseline Train', linestyle='--')
            ax.plot(baseline_history['val_loss'], label='Baseline Val', linestyle='-')
        if improved_history:
            ax.plot(improved_history['train_loss'], label='Improved Train', linestyle='--')
            ax.plot(improved_history['val_loss'], label='Improved Val', linestyle='-')
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        
        # Accuracy
        ax = axes[0, 1]
        if baseline_history:
            ax.plot(baseline_history['train_acc'], label='Baseline Train', linestyle='--')
            ax.plot(baseline_history['val_acc'], label='Baseline Val', linestyle='-')
        if improved_history:
            ax.plot(improved_history['train_acc'], label='Improved Train', linestyle='--')
            ax.plot(improved_history['val_acc'], label='Improved Val', linestyle='-')
        ax.set_xlabel('Época')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(True)
        
        # Comparación de accuracy
        ax = axes[1, 0]
        models = []
        train_accs = []
        val_accs = []
        test_accs = []
        
        if baseline_history:
            models.append('Baseline')
            train_accs.append(baseline_history['train_acc'][-1] if baseline_history['train_acc'] else 0)
            val_accs.append(baseline_history['val_acc'][-1] if baseline_history['val_acc'] else 0)
            test_accs.append(baseline_results['test_acc'] if baseline_results else 0)
        
        if improved_history:
            models.append('Improved')
            train_accs.append(improved_history['train_acc'][-1] if improved_history['train_acc'] else 0)
            val_accs.append(improved_history['val_acc'][-1] if improved_history['val_acc'] else 0)
            test_accs.append(improved_results['test_acc'] if improved_results else 0)
        
        x = np.arange(len(models))
        width = 0.25
        ax.bar(x - width, train_accs, width, label='Train', alpha=0.8)
        ax.bar(x, val_accs, width, label='Val', alpha=0.8)
        ax.bar(x + width, test_accs, width, label='Test', alpha=0.8)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, axis='y')
        
        # Comparación de loss
        ax = axes[1, 1]
        train_losses = []
        val_losses = []
        test_losses = []
        
        if baseline_history:
            train_losses.append(baseline_history['train_loss'][-1] if baseline_history['train_loss'] else 0)
            val_losses.append(baseline_history['val_loss'][-1] if baseline_history['val_loss'] else 0)
            test_losses.append(baseline_results['test_loss'] if baseline_results else 0)
        
        if improved_history:
            train_losses.append(improved_history['train_loss'][-1] if improved_history['train_loss'] else 0)
            val_losses.append(improved_history['val_loss'][-1] if improved_history['val_loss'] else 0)
            test_losses.append(improved_results['test_loss'] if improved_results else 0)
        
        ax.bar(x - width, train_losses, width, label='Train', alpha=0.8)
        ax.bar(x, val_losses, width, label='Val', alpha=0.8)
        ax.bar(x + width, test_losses, width, label='Test', alpha=0.8)
        ax.set_ylabel('Loss')
        ax.set_title('Model Loss Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        comparison_path = os.path.join(config.RESULTS_DIR, 'model_comparison.png')
        plt.savefig(comparison_path)
        print(f"\nComparación visual guardada en: {comparison_path}")
    
    # Guardar resumen de comparación
    comparison_summary = {
        'baseline': {
            'test_acc': baseline_results['test_acc'] if baseline_results else None,
            'test_loss': baseline_results['test_loss'] if baseline_results else None,
        },
        'improved': {
            'test_acc': improved_results['test_acc'] if improved_results else None,
            'test_loss': improved_results['test_loss'] if improved_results else None,
        }
    }
    
    if baseline_results and improved_results:
        comparison_summary['improvement'] = {
            'accuracy_gain': improved_results['test_acc'] - baseline_results['test_acc'],
            'loss_reduction': baseline_results['test_loss'] - improved_results['test_loss']
        }
    
    summary_path = os.path.join(config.RESULTS_DIR, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    print(f"Resumen de comparación guardado en: {summary_path}")


if __name__ == '__main__':
    compare_models()

