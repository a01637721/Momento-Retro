"""
Script de predicción para modelos de reconocimiento de acciones
"""
import torch
import argparse
import os
import sys
import pickle
import numpy as np

# Agregar paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from utils.data_loader import load_processed_data, SkeletonDataset
from models.baseline_lstm import BaselineLSTM
from models.improved_lstm import ImprovedLSTM


def predict(model_name='improved', checkpoint_path=None, video_id=None):
    """Realizar predicción en un video"""
    print(f"Predicción con modelo: {model_name}")
    
    # Cargar datos procesados
    processed_data = load_processed_data()
    if processed_data is None:
        print("Error al cargar datos procesados.")
        return
    
    class_names = processed_data['class_names']
    labels_map = processed_data['labels_map']
    reverse_labels_map = {v: k for k, v in labels_map.items()}
    
    # Si no se especifica video_id, usar uno aleatorio del test
    if video_id is None:
        test_annotations = processed_data['test']
        if len(test_annotations) == 0:
            print("No hay videos en el conjunto de test.")
            return
        import random
        random_ann = random.choice(test_annotations)
        video_id = random_ann['frame_dir']
        print(f"No se especificó video_id, usando uno aleatorio: {video_id}")
    
    # Buscar anotación del video
    all_annotations = processed_data['train'] + processed_data['val'] + processed_data['test']
    video_ann = None
    for ann in all_annotations:
        if ann['frame_dir'] == video_id:
            video_ann = ann
            break
    
    if video_ann is None:
        print(f"Error: No se encontró el video {video_id}")
        print("Videos disponibles (primeros 10):")
        for i, ann in enumerate(all_annotations[:10]):
            print(f"  - {ann['frame_dir']}")
        return
    
    # Crear dataset para este video
    dataset = SkeletonDataset([video_ann], labels_map, augment=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
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
    
    # Realizar predicción
    print(f"\nPrediciendo para video: {video_id}")
    with torch.no_grad():
        for inputs, label in data_loader:
            inputs = inputs.to(config.TRAIN_CONFIG['device'])
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            pred_class_idx = predicted.item()
            true_class_idx = label.item()
            
            # Obtener nombres de clases
            pred_class_name = reverse_labels_map[pred_class_idx]
            true_class_name = reverse_labels_map[true_class_idx]
            
            # Mostrar resultados
            print(f"\n{'='*50}")
            print(f"Video ID: {video_id}")
            print(f"{'='*50}")
            print(f"Clase verdadera: {true_class_name}")
            print(f"Clase predicha: {pred_class_name}")
            print(f"Confianza: {probabilities[0][pred_class_idx].item()*100:.2f}%")
            print(f"{'='*50}\n")
            
            # Mostrar top 3 predicciones
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)
            print("Top 3 predicciones:")
            for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                class_name = reverse_labels_map[idx.item()]
                print(f"  {i+1}. {class_name}: {prob.item()*100:.2f}%")
            
            # Resultado
            is_correct = "✓ CORRECTO" if pred_class_idx == true_class_idx else "✗ INCORRECTO"
            print(f"\nResultado: {is_correct}")
            
            return {
                'video_id': video_id,
                'true_class': true_class_name,
                'predicted_class': pred_class_name,
                'confidence': float(probabilities[0][pred_class_idx].item()),
                'is_correct': pred_class_idx == true_class_idx,
                'top3': [
                    {
                        'class': reverse_labels_map[idx.item()],
                        'probability': float(prob.item())
                    }
                    for prob, idx in zip(top3_probs, top3_indices)
                ]
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predecir acción en un video')
    parser.add_argument('--model', type=str, default='improved', choices=['baseline', 'improved'],
                        help='Modelo a usar')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Ruta al checkpoint')
    parser.add_argument('--video_id', type=str, default=None,
                        help='ID del video a predecir (frame_dir)')
    
    args = parser.parse_args()
    predict(args.model, args.checkpoint, args.video_id)

