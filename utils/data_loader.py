"""
Carga y preprocesamiento de datos de esqueletos UCF101
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import sys

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SkeletonDataset(Dataset):
    """Dataset para esqueletos 2D de UCF101"""
    
    def __init__(self, annotations, labels_map, augment=False):
        """
        Args:
            annotations: Lista de anotaciones de esqueletos
            labels_map: Diccionario que mapea labels a índices
            augment: Si True, aplica data augmentation
        """
        self.annotations = annotations
        self.labels_map = labels_map
        self.augment = augment
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Obtener keypoints y scores
        keypoints = ann['keypoint']  # Shape: [M, T, V, C]
        keypoint_scores = ann['keypoint_score']  # Shape: [M, T, V]
        
        # Procesar keypoints: seleccionar máximo MAX_PERSONS personas
        M, T, V, C = keypoints.shape
        
        # Seleccionar la persona con más keypoints visibles (o la primera)
        if M > config.MAX_PERSONS:
            # Calcular visibilidad promedio por persona
            visibility = np.mean(keypoint_scores, axis=(1, 2))  # [M]
            top_persons = np.argsort(visibility)[-config.MAX_PERSONS:][::-1]
            keypoints = keypoints[top_persons]
            keypoint_scores = keypoint_scores[top_persons]
            M = config.MAX_PERSONS
        
        # Padding/truncate frames
        if T > config.MAX_FRAMES:
            keypoints = keypoints[:, :config.MAX_FRAMES, :, :]
            keypoint_scores = keypoint_scores[:, :config.MAX_FRAMES, :]
            T = config.MAX_FRAMES
        elif T < config.MAX_FRAMES:
            # Padding con ceros
            pad_length = config.MAX_FRAMES - T
            keypoints = np.pad(keypoints, ((0, 0), (0, pad_length), (0, 0), (0, 0)), mode='constant')
            keypoint_scores = np.pad(keypoint_scores, ((0, 0), (0, pad_length), (0, 0)), mode='constant')
            T = config.MAX_FRAMES
        
        # Combinar personas: promediar o concatenar
        # Estrategia: promediar keypoints de múltiples personas
        if M > 1:
            # Promediar keypoints ponderados por scores
            keypoints_weighted = keypoints * keypoint_scores[:, :, :, np.newaxis]
            keypoints_avg = np.sum(keypoints_weighted, axis=0) / (np.sum(keypoint_scores, axis=0, keepdims=True) + 1e-8)
            keypoints = keypoints_avg[np.newaxis, :, :, :]
        
        # Aplicar data augmentation si está habilitado
        if self.augment and config.AUGMENT_CONFIG['enable']:
            keypoints = self._augment_keypoints(keypoints[0])  # [T, V, C]
            keypoints = keypoints[np.newaxis, :, :, :]
        
        # Convertir a tensor: [M, T, V*C] -> [T, V*C] (flatten keypoints)
        keypoints = keypoints[0]  # [T, V, C]
        keypoints_flat = keypoints.reshape(T, -1)  # [T, V*C]
        
        # Normalizar keypoints (centrar y escalar)
        keypoints_flat = self._normalize_keypoints(keypoints_flat)
        
        # Obtener label
        label = self.labels_map[ann['label']]
        
        return torch.FloatTensor(keypoints_flat), torch.LongTensor([label])[0]
    
    def _normalize_keypoints(self, keypoints):
        """Normalizar keypoints: centrar y escalar"""
        # keypoints shape: [T, V*C]
        # Separar x e y
        num_keypoints = config.NUM_KEYPOINTS
        x_coords = keypoints[:, :num_keypoints]
        y_coords = keypoints[:, num_keypoints:]
        
        # Centrar (restar media)
        x_mean = np.mean(x_coords[x_coords != 0])
        y_mean = np.mean(y_coords[y_coords != 0])
        
        x_coords = x_coords - x_mean
        y_coords = y_coords - y_mean
        
        # Escalar (dividir por desviación estándar)
        x_std = np.std(x_coords[x_coords != 0]) + 1e-8
        y_std = np.std(y_coords[y_coords != 0]) + 1e-8
        
        x_coords = x_coords / x_std
        y_coords = y_coords / y_std
        
        # Reconstruir
        keypoints_norm = np.concatenate([x_coords, y_coords], axis=1)
        return keypoints_norm
    
    def _augment_keypoints(self, keypoints):
        """Aplicar data augmentation a keypoints"""
        # keypoints shape: [T, V, C]
        aug_config = config.AUGMENT_CONFIG
        
        # Ruido gaussiano
        if aug_config['noise_std'] > 0:
            noise = np.random.normal(0, aug_config['noise_std'], keypoints.shape)
            keypoints = keypoints + noise
        
        # Escalado
        if aug_config['scale_range']:
            scale = np.random.uniform(aug_config['scale_range'][0], aug_config['scale_range'][1])
            keypoints = keypoints * scale
        
        # Rotación (simple rotación 2D)
        if aug_config['rotation_range']:
            angle = np.random.uniform(
                np.radians(aug_config['rotation_range'][0]),
                np.radians(aug_config['rotation_range'][1])
            )
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            keypoints = keypoints @ rotation_matrix.T
        
        return keypoints


def load_skeleton_data(skeleton_file):
    """Cargar datos de esqueletos desde archivo pickle"""
    print(f"Cargando datos desde {skeleton_file}...")
    with open(skeleton_file, 'rb') as f:
        data = pickle.load(f)
    
    return data['split'], data['annotations']


def filter_classes(annotations, selected_classes, class_names):
    """Filtrar anotaciones por clases seleccionadas"""
    # Crear mapeo de labels
    labels_map = {class_name: idx for idx, class_name in enumerate(selected_classes)}
    
    # Filtrar anotaciones
    filtered_annotations = []
    for ann in annotations:
        # Obtener nombre de clase desde frame_dir o label
        # Asumimos que el label es un índice que corresponde a class_names
        label_idx = ann['label']
        if label_idx < len(class_names):
            class_name = class_names[label_idx]
            if class_name in selected_classes:
                # Actualizar label al nuevo índice
                ann_copy = ann.copy()
                ann_copy['label'] = class_name
                filtered_annotations.append(ann_copy)
    
    print(f"Filtradas {len(filtered_annotations)} anotaciones de {len(selected_classes)} clases")
    return filtered_annotations, labels_map


def get_class_names_from_annotations(annotations):
    """Extraer nombres de clases desde las anotaciones"""
    # En UCF101, los nombres de clases están en frame_dir
    class_names = set()
    for ann in annotations:
        frame_dir = ann['frame_dir']
        # Formato típico: "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01"
        parts = frame_dir.split('/')
        if len(parts) > 0:
            class_name = parts[0]
            class_names.add(class_name)
    
    return sorted(list(class_names))


def prepare_data():
    """Preparar y dividir datos en train/val/test"""
    # Cargar datos
    if not os.path.exists(config.SKELETON_FILE):
        print(f"Error: No se encontró el archivo {config.SKELETON_FILE}")
        print("Por favor descarga ucf101_2d.pkl desde:")
        print("https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl")
        return None
    
    splits, annotations = load_skeleton_data(config.SKELETON_FILE)
    
    # Obtener nombres de clases
    all_class_names = get_class_names_from_annotations(annotations)
    print(f"Total de clases encontradas: {len(all_class_names)}")
    
    # Si las clases seleccionadas no están en el dataset, usar las primeras N clases
    available_classes = [c for c in config.SELECTED_CLASSES if c in all_class_names]
    if len(available_classes) < config.NUM_CLASSES:
        print(f"Advertencia: Algunas clases seleccionadas no están disponibles.")
        print(f"Usando las primeras {config.NUM_CLASSES} clases disponibles.")
        available_classes = all_class_names[:config.NUM_CLASSES]
    else:
        available_classes = available_classes[:config.NUM_CLASSES]
    
    print(f"Clases seleccionadas: {available_classes}")
    
    # Filtrar anotaciones
    filtered_annotations, labels_map = filter_classes(annotations, available_classes, all_class_names)
    
    # Dividir datos
    # Primero separar train del resto
    train_annotations, temp_annotations = train_test_split(
        filtered_annotations,
        test_size=(1 - config.TRAIN_CONFIG['train_split']),
        random_state=42,
        stratify=[ann['label'] for ann in filtered_annotations]
    )
    
    # Luego separar val y test
    val_size = config.TRAIN_CONFIG['val_split'] / (config.TRAIN_CONFIG['val_split'] + config.TRAIN_CONFIG['test_split'])
    val_annotations, test_annotations = train_test_split(
        temp_annotations,
        test_size=(1 - val_size),
        random_state=42,
        stratify=[ann['label'] for ann in temp_annotations]
    )
    
    print(f"Train: {len(train_annotations)}, Val: {len(val_annotations)}, Test: {len(test_annotations)}")
    
    # Guardar datos procesados
    processed_data = {
        'train': train_annotations,
        'val': val_annotations,
        'test': test_annotations,
        'labels_map': labels_map,
        'class_names': available_classes
    }
    
    processed_file = os.path.join(config.PROCESSED_DATA_DIR, 'processed_data.pkl')
    with open(processed_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Datos procesados guardados en {processed_file}")
    return processed_data


def load_processed_data():
    """Cargar datos procesados"""
    processed_file = os.path.join(config.PROCESSED_DATA_DIR, 'processed_data.pkl')
    if not os.path.exists(processed_file):
        print("Datos procesados no encontrados. Ejecutando preparación...")
        return prepare_data()
    
    with open(processed_file, 'rb') as f:
        return pickle.load(f)


def get_data_loaders(augment_train=True):
    """Obtener DataLoaders para train, val y test"""
    processed_data = load_processed_data()
    if processed_data is None:
        return None, None, None
    
    # Crear datasets
    train_dataset = SkeletonDataset(
        processed_data['train'],
        processed_data['labels_map'],
        augment=augment_train
    )
    val_dataset = SkeletonDataset(
        processed_data['val'],
        processed_data['labels_map'],
        augment=False
    )
    test_dataset = SkeletonDataset(
        processed_data['test'],
        processed_data['labels_map'],
        augment=False
    )
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=config.TRAIN_CONFIG['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config.TRAIN_CONFIG['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config.TRAIN_CONFIG['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, processed_data['class_names']


if __name__ == '__main__':
    # Preparar datos
    prepare_data()

