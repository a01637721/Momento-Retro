"""
Configuración del proyecto para reconocimiento de acciones en UCF101
"""
import os

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Crear directorios si no existen
for dir_path in [DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset
SKELETON_FILE = os.path.join(DATA_DIR, 'ucf101_2d.pkl')
NUM_CLASSES = 10  # Subset de clases para entrenamiento rápido
SELECTED_CLASSES = [
    'Basketball', 'Biking', 'Diving', 'GolfSwing', 'JumpRope',
    'PushUps', 'Skiing', 'SoccerJuggling', 'TennisSwing', 'VolleyballSpiking'
]

# Preprocesamiento
MAX_PERSONS = 2  # Máximo número de personas por frame
MAX_FRAMES = 300  # Máximo número de frames por video (padding/truncate)
NUM_KEYPOINTS = 17  # COCO format
KEYPOINT_DIM = 2  # 2D keypoints

# Modelo Baseline
BASELINE_CONFIG = {
    'hidden_size': 128,
    'num_layers': 1,
    'dropout': 0.0,
    'bidirectional': False,
}

# Modelo Mejorado
IMPROVED_CONFIG = {
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': True,
    'use_batch_norm': True,
    'use_attention': True,
}

# Entrenamiento
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'num_workers': 4,
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
}

# Data Augmentation
AUGMENT_CONFIG = {
    'noise_std': 0.02,  # Desviación estándar del ruido gaussiano
    'scale_range': (0.9, 1.1),  # Rango de escalado
    'rotation_range': (-5, 5),  # Rango de rotación en grados
    'enable': True,  # Habilitar/deshabilitar augmentación
}

