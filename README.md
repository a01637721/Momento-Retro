# Detección de Acciones Humanas en UCF101

Proyecto de deep learning para la clasificación de acciones humanas en videos del dataset UCF101 utilizando representaciones de esqueletos 2D.

## Descripción

Este proyecto implementa modelos de deep learning para reconocimiento de acciones humanas usando anotaciones de esqueletos 2D del dataset UCF101. Se incluyen:

- **Modelo Baseline**: LSTM simple para secuencias de esqueletos
- **Modelo Mejorado**: LSTM bidireccional con capas de atención y dropout para regularización

## Estructura del Proyecto

```
momento-retro/
├── data/
│   ├── ucf101_2d.pkl          # Anotaciones de esqueletos (descargar)
│   └── processed/              # Datos procesados
├── models/
│   ├── baseline_lstm.py     # Modelo baseline
│   └── improved_lstm.py       # Modelo mejorado
├── utils/
│   ├── data_loader.py          # Carga y preprocesamiento de datos
│   └── visualization.py        # Visualización de resultados
├── config.py                   # Configuración del proyecto
├── train.py                    # Script de entrenamiento
├── evaluate.py                 # Script de evaluación
├── predict.py                  # Script de predicción
├── requirements.txt            # Dependencias
└── README.md                   # Este archivo
```

## Instalación

Instalar dependencias:
```bash
pip install -r requirements.txt
```

Descargar los datos:
```bash
# Descargar anotaciones de esqueletos UCF101 2D desde:
# https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl
# Colocar el archivo en data/ucf101_2d.pkl
```

## Uso

### 1. Preprocesamiento de datos

Primero, descarga el archivo de anotaciones de esqueletos:
- URL: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl
- Coloca el archivo en `data/ucf101_2d.pkl`

Luego, prepara los datos:
```bash
python -m utils.data_loader
```

Script:
- Carga las anotaciones de esqueletos
- Filtra las clases seleccionadas (10 clases por defecto)
- Divide los datos en train/val/test (70/15/15)
- Guarda los datos procesados en `data/processed/processed_data.pkl`

### 2. Entrenamiento

```bash
# Entrenar modelo baseline
python train.py --model baseline --epochs 50

# Entrenar modelo mejorado
python train.py --model improved --epochs 50
```

El entrenamiento:
- Guarda el mejor modelo basado en accuracy de validación
- Implementa early stopping para evitar overfitting

### 3. Evaluación

```bash
# Evaluar modelo baseline
python evaluate.py --model baseline

# Evaluar modelo mejorado
python evaluate.py --model improved
```

La evaluación genera:
- Métricas de clasificación (accuracy, precision, recall, F1)
- Matriz de confusión
- Reporte de clasificación por clase

### 4. Comparación de Modelos

```bash
python compare_models.py
```

Compara ambos modelos y genera visualizaciones comparativas.

### 5. Predicción

```bash
# Predicción con video aleatorio del test set
python predict.py --model improved

# Predicción con video específico
python predict.py --model improved --video_id "Basketball/v_Basketball_g01_c01"
```

## Resultados

Los resultados de entrenamiento y evaluación se guardan en:
- `logs/`: Logs de entrenamiento
- `checkpoints/`: Modelos guardados
- `results/`: Métricas y visualizaciones

## Modelos Implementados

### Baseline Model
- **Arquitectura**: LSTM simple (1 capa, 128 unidades ocultas)
- **Regularización**: Ninguna
- **Optimizador**: Adam (lr=0.001)
- **Características**: Modelo simple para establecer línea base

### Improved Model
- **Arquitectura**: LSTM bidireccional (2 capas, 256 unidades ocultas)
- **Regularización**: 
  - Dropout (0.3) en capas fully connected
  - Batch Normalization en la entrada
- **Atención**: Módulo de atención para enfocarse en frames importantes
- **Optimizador**: Adam (lr=0.001) con ReduceLROnPlateau scheduler
- **Técnicas adicionales**:
  - Early stopping (patience=10)
  - Data augmentation (ruido gaussiano, escalado, rotación)
  - Normalización de keypoints

### Mejoras Implementadas

1. **LSTM Bidireccional**: Captura información tanto del pasado como del futuro en la secuencia
2. **Múltiples Capas**: Mayor capacidad de modelado
3. **Atención**: Permite al modelo enfocarse en frames más relevantes
4. **Regularización**: Dropout y BatchNorm previenen overfitting
5. **Data Augmentation**: Aumenta la robustez del modelo
6. **Learning Rate Scheduling**: Ajusta dinámicamente el learning rate

## Dataset

- **Dataset**: UCF101
- **Formato**: Esqueletos 2D (17 keypoints por persona, formato COCO)
- **Clases**: Subset de 10 clases seleccionadas para entrenamiento rápido
- **Split**: Train/Val/Test (70/15/15)



