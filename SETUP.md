# Guía de Configuración del Proyecto

## Pasos Iniciales

### 1. Descargar el Dataset

Descarga el archivo de anotaciones de esqueletos UCF101 2D:

```bash
# Opción 1: Usando wget (Linux/Mac)
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl -O data/ucf101_2d.pkl

# Opción 2: Descarga manual
# Visita: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl
# Guarda el archivo en: data/ucf101_2d.pkl
```

### 2. Instalar Dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Verificar Configuración

Ejecuta el script de inicio rápido para verificar que todo esté configurado:

```bash
python quick_start.py
```

### 4. Preparar Datos

```bash
python -m utils.data_loader
```

Este comando procesará los datos y los guardará en `data/processed/`.

## Flujo de Trabajo Recomendado

### Paso 1: Entrenar Modelo Baseline

```bash
python train.py --model baseline --epochs 30
```

Esto entrenará el modelo baseline y guardará:
- Mejor modelo: `checkpoints/baseline/baseline_best.pth`
- Historial: `logs/baseline_history.json`

### Paso 2: Entrenar Modelo Mejorado

```bash
python train.py --model improved --epochs 30
```

Esto entrenará el modelo mejorado con todas las mejoras implementadas.

### Paso 3: Evaluar Modelos

```bash
# Evaluar baseline
python evaluate.py --model baseline

# Evaluar improved
python evaluate.py --model improved
```

### Paso 4: Comparar Resultados

```bash
python compare_models.py
```

Esto generará visualizaciones comparativas en `results/`.

### Paso 5: Hacer Predicciones

```bash
# Predicción aleatoria
python predict.py --model improved

# Predicción específica
python predict.py --model improved --video_id "Basketball/v_Basketball_g01_c01"
```

## Configuración Avanzada

Puedes modificar los parámetros en `config.py`:

- **Número de clases**: Cambia `NUM_CLASSES` y `SELECTED_CLASSES`
- **Tamaño de batch**: Modifica `TRAIN_CONFIG['batch_size']`
- **Hiperparámetros del modelo**: Ajusta `BASELINE_CONFIG` o `IMPROVED_CONFIG`
- **Data augmentation**: Configura `AUGMENT_CONFIG`

## Solución de Problemas

### Error de Memoria (CUDA OOM)

Si tienes problemas de memoria:

1. Reduce el batch size en `config.py`:
   ```python
   TRAIN_CONFIG['batch_size'] = 16  # o menor
   ```

2. Reduce el número máximo de frames:
   ```python
   MAX_FRAMES = 150  # en lugar de 300
   ```

3. Usa CPU en lugar de GPU:
   ```python
   TRAIN_CONFIG['device'] = 'cpu'
   ```

### Datos No Encontrados

Si obtienes errores sobre datos no encontrados:

1. Verifica que `data/ucf101_2d.pkl` existe
2. Ejecuta `python -m utils.data_loader` para procesar los datos
3. Verifica que `data/processed/processed_data.pkl` se creó correctamente

### Bajo Rendimiento

Si el modelo tiene bajo accuracy:

1. Aumenta el número de épocas
2. Verifica que los datos se están cargando correctamente
3. Ajusta los hiperparámetros en `config.py`
4. Considera usar más clases o más datos

## Estructura de Archivos Generados

Después de ejecutar el pipeline, tendrás:

```
momento-retro/
├── data/
│   ├── ucf101_2d.pkl              # Datos originales (descargar)
│   └── processed/
│       └── processed_data.pkl     # Datos procesados
├── checkpoints/
│   ├── baseline/
│   │   └── baseline_best.pth     # Mejor modelo baseline
│   └── improved/
│       └── improved_best.pth     # Mejor modelo mejorado
├── logs/
│   ├── baseline_history.json     # Historial baseline
│   └── improved_history.json     # Historial improved
└── results/
    ├── baseline_results.json      # Resultados baseline
    ├── improved_results.json      # Resultados improved
    ├── baseline_confusion_matrix.png
    ├── improved_confusion_matrix.png
    └── model_comparison.png       # Comparación visual
```

## Próximos Pasos

1. Experimenta con diferentes hiperparámetros
2. Prueba con más clases del dataset
3. Implementa técnicas adicionales de regularización
4. Experimenta con diferentes arquitecturas

