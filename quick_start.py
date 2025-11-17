"""
Script de inicio rápido para probar el pipeline completo
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from utils.data_loader import prepare_data, load_processed_data


def quick_start():
    """Ejecutar pipeline completo de forma rápida"""
    print("="*60)
    print("QUICK START - Pipeline de Reconocimiento de Acciones")
    print("="*60)
    
    # 1. Verificar datos
    print("\n1. Verificando datos...")
    if not os.path.exists(config.SKELETON_FILE):
        print(f"[ERROR] No se encontro {config.SKELETON_FILE}")
        print("\nPor favor descarga el archivo desde:")
        print("https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl")
        print(f"Y colocalo en: {config.SKELETON_FILE}")
        return
    else:
        print(f"[OK] Archivo encontrado: {config.SKELETON_FILE}")
    
    # 2. Preparar datos
    print("\n2. Preparando datos...")
    processed_file = os.path.join(config.PROCESSED_DATA_DIR, 'processed_data.pkl')
    if os.path.exists(processed_file):
        print(f"✓ Datos ya procesados: {processed_file}")
        processed_data = load_processed_data()
    else:
        print("Procesando datos por primera vez...")
        processed_data = prepare_data()
        if processed_data is None:
            return
    
    if processed_data:
        print(f"[OK] Clases: {processed_data['class_names']}")
        print(f"[OK] Train: {len(processed_data['train'])} videos")
        print(f"[OK] Val: {len(processed_data['val'])} videos")
        print(f"[OK] Test: {len(processed_data['test'])} videos")
    
    # 3. Información sobre entrenamiento
    print("\n3. Información de entrenamiento:")
    print(f"   Device: {config.TRAIN_CONFIG['device']}")
    print(f"   Batch size: {config.TRAIN_CONFIG['batch_size']}")
    print(f"   Épocas: {config.TRAIN_CONFIG['num_epochs']}")
    print(f"   Learning rate: {config.TRAIN_CONFIG['learning_rate']}")
    
    # 4. Comandos sugeridos
    print("\n4. Próximos pasos:")
    print("\n   Para entrenar el modelo baseline:")
    print("   python train.py --model baseline --epochs 30")
    print("\n   Para entrenar el modelo mejorado:")
    print("   python train.py --model improved --epochs 30")
    print("\n   Para evaluar un modelo:")
    print("   python evaluate.py --model baseline")
    print("\n   Para hacer una predicción:")
    print("   python predict.py --model improved")
    print("\n   Para comparar modelos:")
    print("   python compare_models.py")
    
    print("\n" + "="*60)
    print("[OK] Quick start completado!")
    print("="*60)


if __name__ == '__main__':
    quick_start()

