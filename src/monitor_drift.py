import pandas as pd
from evidently.presets import DataDriftPreset
from evidently import Report
import os
import warnings
import numpy as np

# Configurar warnings al inicio del script
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def clean_data_for_drift(df):
    """Limpia los datos para evitar warnings en el análisis de drift"""
    df_clean = df.copy()
    
    # Reemplazar infinitos con NaN y luego rellenar con 0
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    
    # Identificar columnas con varianza cero (constantes)
    constant_columns = []
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            if df_clean[col].std() == 0:
                constant_columns.append(col)
    
    # Agregar pequeño ruido aleatorio a columnas constantes
    rng = np.random.default_rng(42)
    for col in constant_columns:
        if col != 'target':  # No modificar la variable objetivo
            df_clean[col] = df_clean[col] + rng.normal(0, 1e-8, len(df_clean))
    
    return df_clean

def monitor_drift():
    # Suprimir todos los warnings durante la ejecución
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Cargar datasets
        print("Loading datasets...")
        train_data = pd.read_csv('data/processed/train.csv')
        val_data = pd.read_csv('data/processed/val.csv')
        test_data = pd.read_csv('data/processed/test.csv')
        
        # Limpiar datos para evitar problemas numéricos
        print("Cleaning data for drift analysis...")
        train_clean = clean_data_for_drift(train_data)
        val_clean = clean_data_for_drift(val_data)
        test_clean = clean_data_for_drift(test_data)
        
        # Crear directorio de salida si no existe
        os.makedirs('output/data_drift', exist_ok=True)
        
        # Report 1: Train vs Validation
        print("Generating drift report: Train vs Validation...")
        report_train_val = Report(metrics=[DataDriftPreset()])
        report_tval = report_train_val.run(reference_data=train_clean, current_data=val_clean)
        report_tval.save_html('output/data_drift/drift_train_vs_val.html')
        
        # Report 2: Train vs Test
        print("Generating drift report: Train vs Test...")
        report_train_test = Report(metrics=[DataDriftPreset()])
        report_ttest = report_train_test.run(reference_data=train_clean, current_data=test_clean)
        report_ttest.save_html('output/data_drift/drift_train_vs_test.html')
        
        # Report 3: Validation vs Test
        print("Generating drift report: Validation vs Test...")
        report_val_test = Report(metrics=[DataDriftPreset()])
        report_vtest = report_val_test.run(reference_data=val_clean, current_data=test_clean)
        report_vtest.save_html('output/data_drift/drift_val_vs_test.html')

    print("All data drift reports saved as HTML files:")
    print("- output/data_drift/drift_train_vs_val.html")
    print("- output/data_drift/drift_train_vs_test.html") 
    print("- output/data_drift/drift_val_vs_test.html")

if __name__ == "__main__":
    monitor_drift()
