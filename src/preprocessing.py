import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_split_data(path: str):
    df = pd.read_csv(path)
    
    # Eliminar columnas no numéricas que no son útiles para el modelo
    # Mantener solo columnas numéricas y la columna target
    columns_to_drop = ['key_value', 'p_codmes']  # Columnas de identificación
    
    # Buscar columnas categóricas/texto adicionales
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'target':
            # Si es categórica con pocos valores únicos, convertir a numérico
            if df[col].nunique() <= 10:
                df[col] = pd.Categorical(df[col]).codes
            else:
                columns_to_drop.append(col)
    
    # Eliminar columnas identificadas
    df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Manejar valores problemáticos
    # Reemplazar infinitos con NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Rellenar NaN con 0
    df_clean = df_clean.fillna(0)
    
    # Identificar y manejar columnas con varianza muy baja
    for col in df_clean.columns:
        if col != 'target' and df_clean[col].dtype in ['int64', 'float64']:
            # Si la columna tiene varianza muy baja, agregar pequeño ruido
            if df_clean[col].std() < 1e-10:
                print(f"Warning: Column '{col}' has very low variance. Adding small noise.")
                # Usar un generador de números aleatorios moderno
                rng = np.random.default_rng(42)
                df_clean[col] = df_clean[col] + rng.normal(0, 1e-8, len(df_clean))
    
    # Dividir en train, val y test
    # Primero dividir en train+val (80%) y test (20%)
    train_val, test = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['target'])
    
    # Luego dividir train+val en train (60% del total) y val (20% del total)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42, stratify=train_val['target'])
    
    # Guardar los datasets
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    
    print("Data preprocessed and saved.")
    print(f"Train shape: {train.shape} (60%)")
    print(f"Validation shape: {val.shape} (20%)")
    print(f"Test shape: {test.shape} (20%)")
    print(f"Columns kept: {list(df_clean.columns)}")
    
    # Verificar distribución de clases
    print("\nClass distribution:")
    print(f"Train - Class 0: {(train['target'] == 0).sum()}, Class 1: {(train['target'] == 1).sum()}")
    print(f"Val   - Class 0: {(val['target'] == 0).sum()}, Class 1: {(val['target'] == 1).sum()}")
    print(f"Test  - Class 0: {(test['target'] == 0).sum()}, Class 1: {(test['target'] == 1).sum()}")
    
if __name__ == "__main__":
    load_and_split_data('data/raw/Data_CU_venta.csv')