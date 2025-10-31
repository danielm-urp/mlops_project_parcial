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
    
    # Manejar valores NaN
    df_clean = df_clean.fillna(0)
    
    # Dividir en train y test
    train, test = train_test_split(df_clean, test_size=0.2, random_state=42)
    train.to_csv('data/processed/train.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    
    print("Data preprocessed and saved.")
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Columns kept: {list(df_clean.columns)}")
    
if __name__ == "__main__":
    load_and_split_data('data/raw/Data_CU_venta.csv')