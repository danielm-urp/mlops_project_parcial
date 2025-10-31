import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import pickle

def train_model():
    # Parámetros del modelo
    n_estimators = 100
    max_depth = 10
    min_samples_split = 5
    min_samples_leaf = 2
    random_state = 42
    
    mlflow.set_experiment("Mlops_RandomForest_Classification")
    with mlflow.start_run():
        # Cargar datos
        data = pd.read_csv('data/processed/train.csv')
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Log de información del dataset
        class_distribution = y.value_counts()
        mlflow.log_param("class_0_count", int(class_distribution[0]))
        mlflow.log_param("class_1_count", int(class_distribution[1]))
        mlflow.log_param("imbalance_ratio", float(class_distribution[0] / class_distribution[1]))
        
        # Dividir en train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # Crear y configurar modelo con balanceo de clases
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced'  # Importante: balancear las clases automáticamente
        )
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calcular métricas
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        # Métricas adicionales para clases desbalanceadas
        from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
        
        # Obtener métricas balanceadas
        precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        
        # Matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Log de parámetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        
        # Log de métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision_macro", precision_macro)
        mlflow.log_metric("recall_macro", recall_macro)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("balanced_accuracy", balanced_acc)
        mlflow.log_metric("specificity", specificity)
        
        # Guardar el modelo en formato pickle
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Log del modelo en MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print("Model trained and logged to MLflow.")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Precision Macro: {precision_macro:.4f}")
        print(f"Recall Macro: {recall_macro:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print("Model saved as models/model.pkl")

if __name__ == "__main__":
    train_model()
