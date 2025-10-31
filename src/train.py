import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

def train_model():
    mlflow.set_experiment("mlops_demo")
    with mlflow.start_run():
        data = pd.read_csv('data/processed/train.csv')
        X = data.drop('target', axis=1)
        y = data['target']
        model = LogisticRegression()
        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model")
        print("Model trained and logged to MLflow.")

if __name__ == "__main__":
    train_model()
