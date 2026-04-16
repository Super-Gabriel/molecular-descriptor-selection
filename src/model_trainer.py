import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib


class ModelTrainer:
    def __init__(self, X:np.ndarray, y:np.ndarray):
        self.X = X
        self.y = y

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.scaler = None 
        self.model = None

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def normalize_features(self):
        """Ajusta el scaler con los datos de entrenamiento y transforma train y test."""
        self.scaler = MinMaxScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("\n\tDescriptores normalizados (MinMaxScaler) en rango [0,1]")

    def train_random_forest(self, n_estimators=100, random_state=42):
        print("\n\tEntrenando modelo...")
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)

    def save_model(self, path:str):
        joblib.dump(self.model, path)

    def save_limits_PSO(self, path:str):
        # Obtenemos los límites utilizando X_train para evitar fuga de información
        lower = np.percentile(self.X_train, 1, axis=0)
        upper = np.percentile(self.X_train, 99, axis=0)
        
        limits = [(lower[i], upper[i]) for i in range(self.X_train.shape[1])]

        joblib.dump(limits, path)

    def save_limits_PSO_normalized(self, path:str):
        # Aplicamos la misma lógica del percentil 1 y 99 pero en el espacio normalizado usando X_train
        lower = np.percentile(self.X_train, 1, axis=0)
        upper = np.percentile(self.X_train, 99, axis=0)
        
        limits = [(lower[i], upper[i]) for i in range(self.X_train.shape[1])]
        joblib.dump(limits, path)
        print(f"\n\tLímites para PSO guardados en {path}")

    def save_scaler(self, path: str):
        joblib.dump(self.scaler, path)

    def evaluate_model(self):
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        print("\nR2 entrenamiento:", r2_score(self.y_train, y_pred_train))
        print("R2 prueba:", r2_score(self.y_test, y_pred_test))

        rmse = mean_squared_error(self.y_test, y_pred_test) ** 0.5
        print("RMSE prueba:", rmse)
        