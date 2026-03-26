import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
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

        self.model = None

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def train_random_forest(self, n_estimators=100, random_state=42):
        print("\n\tEntrenando modelo...")
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)

    def save_model(self, path:str):
        joblib.dump(self.model, path)

    def save_limits_PSO(self, path:str):
        limits = [(self.X[:,i].min(), self.X[:,i].max()) for i in range(self.X.shape[1])]
        joblib.dump(limits, path)

    def evaluate_model(self):
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        print("\nR2 entrenamiento:", r2_score(self.y_train, y_pred_train))
        print("R2 prueba:", r2_score(self.y_test, y_pred_test))

        rmse = mean_squared_error(self.y_test, y_pred_test) ** 0.5
        print("RMSE prueba:", rmse)
        