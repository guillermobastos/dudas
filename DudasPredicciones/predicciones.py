import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_excel("Tabla_Sin_Vacios.xlsx")
df = df.dropna()

# Instanciar el transformador
scaler = StandardScaler()
df['UI_normalized'] = scaler.fit_transform(df[['UI']])
X = df[
        [
            # "Dia",
            # "Mes",
            # "Ano",
            # "U",
            # "V",
            "UI_normalized",
            "Fosfato",
            "Nitrato",
            "Nitrito",
            "Silicato",
            "Temp_1",
            "Temp_2",
            "Temp_3",
            "Salin_1",
            "Salin_2",
            "Salin_3",
            # "GYMNCATE"
        ]
    ]  

# ----------------------------------------------------------------
def xgboost():
    y = df["PSEUSPP"]
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir el modelo XGBoost
    model = xgb.XGBRegressor()

    # Entrenar el modelo
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"Coeficiente de determinación (R^2): {r2:.4f}")

    # Hacer predicciones en el conjunto de prueba
    predictions = model.predict(X_test)
    print(predictions)

    # Calcular otras métricas de evaluación
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"Error Absoluto Medio (MAE): {mae:.4f}")
    

# ----------------------------------------------------------------    
def random_forest():
    y = df["PSEUSPP"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir el modelo Random Forest Classifier
    model = RandomForestRegressor(random_state=42)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    predictions = model.predict(X_test)

    # Calcular el coeficiente de determinación (R^2) en el conjunto de prueba
    r2 = r2_score(y_test, predictions)
    print("Coeficiente de determinación (R^2):", r2)

# ----------------------------------------------------------------    
def adaboost():
    y = df["PSEUSPP"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3333, random_state=42)

    # Definir el modelo AdaBoost Regressor
    model = AdaBoostRegressor(random_state=42)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Calcular el coeficiente de determinación (R^2) en el conjunto de prueba
    r2 = model.score(X_test, y_test)
    print("Coeficiente de determinación (R^2):", r2)

    # Hacer predicciones en el conjunto de prueba
    predictions = model.predict(X_test)
    print(predictions.shape)


# ----------------------------------------------------------------    
def main():
    xgboost()
    # random_forest()
    # adaboost()
    plt.show()

if __name__ == "__main__":
    main()