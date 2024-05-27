import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_excel("Tabla_Sin_Vacios.xlsx")
df = df.dropna()

    # Instanciar el transformador
scaler = StandardScaler()
df['UI'] = scaler.fit_transform(df[['UI']])

# Definir características (features) y objetivo (target)
features_to_scale = [
    "UI",
    "Fosfato",
    "Nitrato",
    "Nitrito",
    "Silicato",
    "Temp_1",
    "Temp_2",
    "Temp_3",
    "Salin_1",
    "Salin_2",
    "Salin_3"
]
features = df[features_to_scale]
    
def predicciones():
    # Establecemos las columnas target
    target = df[["PSEUSPP", "GYMNCATE"]]

    # Dividir los datos en conjuntos de entrenamiento (2/3) y prueba (1/3)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3333, random_state=0)

    # Entrenar el modelo de Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    # Predecir valores para los datos de prueba
    y_pred = model.predict(X_test)
    # print("Shape: ",y_pred.shape) 
    # print("Todo: ",y_pred)

    # Accuracy RandomForest Model
    accuracy_pseuspp_rf = accuracy_score(y_test["PSEUSPP"], y_pred[:, 0])
    accuracy_gymncate_rf = accuracy_score(y_test["GYMNCATE"], y_pred[:, 1])
    print(f"Precisión del modelo Random Forest para PSEUSPP: {accuracy_pseuspp_rf * 100:.2f}%")
    print(f"Precisión del modelo Random Forest para GYMNCATE: {accuracy_gymncate_rf * 100:.2f}%")

    # Entrenar el modelo de KMeans con los datos normalizados de prueba
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_test)

    # Asignar los nuevos puntos de datos a un clúster
    new_clusters = kmeans.predict(X_test)

    # Contar las repeticiones de cada cluster
    cluster_counts = pd.Series(new_clusters).value_counts().sort_index()

    # Crear la gráfica de barras
    plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title('Cantidad de Predicciones de PSEUSPP por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Cantidad de Predicciones')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()

    # Añadir las predicciones y clústeres al DataFrame original
    df['Predicciones_PSEUSPP'] = np.concatenate((y_pred[:, 0], np.full(len(df) - len(y_pred), np.nan)))
    df['Predicciones_GYMNCATE'] = np.concatenate((y_pred[:, 1], np.full(len(df) - len(y_pred), np.nan)))
    df['Cluster'] = np.concatenate((new_clusters, np.full(len(df) - len(new_clusters), np.nan)))
    df.dropna(subset=['Predicciones_GYMNCATE', 'Predicciones_PSEUSPP'], inplace=True)
    df.to_csv('Tablas/resultados_predicciones2.csv', index=False)


    indice_PSEUSPP = df.columns.get_loc("Predicciones_PSEUSPP")
    indice_GYMNCATE = df.columns.get_loc("Predicciones_GYMNCATE")
    conteos_cluster_PSEUSPP = [0] * 3
    conteos_cluster_GYMNCATE = [0] * 3
    porcentajes = []
    total_PSEUSPP = 0
    total_GYMNCATE = 0
    
    # Iterar sobre cada cluster
    for i in range(3):
        # Obtener los índices de las filas en X correspondientes al cluster i
        indices_cluster_i = np.where(new_clusters == i)[0]
        valores_cluster_PSEUSPP = df.iloc[indices_cluster_i, indice_PSEUSPP].values
        valores_cluster_GYMNCATE = df.iloc[indices_cluster_i, indice_GYMNCATE].values

        conteos_cluster_PSEUSPP[i] = np.sum(valores_cluster_PSEUSPP > 0)
        conteos_cluster_GYMNCATE[i] = np.sum(valores_cluster_GYMNCATE > 0)   
        total_PSEUSPP += conteos_cluster_PSEUSPP[i]
        total_GYMNCATE += conteos_cluster_GYMNCATE[i]
        
    for i in range(3):
        porcentaje_PSEUSPP = (conteos_cluster_PSEUSPP[i]/total_PSEUSPP) * 100
        porcentaje_GYMNCATE = (conteos_cluster_GYMNCATE[i]/total_GYMNCATE) * 100
        porcentajes.append((porcentaje_PSEUSPP, porcentaje_GYMNCATE))

    print("Modelo PCA -> GYMNCATE",conteos_cluster_GYMNCATE)
    print("Modelo PCA -> PSEUSPP",conteos_cluster_PSEUSPP)

    num_clusters = len(porcentajes)
    x = np.arange(num_clusters)
    
    # Nombres de las etiquetas de los clusters
    labels = [f'Cluster {i+1}' for i in range(num_clusters)]
    # Ancho de las barras
    width = 0.35

    porcentaje_PSEUSPP = [p[0] for p in porcentajes]
    porcentaje_GYMNCATE = [p[1] for p in porcentajes]
    # Crear el gráfico de barras
    fig, ax = plt.subplots()
    ax.bar(x - width/3, porcentaje_PSEUSPP, width, label='PSEUSPP')
    ax.bar(x + width/3, porcentaje_GYMNCATE, width, label='GYMNCATE')
    # Etiquetas, título y leyenda
    ax.set_ylabel('Porcentaje')
    ax.set_title('Gráfica Predicciones')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ylim_min = 0
    ylim_max = 100
    yticks_interval = 20
    ax.set_yticks(np.arange(ylim_min, ylim_max + yticks_interval, yticks_interval))
    # Agregar etiquetas en los ticks del eje y con los valores exactos de cada porcentaje
    for i in range(num_clusters):
        ax.text(x[i] - width/3, porcentaje_PSEUSPP[i] + 1, f'{porcentaje_PSEUSPP[i]:.2f}%', ha='center')
        ax.text(x[i] + width/3, porcentaje_GYMNCATE[i] + 1, f'{porcentaje_GYMNCATE[i]:.2f}%', ha='center')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    

# ----------------------------------------------------------------
def main():
    predicciones()

if __name__ == "__main__":
    main()