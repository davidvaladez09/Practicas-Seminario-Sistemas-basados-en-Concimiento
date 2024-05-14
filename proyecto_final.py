import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.widgets import Button
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

# Leer el archivo CSV con pandas (Este código no se ha modificado)
df = pd.read_csv('D:\\Documentos\\8vo\\Clasificacion Inteligente de Datos\\Dataset\\rotten_tomatoes_movies(1).csv')

# Mostrar los datos
print(df.head())

# Limpieza de datos
df = df.drop(columns=['movie_title', 'movie_info', 'critics_consensus', 'production_company', 'PG', 'R', 'NR', 'G', 'PG-13', 'NC17', 'Certified_Fresh', 'Rotten', 'Fresh', 'Comedy', 'Drama', 'Action_&_Adventure', 'Science_Fiction_&_Fantasy', 'Romance', 'Classics', 'Kids_&_Family', 'Mystery_&_Suspense', 'Western', 'Art_House_&_International', 'Faith_&_Spirituality', 'Documentary', 'Special_Interest']) # Eliminar columnas innecesarias

df = df.drop_duplicates() # Eliminar filas duplicadas

# Eliminar espacios en blanco en una columna de texto
df['movie_title', 'movie_info', 'critics_consensus', 'production_company', 'PG', 'R', 'NR', 'G', 'PG-13', 'NC17', 'Certified_Fresh', 'Rotten', 'Fresh', 'Comedy', 'Drama', 'Action_&_Adventure', 'Science_Fiction_&_Fantasy', 'Romance', 'Classics', 'Kids_&_Family', 'Mystery_&_Suspense', 'Western', 'Art_House_&_International', 'Faith_&_Spirituality', 'Documentary', 'Special_Interest'] = df['columna_texto'].str.strip()

# Mostrar los datos despues de la limpieza
print(df.head())


# Box plot
for column in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot de la columna {column}')
    plt.show()

# Seleccionar características relevantes para el clustering
X = df[['tomatometer_rating', 'audience_rating']]

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcular la inercia para diferentes valores de K
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Generar datos de ejemplo para las otras gráficas (Este código no se ha modificado)
x = np.linspace(0, 10, 100)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x)
y5 = np.log(x + 1)
y6 = np.sqrt(x)
y7 = x ** 2
y8 = np.random.normal(size=100)
y9 = np.random.uniform(size=100)

gs = gridspec.GridSpec(3, 3)  # Definir la cuadrícula

# Función para mostrar la gráfica de Voronoi
def mostrar_voronoi(event):
    # resultado del clustering utilizando K-Means en las calificaciones de la audiencia y de los críticos.

    # Codificar la columna 'genres' en valores numéricos
    label_encoder = LabelEncoder()
    df['genres_encoded'] = label_encoder.fit_transform(df['genres'])

    # Seleccionar características relevantes para el clustering
    X = df[['runtime', 'tomatometer_rating']]

    # Normalizar los datos
    X_normalized = (X - X.mean()) / X.std()

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_normalized)

    # Calcular la silueta
    silhouette_avg = silhouette_score(X_normalized, clusters)
    print("Silhouette Score:", silhouette_avg)

    # Calcular el valor de la silueta para cada muestra
    sample_silhouette_values = silhouette_samples(X_normalized, clusters)

    # Obtener los centroides de los clusters
    centroids = kmeans.cluster_centers_

    # Crear una malla para la gráfica de Voronoi
    x_min, x_max = X_normalized.iloc[:, 0].min() - 1, X_normalized.iloc[:, 0].max() + 1
    y_min, y_max = X_normalized.iloc[:, 1].min() - 1, X_normalized.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Calcular las asignaciones de los puntos de la malla a los clusters
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Visualizar los clusters y los datos
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_normalized.iloc[:, 0], X_normalized.iloc[:, 1], c=clusters, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', c='red', s=200, edgecolor='k')
    plt.xlabel('Tiempo en minutos')
    plt.ylabel('Tomatometer Audience Rating')
    plt.title('K-means Clustering con Diagrama de Voronoi')
    plt.colorbar()
    plt.grid(True)
    plt.show()


# Crear el dashboard con las gráficas (Este código se ha modificado)
fig = plt.figure(figsize=(15, 15))
plt.suptitle('Características de visualizaciones en Amazon Prime Video', fontsize=16)  # Título del dashboard

# Gráfica 1 a 7 (Este código no se ha modificado)
for i in range(0, 9):
    ax = fig.add_subplot(gs[i // 3, i % 3])  # Acceder a cada subplot
    if i == 0:
        #  muestra la suma de la audiencia para cada clasificación de contenido.
        df.groupby('content_rating')['audience_count'].sum().plot(kind='bar', ax=ax)
        ax.set_title('Suma de Audiencia por Clasificación de Contenido')
        ax.set_xlabel('Clasificación de Contenido')
        ax.set_ylabel('Suma de Audiencia')
    elif i == 1:
        # porcentaje de películas para cada estado de Tomatometer.
        genre_counts = df['tomatometer_status'].value_counts()
        ax.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
        ax.set_title('Porcentaje de Audiencia por Tomatometer Status')
    elif i == 2:
        # porcentaje de películas para cada clasificación de contenido.
        genre_counts = df['content_rating'].value_counts()
        ax.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
        ax.set_title('Porcentaje de Audiencia por Clasificacion')
    elif i == 3:
        # relación entre la Tiempo de la película (en minutos) y la calificación de la audiencia.
        ax.scatter(df['runtime'], df['audience_rating'])
        ax.set_title('Dispersión de Audiencia vs Tiempo en Minutos')
        ax.set_xlabel('Tiempo en Minutos')
        ax.set_ylabel('Audiencia')
    elif i == 4:
        # resultado del clustering utilizando K-Means en las calificaciones de la audiencia y de los críticos.

        # Seleccionar características relevantes para el clustering
        # X = df[['tomatometer_rating', 'audience_rating']]
        X = df[['runtime', 'audience_rating']]
        
        # Normalizar los datos
        # X_normalized = (X - X.mean()) / X.std()

        # Aplicar K-Means
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X)

        # Calcular la silueta
        silhouette_avg = silhouette_score(X, clusters)
        print("Silhouette Score:", silhouette_avg)

        # Calcular el valor de la silueta para cada muestra
        sample_silhouette_values = silhouette_samples(X, clusters)

        # Crear una gráfica de silueta
        # ax.scatter(X['tomatometer_rating'], X['audience_rating'], c=clusters, cmap='viridis', alpha=0.7)
        ax.scatter(X['runtime'], X['audience_rating'], c=clusters, cmap='viridis', alpha=0.7)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', s=100)
        ax.set_title("Clustering de Audiencia y Calificación de Críticos")
        # ax.set_xlabel("Tomatometer Rating")
        ax.set_xlabel("Tiempo en minutos")
        ax.set_ylabel("Audiencia Rating")
        
    elif i == 5:
        # Graficar el gráfico de codo
        #  muestra la inercia (suma de distancias al cuadrado de las muestras al centro del clúster más cercano) en función del número de clústeres.
        ax.plot(range(1, 11), inertia, marker='o', linestyle='-')
        ax.set_xlabel('Número de clústeres (K)')
        ax.set_ylabel('Inercia')
        ax.set_title('Número óptimo de clústeres')
        ax.grid(True)
    elif i == 6:
        # Histograma
        # muestra la distribución de la audiencia.
        sns.histplot(data=df, x='audience_rating', bins=20, kde=True, ax=ax)
        ax.set_title('Histograma de Audiencia')
        ax.set_xlabel('Audiencia')
        ax.set_ylabel('Frecuencia')
    elif i == 7:
        # Grafica se silueta

        # Seleccionar características relevantes para el clustering
        X = df[['runtime', 'audience_rating']]

        # Normalizar los datos
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # Aplicar K-Means
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_normalized)

        # Calcular la silueta
        silhouette_avg = silhouette_score(X_normalized, clusters)
        print("Silhouette Score:", silhouette_avg)

        # Calcular el valor de la silueta para cada muestra
        sample_silhouette_values = silhouette_samples(X_normalized, clusters)

        # Obtener los centroides de los clusters
        centroids = kmeans.cluster_centers_

        # Crear una malla para la gráfica de Voronoi
        x_min, x_max = X_normalized[:, 0].min() - 1, X_normalized[:, 0].max() + 1
        y_min, y_max = X_normalized[:, 1].min() - 1, X_normalized[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # Calcular las asignaciones de los puntos de la malla a los clusters
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)


        # Gráfica de Silueta
        ax = fig.add_subplot(gs[2, 1])  # Acceder al subplot 8
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Línea de referencia para la silueta media
        y_lower = 10
        for i in range(4):  # Iterar sobre cada cluster
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = plt.cm.viridis(float(i) / 4)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax.set_title('Gráfica de Silueta')
        ax.set_xlabel('Coeficiente de Silueta')
        ax.set_ylabel('Cluster')


# Crear el botón y colocarlo en la posición deseada
# muestra la partición del espacio en regiones asociadas a cada centroide de clúster.
ax_button = plt.subplot(gs[2, 2])  # Acceder al subplot 9 (posición plt.subplot(3, 3, 9))
button = Button(ax_button, 'Mostrar Grafica Voronoi')
button.on_clicked(mostrar_voronoi)

plt.tight_layout()
plt.show()
