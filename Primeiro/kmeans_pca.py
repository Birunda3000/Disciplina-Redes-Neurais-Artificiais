import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from IPython.display import display
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_kmeans_clusters(df, n_clusters):
    # Apply PCA to reduce the dimensionality of the data to 2 dimensions
    pca = PCA(n_components=2)
    pca_df = pca.fit_transform(df)
    
    # Run KMeans clustering on the reduced data
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pca_df)
    
    # Get the average distance of each cluster from its centroid
    cluster_distances = []
    for i in range(n_clusters):
        distances = np.linalg.norm(pca_df[kmeans.labels_ == i] - kmeans.cluster_centers_[i], axis=1)
        cluster_distances.append(np.mean(distances))
    
    # Plot the resulting clusters with different colors and circles with radii equal to the average distances
    plt.figure(figsize=(8, 8))
    for i in range(n_clusters):
        plt.scatter(pca_df[kmeans.labels_ == i, 0], pca_df[kmeans.labels_ == i, 1], s=50, alpha=0.8, label=f'Cluster {i+1}')
        circle = plt.Circle((kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1]), radius=cluster_distances[i], edgecolor='black', facecolor='none', linewidth=2)
        plt.gca().add_patch(circle)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r', label='Centroids')
    plt.legend()
    plt.show()



path = os.path.join(os.path.dirname(__file__), "..", "_DATA_", "Iris.csv")
df = pd.read_csv(path)
df = ut.fn_cat_onehot(df)

display(df.head())

# selecione as colunas de recursos
features = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Call the function to plot the clusters with 4 different colors
plot_kmeans_clusters(features, n_clusters=5)
