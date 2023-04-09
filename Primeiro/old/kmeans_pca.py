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
    
    # Plot the resulting clusters with different colors
    plt.scatter(pca_df[:, 0], pca_df[:, 1], c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
    plt.show()


path = os.path.join(os.path.dirname(__file__), "..", "_DATA_", "Iris.csv")
df = pd.read_csv(path)
df = ut.fn_cat_onehot(df)
df.insert(0, "Bias", 1)
display(df.head())

# selecione as colunas de recursos
features = df[["Bias","SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Call the function to plot the clusters with 4 different colors
plot_kmeans_clusters(features, n_clusters=3)
