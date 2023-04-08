
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
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def get_kmeans_centers_for_rbf(df, n_clusters):
    # Initialize a KMeans model with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    
    # Fit the model to the data
    kmeans.fit(df)
    
    # Return the centers of each cluster as a numpy array
    return kmeans.cluster_centers_



path = os.path.join(os.path.dirname(__file__), "..", "_DATA_", "Iris.csv")
df = pd.read_csv(path)
df = ut.fn_cat_onehot(df)
df.insert(0, "Bias", 1)
display(df.head())

# selecione as colunas de recursos
features = df[["Bias","SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Call the function to get the centers of 5 clusters
centers = get_kmeans_centers_for_rbf(df, 5)

np.set_printoptions(precision=1)

# Print the resulting centers
for i, center in enumerate(centers):
    print("Center {}: {}".format(i, center))





