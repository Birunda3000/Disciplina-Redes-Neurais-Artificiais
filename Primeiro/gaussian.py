#fix the error
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from IPython.display import display
from tqdm import tqdm
import utils

def gaussian(x, mu, sigma):
    # broadcast sigma to match the shape of x and mu
    sigma = np.tile(sigma, (x.shape[0], 1))
    
    # compute the exponent
    exponent = -((x[:, None] - mu)**2) / (2 * sigma**2)
    
    # compute the Gaussian values
    gaussians = np.exp(exponent)
    
    return gaussians

# Load df
path = os.path.join(os.path.dirname(__file__), "..", "_DATA_", "Iris.csv")
df = pd.read_csv(path)
df = ut.fn_cat_onehot(df)
# add bias column
display(df.head())

# select the feature columns
features = df[
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
]
features_array = np.array(features)

# select the label columns
labels = df[
    ["Species_Iris-setosa", "Species_Iris-versicolor", "Species_Iris-virginica"]
]
labels_array = np.array(labels)

# mu and sigma
mu = ut.get_kmeans_centers_for_rbf(df=features, n_clusters=5)#5x4





print(gaussian(x=features.values, mu=mu, sigma=sigma))