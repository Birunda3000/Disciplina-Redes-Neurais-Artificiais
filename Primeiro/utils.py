from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def fn_cat_onehot(df):
    """Generate onehoteencoded features for all categorical columns in df"""
    # print(f"df shape: {df.shape}")
    # NaN handing
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"NaN = **{nan_count}** will be categorized under feature_nan columns")

    model_oh = OneHotEncoder(handle_unknown="ignore", sparse=False)
    for c in list(df.select_dtypes("category").columns) + list(
        df.select_dtypes("object").columns
    ):
        print(f"Encoding **{c}**")  # which column
        matrix = model_oh.fit_transform(
            df[[c]]
        )  # get a matrix of new features and values
        names = model_oh.get_feature_names_out()  # get names for these features
        df_oh = pd.DataFrame(
            data=matrix, columns=names, index=df.index
        )  # create df of these new features

        # display(df_oh.plot.hist())

        df = pd.concat([df, df_oh], axis=1)  # concat with existing df

        df.drop(
            c, axis=1, inplace=True
        )  # drop categorical column so that it is all numerical for modelling

    # print(f"#### New df shape: **{df.shape}**")
    return df

# Activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Derivative of the activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# MSE
def MSE(Y_target, Y_pred):
    return np.mean((Y_target - Y_pred) ** 2)

#np.where(output > 0.5, 1, 0)

def classification_error(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    incorrect = np.sum(y_true != y_pred)
    total = y_true.shape[0]
    error_rate = incorrect / total
    return error_rate

def calc_accuracy(true_labels, pred_labels):
    # transforma as saídas preditas em um array binário
    pred_labels_bin = np.zeros_like(pred_labels)
    pred_labels_bin[np.arange(len(pred_labels)), pred_labels.argmax(axis=1)] = 1
    # calcula o número de acertos e erros
    correct = (pred_labels_bin == true_labels).all(axis=1).sum()
    incorrect = len(true_labels) - correct
    return correct, incorrect

def gaussian_1(x, mu, sigma):
    output = []
    x = np.array(x)
    mu = np.array(mu)
    sigma = np.array(sigma) 
    for sample in x:
        output.append( np.exp( -( np.linalg.norm(sample - mu) )**2 / (2 * sigma**2) ) )
    return np.array(output) 

def gaussian(x, mu, sigma):
    output = []
    for neuron in range(len(mu)):
        output.append( gaussian_1(x=x, mu=mu[neuron], sigma=sigma[neuron]) )
    return np.array(output).T 


def gaussian_derivative_x(x, mu, sigma):
    return -np.exp(-(x - mu)**2 / (2 * sigma**2)) * (x - mu) / sigma**2
def gaussian_derivative_mu(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) * (x - mu) / sigma**2
def gaussian_derivative_sigma(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) * (x - mu)**2 / sigma**3

def get_kmeans_centers_for_rbf(df, n_clusters):
    # Initialize a KMeans model with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    
    # Fit the model to the data
    kmeans.fit(df)
    
    # Return the centers of each cluster as a numpy array
    return kmeans.cluster_centers_


'''def get_closest_center(features, centers):
    distances = np.sqrt(np.sum((features[:, np.newaxis, :] - centers) ** 2, axis=2))
    return np.argmin(distances, axis=1)


# calculate the average distance between each center and all data points in that cluster
def calculate_average_distance(center, cluster):
    distances = np.linalg.norm(center - cluster, axis=1)
    return np.mean(distances)

# calculate initial sigma for each cluster
def calculate_initial_sigma(features, centers):
    sigmas = []
    for i in range(len(centers)):
        cluster = features[np.argwhere(get_closest_center(features, centers) == i)[:, 0]]
        sigma_i = calculate_average_distance(centers[i], cluster)
        sigmas.append(sigma_i)
    return np.array(sigmas)'''

