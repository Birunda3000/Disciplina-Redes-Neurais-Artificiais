from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


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

def gaussian(x, mu, sigma):
    """Gaussian function with mean mu and standard deviation sigma"""
    return np.exp(-(x - mu)**2 / (2 * sigma**2))
def gaussian_derivative_x(x, mu, sigma):
    return -np.exp(-(x - mu)**2 / (2 * sigma**2)) * (x - mu) / sigma**2
def gaussian_derivative_mu(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) * (x - mu) / sigma**2
def gaussian_derivative_sigma(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) * (x - mu)**2 / sigma**3
