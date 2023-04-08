import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from utils import gaussian, gaussian_derivative_x, gaussian_derivative_mu, gaussian_derivative_sigma, sigmoid, sigmoid_derivative
from IPython.display import display
from tqdm import tqdm

# Load df
path = os.path.join(os.path.dirname(__file__), "..", "_DATA_", "Iris.csv")
df = pd.read_csv(path)
df = ut.fn_cat_onehot(df)
df.insert(0, "Bias", 1)
display(df.head())

# Select the feature columns
features = df[["Bias","SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
features_array = np.array(features)

# Select the label columns
labels = df[
    ["Species_Iris-setosa", "Species_Iris-versicolor", "Species_Iris-virginica"]
]
labels_array = np.array(labels)

# Parameters
N = 0.00001
EPOCHS = 50000
cost = []

# Define the number of input, hidden and output neurons
input_neurons = features_array.shape[1]  # 4 number of features
hidden_neurons_1 = 5
output_neurons = labels_array.shape[1]  # 3 number of classes

# Initialize the weights with random values
w_hidden_1 = np.random.uniform(size=(input_neurons, hidden_neurons_1))
w_output = np.random.uniform(size=(hidden_neurons_1, output_neurons))

# mu and sigma
mu = np.random.uniform(size=(1, hidden_neurons_1))
sigma = np.random.uniform(size=(1, hidden_neurons_1))

# Training
for i in tqdm(range(EPOCHS)):
    # Feedforward
    activation_hidden_1 = gaussian(np.dot(features_array, w_hidden_1), mu, sigma)
    activation_output = sigmoid(np.dot(activation_hidden_1, w_output))
    cost.append(ut.classification_error(y_true=labels_array, y_pred=activation_output))

    # Backpropagation
    delta_output = (labels_array - activation_output) * sigmoid_derivative(activation_output)
    delta_hidden_1 = delta_output.dot(w_output.T) * gaussian_derivative_x(activation_hidden_1, mu, sigma)

    # Update the weights
    w_output += np.dot(activation_hidden_1.T, delta_output) * N
    w_hidden_1 += np.dot(features_array.T, delta_hidden_1) * N

    # Update mu and sigma
    delta_mu = np.sum(delta_output.dot(w_output.T) * gaussian_derivative_mu(activation_hidden_1, mu, sigma), axis=0)
    delta_sigma = np.sum(delta_output.dot(w_output.T) * gaussian_derivative_sigma(activation_hidden_1, mu, sigma), axis=0)

    mu += delta_mu * N
    sigma += delta_sigma * N



# Plot
print("Custo final: {}".format(cost[-1]))

corretos, errados = ut.calc_accuracy(true_labels=labels_array, pred_labels=activation_output)
print("Previsões corretas: {}, Previsões erradas: {}".format(corretos, errados))

plt.plot(cost)
plt.title("Taxa de aprendizado: {}".format(N))
plt.xlabel("Épocas")
plt.ylabel("Custo")
plt.show()
