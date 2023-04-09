import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from IPython.display import display
from tqdm import tqdm
import utils as ut
from sklearn.cluster import KMeans

# Load df
path = os.path.join(os.path.dirname(__file__), "..", "_DATA_", "Iris.csv")
df = pd.read_csv(path)
df = ut.fn_cat_onehot(df)

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

# Parameters
N = 0.001
EPOCHS = 1000
cost = []

# Network architecture
input_neurons = features_array.shape[1]  # 4 features
hidden_neurons = 5 # 5 neurons with gaussian activation
output_neurons = labels_array.shape[1]  # 3 neurons with sigmoid activation


# Random weights
w_output = np.random.uniform(size=(hidden_neurons, output_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

# mu and sigma
mu, sigma = ut.get_kmeans_centers_for_rbf(df=features, n_clusters=hidden_neurons)
#sigma = np.ones((hidden_neurons))

# Training
for epoch in tqdm(range(EPOCHS)):
    # Forward propagation
    activation_hidden = ut.gaussian(x=features, mu=mu, sigma=sigma)
    activation_output = ut.sigmoid(np.dot(activation_hidden, w_output) + b_output)
    cost.append(ut.classification_error(y_true=labels_array, y_pred=activation_output))

    # Backpropagation
    delta_output_w = (labels_array - activation_output) * ut.sigmoid_derivative(
        activation_output
    )
    delta_output_b = delta_output_w.sum(axis=0, keepdims=True)

    # Update weights
    w_output += N * activation_hidden.T.dot(delta_output_w)
    b_output += N * delta_output_b

# Plot
print("Custo final: {}".format(cost[-1]))

corretos, errados = ut.calc_accuracy(
    true_labels=labels_array, pred_labels=activation_output
)
print("Previsões corretas: {}, Previsões erradas: {}".format(corretos, errados))

plt.plot(cost)
plt.title("Taxa de aprendizado: {}".format(N))
plt.xlabel("Épocas")
plt.ylabel("Custo")
plt.show()
