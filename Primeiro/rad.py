import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load df
path = os.path.join(os.path.dirname(__file__), "..", "_DATA_", "Iris.csv")
df = pd.read_csv(path)
df = ut.fn_cat_onehot(df)

# select the feature columns
features = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
features_array = np.array(features)

# select the label columns
labels = df[
    ["Species_Iris-setosa", "Species_Iris-versicolor", "Species_Iris-virginica"]
]
labels_array = np.array(labels)

# Split data into train and test sets with equal proportion of each class
X_train, X_test, y_train, y_test = train_test_split(
    features_array, labels_array, test_size=0.4, stratify=labels_array, random_state=42, shuffle=True
)

# Parameters
N = 0.001
EPOCHS = 100000
cost = []

# Network architecture
input_neurons = features_array.shape[1]  # 4 features
hidden_neurons = 5  # 5 neurons with gaussian activation
output_neurons = labels_array.shape[1]  # 3 neurons with sigmoid activation


# Random weights
w_output = np.random.uniform(size=(hidden_neurons, output_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

# mu and sigma
mu, sigma = ut.get_kmeans_centers_for_rbf(df=features, n_clusters=hidden_neurons)
# sigma = np.ones((hidden_neurons))

# Training
for epoch in tqdm(range(EPOCHS)):
    # Forward propagation
    activation_hidden = ut.gaussian(x=X_train, mu=mu, sigma=sigma)
    activation_output = ut.sigmoid(np.dot(activation_hidden, w_output) + b_output)
    cost.append(ut.classification_error(y_true=y_train, y_pred=activation_output))

    # Backpropagation
    delta_output_w = (y_train - activation_output) * ut.sigmoid_derivative(
        activation_output
    )
    delta_output_b = delta_output_w.sum(axis=0, keepdims=True)

    # Update weights
    w_output += N * activation_hidden.T.dot(delta_output_w)
    b_output += N * delta_output_b

# Plot
'''corretos, errados = ut.calc_accuracy(true_labels=labels_array, pred_labels=activation_output)
print("Previsões corretas: {}, Previsões erradas: {}".format(corretos, errados))

plt.plot(cost)
plt.title("Taxa de aprendizado: {}".format(N))
plt.xlabel("Épocas")
plt.ylabel("Custo")
plt.show()'''

# Plot
print("Final Error: {}".format(cost[-1]))

# Test
activation_hidden_test = ut.gaussian(x=X_test, mu=mu, sigma=sigma)
activation_output_test = ut.sigmoid(np.dot(activation_hidden_test, w_output) + b_output)

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(activation_output_test, axis=1)

print("Classification report test:")
ut.get_classification_metrics(y_true=y_true, y_pred=y_pred)

# Training
activation_hidden_train = ut.gaussian(x=X_train, mu=mu, sigma=sigma)
activation_output_train = ut.sigmoid(
    np.dot(activation_hidden_train, w_output) + b_output
)

y_true = np.argmax(y_train, axis=1)
y_pred = np.argmax(activation_output_train, axis=1)

print("Classification report train:")
ut.get_classification_metrics(y_true=y_true, y_pred=y_pred)

plt.plot(cost)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.grid()
plt.show()