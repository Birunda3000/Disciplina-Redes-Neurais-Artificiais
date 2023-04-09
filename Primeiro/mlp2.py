import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from IPython.display import display
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load df
path = os.path.join(os.path.dirname(__file__), "..", "_DATA_", "Iris.csv")
df = pd.read_csv(path)
df = ut.fn_cat_onehot(df)
df.insert(0, "Bias", 1)
display(df.head())

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
input_neurons = features_array.shape[1]  # 4
hidden_neurons = 5
output_neurons = labels_array.shape[1]  # 3

# Random weights
w_hidden_1 = np.random.uniform(size=(input_neurons, hidden_neurons))
w_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Treinamento
for i in tqdm(range(EPOCHS)):
    # Feedforward
    activation_hidden_1 = ut.sigmoid(np.dot(X_train, w_hidden_1))
    activation_output = ut.sigmoid(np.dot(activation_hidden_1, w_output))
    cost.append(ut.classification_error(y_true=y_train, y_pred=activation_output))

    # Backpropagation
    delta_output = (y_train - activation_output) * ut.sigmoid_derivative(
        activation_output
    )
    delta_hidden_1 = delta_output.dot(w_output.T) * ut.sigmoid_derivative(
        activation_hidden_1
    )

    # Update weights
    w_output += np.dot(activation_hidden_1.T, delta_output) * N
    w_hidden_1 += np.dot(X_train.T, delta_hidden_1) * N

# Plot
print("Final classification error: ", cost[-1])

# Test
activation_hidden_test = ut.sigmoid(np.dot(X_test, w_hidden_1))
activation_output_test = ut.sigmoid(np.dot(activation_hidden_test, w_output))

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(activation_output_test, axis=1)

print("Classification report test:")
ut.get_classification_metrics(y_true=y_true, y_pred=y_pred)

# Training
activation_hidden_train = ut.sigmoid(np.dot(X_train, w_hidden_1))
activation_output_train = ut.sigmoid(np.dot(activation_hidden_train, w_output))

y_true = np.argmax(y_train, axis=1)
y_pred = np.argmax(activation_output_train, axis=1)

print("Classification report train:")
ut.get_classification_metrics(y_true=y_true, y_pred=y_pred)

plt.plot(cost)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.grid()
plt.show()