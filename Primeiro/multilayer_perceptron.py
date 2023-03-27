import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from IPython.display import display
from tqdm import tqdm

# Load df
path = os.path.join(os.path.dirname(__file__), "..", "_DATA_", "Iris.csv")
df = pd.read_csv(path)
df = ut.fn_cat_onehot(df)
display(df.head())

# selecione as colunas de recursos
features = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
# transforme em um array numpy
features_array = np.array(features)

# selecione as colunas de label
labels = df[
    ["Species_Iris-setosa", "Species_Iris-versicolor", "Species_Iris-virginica"]
]
# transforme em um array numpy
labels_array = np.array(labels)

# Parametros
N = 0.01
momentum = 0.5
EPOCHS = 1000
cost = []

# Defina o número de neurônios de entrada, ocultos e de saída
input_neurons = features_array.shape[1]  # 4
hidden_neurons = 9
output_neurons = labels_array.shape[1]  # 3

# Inicialize os pesos com valores aleatórios
# Pesos
w_hidden_1 = np.random.uniform(size=(input_neurons, hidden_neurons))
w_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Bias
b_hidden_1 = np.random.uniform(size=(1, hidden_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

# Treinamento
for i in tqdm(range(EPOCHS)):
    # Feedforward
    activation_hidden_1 = ut.sigmoid(np.dot(features_array, w_hidden_1) + b_hidden_1)
    activation_output = ut.sigmoid(np.dot(activation_hidden_1, w_output) + b_output)
    cost.append(ut.classification_error(y_true=labels_array, y_pred=activation_output))

    # Backpropagation
    delta_output = (labels_array - activation_output) * ut.sigmoid_derivative(
        activation_output
    )
    delta_hidden_1 = delta_output.dot(w_output.T) * ut.sigmoid_derivative(
        activation_hidden_1
    )

    #momentum_w_output = momentum * (w_output - (activation_hidden_1.T.dot(delta_output) * N))
    #momentum_w_hidden_1 = momentum * (w_hidden_1 - (features_array.T.dot(delta_hidden_1) * N))
    
    #momentum_b_output = momentum * (b_output - (np.sum(delta_output, axis=0, keepdims=True) * N))
    #momentum_b_hidden_1 = momentum * (b_hidden_1 - (np.sum(delta_hidden_1, axis=0, keepdims=True) * N))

    # Atualize os pesos
    w_output += np.dot(activation_hidden_1.T, delta_output) * N
    w_hidden_1 += np.dot(features_array.T, delta_hidden_1) * N

    # Atualize o bias
    b_output += np.sum(delta_output, axis=0, keepdims=True) * N
    b_hidden_1 += np.sum(delta_hidden_1, axis=0, keepdims=True) * N

# Plot
print("Custo final: {}".format(cost[-1]))

corretos, errados = ut.calc_accuracy(true_labels=labels_array, pred_labels=activation_output)
print("Previsões corretas: {}, Previsões erradas: {}".format(corretos, errados))

plt.plot(cost)
plt.title("Taxa de aprendizado: {} - Momentum: {}".format(N, momentum))
plt.xlabel("Épocas")
plt.ylabel("Custo")
plt.show()