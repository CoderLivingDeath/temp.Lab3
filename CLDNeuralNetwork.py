import os
import pickle
from typing import List

import numpy as np
import pandas as pd


def LeakyReLU(inputs, alpha=0.01):
    return np.where(inputs > 0, inputs, alpha * inputs)


def LeakyReLU_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


def ReLUActivation(inputs):
    return np.maximum(0, inputs)


def ReLU_derivative(z):
    return (z > 0).astype(float)


def Softmax(inputs):
    exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)


ACTIVATIONS = {
    "LeakyReLU": LeakyReLU,
    "ReLU": ReLUActivation,
    "Softmax": Softmax,
}


def mse_loss(output, y_true):
    return np.mean((output - y_true) ** 2)


def mse_grad(output, y_true):
    return 2 * (output - y_true) / y_true.shape[0]


class NeuralLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

        self.inputs = None
        self.z = None
        self.a = None
        self.activation_name = activation

        if activation:
            self.activation = ACTIVATIONS[activation]
        else:
            self.activation = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        if self.activation:
            self.a = self.activation(self.z)
            return self.a
        return self.z

    def backward(self, grad_output, learning_rate):
        grad_z = grad_output

        grad_weights = np.dot(self.inputs.T, grad_z)
        self.weights -= learning_rate * grad_weights

        grad_bias = np.sum(grad_z, axis=0, keepdims=True)
        self.bias -= learning_rate * grad_bias

        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input


class NeuralNetwork:
    def __init__(self, layers: List[NeuralLayer]):
        self.layers = layers

    def feedForward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            output = self.feedForward(X)
            loss = mse_loss(output, y)

            grad = mse_grad(output, y)
            total_grad = 0

            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)
                total_grad += np.mean(np.abs(grad))

            print(f"Эпоха {epoch}, Loss: {loss:.4f}, Grad: {total_grad:.6f}")

    def save_csv(self, filename):
        data_rows = []

        data_rows.append(
            {
                "type": "HEADER",
                "layer": len(self.layers),
                "input_activation": self.layers[0].activation_name,
                "output_activation": self.layers[-1].activation_name,
            }
        )
        # Каждый слой
        for i, layer in enumerate(self.layers):
            # Размеры слоя
            data_rows.append(
                {
                    "type": "LAYER_SHAPE",
                    "layer": i,
                    "input_size": layer.weights.shape[0],
                    "output_size": layer.weights.shape[1],
                }
            )

            # Веса слоя
            for r in range(layer.weights.shape[0]):
                for c in range(layer.weights.shape[1]):
                    data_rows.append(
                        {
                            "type": f"WEIGHTS_L{i}",
                            "layer": i,
                            "row": r,
                            "col": c,
                            "value": layer.weights[r, c],
                        }
                    )

            # Bias слоя
            for c in range(layer.bias.shape[1]):
                data_rows.append(
                    {
                        "type": f"BIAS_L{i}",
                        "layer": i,
                        "row": 0,
                        "col": c,
                        "value": layer.bias[0, c],
                    }
                )

        df = pd.DataFrame(data_rows)
        df.to_csv(filename, index=False)
        print(f"Model saved to {filename}")

    def load_csv(self, filename):
        """Загружает слои из CSV файла"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found!")

        df = pd.read_csv(filename)

        # Читаем заголовок
        header_row = df[df["type"] == "HEADER"].iloc[0]
        layers_count = int(header_row["layer"])
        input_activation = header_row["input_activation"]
        output_activation = header_row["output_activation"]
        self.layers = []

        activation_map = {0: input_activation, 1: output_activation}

        for i in range(layers_count):
            # Размеры слоя
            shape_row = df[df["type"] == f"LAYER_SHAPE"].iloc[i]
            input_size = int(shape_row["input_size"])
            output_size = int(shape_row["output_size"])

            # Создаём слой с активацией
            act = activation_map.get(i)
            layer = NeuralLayer(input_size, output_size, activation=act)

            # Загружаем веса
            weights_data = df[(df["type"] == f"WEIGHTS_L{i}")].sort_values(
                ["row", "col"]
            )
            weights = np.zeros((input_size, output_size))
            for _, row in weights_data.iterrows():
                r, c = int(row["row"]), int(row["col"])
                weights[r, c] = row["value"]
            layer.weights = weights

            # Загружаем bias
            bias_data = df[(df["type"] == f"BIAS_L{i}")].sort_values("col")
            bias = np.zeros((1, output_size))
            for _, row in bias_data.iterrows():
                c = int(row["col"])
                bias[0, c] = row["value"]
            layer.bias = bias

            self.layers.append(layer)
