import os
import pickle
from typing import List

import numpy as np
import pandas as pd


def ReLUActivation(inputs):
    return np.maximum(0, inputs)


def Softmax(inputs):
    exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)


def ReLU_derivative(z):
    return (z > 0).astype(float)


def softmax_cross_entropy_derivative(output, y_true):
    # Для softmax + cross-entropy: dL/dz = output - y_true
    return (output - y_true) / y_true.shape[0]


class NeuralLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

        self.inputs = None
        self.z = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
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
    def __init__(
        self,
        layers: List[NeuralLayer],
        activation=ReLUActivation,
        outputActivation=Softmax,
    ):
        self.layers = layers
        self.activation = activation
        self.outputActivation = outputActivation

    def feedForward(self, inputs):
        current = inputs
        for i, layer in enumerate(self.layers):
            current = layer.forward(current)
            if i < len(self.layers) - 1:
                current = self.activation(current)
            else:
                current = self.outputActivation(current)
        return current

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            output = self.feedForward(X)

            loss = -np.mean(y * np.log(output + 1e-8))

            grad = softmax_cross_entropy_derivative(output, y)

            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]

                if i < len(self.layers) - 1:
                    grad = grad * ReLU_derivative(layer.z)

                grad = layer.backward(grad, learning_rate)

            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, Loss: {loss:.4f}")

    # ==== AI ====
    def save_csv(self, filename):
        data_rows = []

        data_rows.append(
            {
                "type": "HEADER",
                "layer": len(self.layers),
                "input_activation": self.activation.__name__,
                "output_activation": self.outputActivation.__name__,
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
        print(f"✅ Сеть сохранена в {filename}")

    def load_csv(self, filename):
        """Загружает слои из CSV файла"""
        if not os.path.exists(filename):
            print(f"❌ Файл {filename} не найден!")
            return

        df = pd.read_csv(filename)

        # Читаем заголовок
        header_row = df[df["type"] == "HEADER"].iloc[0]
        layers_count = int(header_row["layer"])
        self.layers = []

        for i in range(layers_count):
            # Размеры слоя
            shape_row = df[df["type"] == f"LAYER_SHAPE"].iloc[i]
            input_size = int(shape_row["input_size"])
            output_size = int(shape_row["output_size"])

            # Создаём пустой слой
            layer = NeuralLayer(input_size, output_size)

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

        print(f"✅ Загружено {layers_count} слоёв из {filename}")
