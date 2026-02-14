import os
from typing import List, Optional

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
    "ReLU": (ReLUActivation, ReLU_derivative),
    "Softmax": (Softmax, None),
}


def mse_loss(output, y_true):
    return np.mean((output - y_true) ** 2)


def mse_grad(output, y_true):
    return 2 * (output - y_true) / y_true.shape[0]


class NeuralLayer:
    def __init__(self, input_size, output_size, activation: Optional[str] = None):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

        self.inputs = None
        self.z = None
        self.a = None
        self.activation = activation

    def __str__(self):
        str = ""
        str += f"Layer Weights:\n {self.weights}\n"
        str += f"Layer Bias:\n {self.bias}\n"
        return str

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        if self.activation:
            self.a = ACTIVATIONS[self.activation][0](self.z)
            return self.a
        return self.z

    def backward(self, grad_output, learning_rate):
        if self.activation and ACTIVATIONS[self.activation][1]:
            grad_z = grad_output * ACTIVATIONS[self.activation][1](self.z)
        else:
            grad_z = grad_output

        assert self.inputs is not None, "forward() must be called first"
        grad_weights = np.dot(self.inputs.T, grad_z)
        self.weights -= learning_rate * grad_weights

        grad_bias = np.mean(grad_z, axis=0, keepdims=True)
        self.bias -= learning_rate * grad_bias

        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input


class NeuralNetwork:
    def __init__(self, layers: Optional[List[NeuralLayer]] = None):
        self.layers = layers if layers else []

    def add_layer(self, layer):
        self.layers.append(layer)

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

    # nn
    # layer1
    # - weights
    # - bias
    # - activation
    # layer2
    # - weights
    # - bias
    # - activation
    # ...

    def save_csv(self, filename):
        columns = [
            "type",
            "Layer",
            "activation",
            "input_size",
            "output_size",
            "row",
            "col",
            "value",
        ]

        data = []

        data.append(
            [
                "HEADER",
                len(self.layers),
                None,
                self.layers[0].weights.shape[0],
                self.layers[-1].weights.shape[1],
                None,
                None,
                None,
            ]
        )

        for i, layer in enumerate(self.layers):
            data.append(
                [
                    "LAYER",
                    i,
                    layer.activation,
                    layer.weights.shape[0],
                    layer.weights.shape[1],
                    None,
                    None,
                    None,
                ]
            )

            for r in range(layer.weights.shape[0]):
                for c in range(layer.weights.shape[1]):
                    data.append(
                        [
                            "WEIGHT",
                            i,
                            None,
                            None,
                            None,
                            r,
                            c,
                            layer.weights[r, c],
                        ]
                    )

            for c in range(layer.bias.shape[1]):
                data.append(
                    [
                        "BIAS",
                        i,
                        None,
                        None,
                        None,
                        0,
                        c,
                        layer.bias[0, c],
                    ]
                )

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filename)

    def load_csv(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found!")

        df = pd.read_csv(filename)
        self.layers = []

        header_row = df[df["type"] == "HEADER"].iloc[0]  # type: ignore
        layers_count = int(header_row["Layer"])  # type: ignore

        for i in range(layers_count):
            layer_row = df[(df["type"] == "LAYER") & (df["Layer"] == i)].iloc[0]  # type: ignore
            activation = layer_row["activation"]  # type: ignore
            input_size = int(layer_row["input_size"])  # type: ignore
            output_size = int(layer_row["output_size"])  # type: ignore

            layer = NeuralLayer(input_size, output_size, activation=activation)

            weights_data = df[(df["type"] == "WEIGHT") & (df["Layer"] == i)]
            for _, row in weights_data.iterrows():
                r = int(row["row"])  # type: ignore
                c = int(row["col"])  # type: ignore
                layer.weights[r, c] = row["value"]  # type: ignore

            bias_data = df[(df["type"] == "BIAS") & (df["Layer"] == i)]
            for _, row in bias_data.iterrows():
                c = int(row["col"])  # type: ignore
                layer.bias[0, c] = row["value"]  # type: ignore

            self.layers.append(layer)
