import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import CLDNeuralNetwork as cldnn


def Train(outputPath):
    print("- Загрузка датасета Iris...")
    iris = load_iris()

    y = pd.get_dummies(iris.target).values
    X_norm = (iris.data - iris.data.mean(0)) / iris.data.std(0)
    X_train, _, y_train, _ = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    print("- Создание нейронной сети: 4 -> 12 (LeakyReLU) -> 3 (Softmax)")
    nn = cldnn.NeuralNetwork(
        [
            cldnn.NeuralLayer(
                4,
                12,
                activation=cldnn.LeakyReLU,
                activation_derivative=cldnn.LeakyReLU_derivative,
            ),
            cldnn.NeuralLayer(12, 3, activation=cldnn.Softmax),
        ]
    )
    print(f"- Начало обучения: {len(X_train)} примеров, 1000 эпох, lr=0.01...")
    nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)
    nn.save_csv(outputPath)
    print(f"- Модель сохранена в {outputPath}")


def Predict(modelPath, data):
    nn = cldnn.NeuralNetwork(
        [
            cldnn.NeuralLayer(
                4,
                12,
                activation=cldnn.ReLUActivation,
                activation_derivative=cldnn.ReLU_derivative,
            ),
            cldnn.NeuralLayer(12, 3, activation=cldnn.Softmax),
        ]
    )
    nn.load_csv(modelPath)

    iris = load_iris()
    X_norm = (data - iris.data.mean(0)) / iris.data.std(0)
    result = nn.feedForward(X_norm)
    return result


def ParsePredict(data):
    if len(data) < 3:
        sys.exit(1)
    modelPath = data[1]
    data_str = data[2]
    parsed_data = np.array([float(x) for x in data_str.split(",")], ndmin=2)
    return modelPath, parsed_data


def ParseTrain(data):
    if len(data) < 3:
        sys.exit(1)
    return data[2]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "train":
        Train(ParseTrain(sys.argv))
        pass
    elif cmd == "predict":
        Predict(ParsePredict(sys.argv))
        pass
