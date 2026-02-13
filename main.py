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
    X_norm = (iris.data - iris.data.mean()) / iris.data.std()
    X_train, _, y_train, _ = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    print("- Создание нейронной сети: 4 -> 12 (LeakyReLU) -> 3 (Softmax)")
    nn = cldnn.NeuralNetwork(
        [
            cldnn.NeuralLayer(4, 12, activation="LeakyReLU"),
            cldnn.NeuralLayer(12, 3, activation="Softmax"),
        ]
    )
    print(f"- Начало обучения: {len(X_train)} примеров, 10000 эпох, lr=0.001...")
    nn.train(X_train, y_train, epochs=10000, learning_rate=0.001)
    nn.save_csv(outputPath)
    print(f"- Модель сохранена в {outputPath}")


def Predict(modelPath, data):
    nn = cldnn.NeuralNetwork(
        [
            cldnn.NeuralLayer(4, 12, activation="ReLU"),
            cldnn.NeuralLayer(12, 3, activation="Softmax"),
        ]
    )
    nn.load_csv(modelPath)

    iris = load_iris()
    X_norm = (data - iris.data.mean(0)) / iris.data.std(0)
    result = nn.feedForward(X_norm)
    return result


def ParsePredict(data):
    if len(data) < 4:
        sys.exit(1)
    modelPath = data[2]
    data_str = data[3]
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
        modelPath, data = ParsePredict(sys.argv)
        r = Predict(modelPath, data)
        classes = ["setosa", "versicolor", "virginica"]
        predicted = classes[np.argmax(r)]
        print(f"- Входные данные: {data}")
        print(f"- setosa: {r[0][0]:.4f}")
        print(f"- versicolor: {r[0][1]:.4f}")
        print(f"- virginica: {r[0][2]:.4f}")
        print(f"- Предсказание: {predicted}")
        pass
