import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import CLDNeuralNetwork as cldnn


def main():
    iris = load_iris()
    y = pd.get_dummies(iris.target).values
    X_norm = (iris.data - iris.data.mean(0)) / iris.data.std(0)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)

    nn = cldnn.NeuralNetwork(
        [cldnn.NeuralLayer(4, 12), cldnn.NeuralLayer(12, 3)],
        cldnn.ReLUActivation,
        cldnn.Softmax,
    )
    nn.train(X_train, y_train)

    pred = nn.feedForward(X_test)

    # подсчет точности
    correct = 0
    total = len(X_test)
    for i in range(total):
        true_class = 0
        max_true = y_test[i][0]
        for j in range(1, 3):
            if y_test[i][j] > max_true:
                max_true = y_test[i][j]
                true_class = j

        pred_class = 0
        max_pred = pred[i][0]
        for j in range(1, 3):
            if pred[i][j] > max_pred:
                max_pred = pred[i][j]
                pred_class = j

        if pred_class == true_class:
            correct = correct + 1

    accuracy = correct / total
    print(f"\nТочность: {accuracy * 100:.1f}% ({total} тестов)")

    # Первые 5 предсказаний
    print("Первые предсказания:")
    class_names = ["setosa", "versicolor", "virginica"]
    for i in range(5):
        # Находим истинный класс
        true_class = 0
        for j in range(3):
            if y_test[i][j] == 1:
                true_class = j
                break

        # Находим предсказанный класс
        pred_class = 0
        max_prob = pred[i][0]
        for j in range(1, 3):
            if pred[i][j] > max_prob:
                max_prob = pred[i][j]
                pred_class = j

        true_name = class_names[true_class]
        pred_name = class_names[pred_class]
        status = "OK" if true_class == pred_class else "ERROR"
        print(f"  {true_name} → {pred_name} {max_prob:.1%} {status}")


if __name__ == "__main__":
    main()
