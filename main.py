import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import CLDNeuralNetwork as cldnn


def main():
    iris = load_iris()
    y = pd.get_dummies(iris.target).values

    # Нормализация + train/test
    X_norm = (iris.data - iris.data.mean(0)) / iris.data.std(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42
    )

    nn = cldnn.NeuralNetwork(
        [cldnn.NeuralLayer(4, 12), cldnn.NeuralLayer(12, 3)],
        cldnn.ReLUActivation,
        cldnn.Softmax,
    )
    nn.train(X_train, y_train)

    # Тест на датасете
    pred = nn.feedForward(X_test)
    predicted_classes = np.argmax(pred, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)

    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"✅ Общая точность: {accuracy * 100:.1f}%")

    # Ручной ввод
    print("\n🖱️ РУЧНОЙ ВВОД ДЛЯ ПРЕДСКАЗАНИЯ")
    print("Введите 4 измерения Iris (в см):")
    print("sepal length, sepal width, petal length, petal width")

    while True:
        try:
            line = input("Данные (4 числа через пробел) или 'q' для выхода: ").strip()
            if line.lower() == "q":
                break

            values = np.array(list(map(float, line.split()))).reshape(1, -1)

            # Нормализация по тем же параметрам
            mean = iris.data.mean(0)
            std = iris.data.std(0)
            values_norm = (values - mean) / std

            prediction = nn.feedForward(values_norm)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]

            class_names = ["setosa", "versicolor", "virginica"]
            print(
                f"🌸 Предсказание: **{class_names[predicted_class]}** "
                f"({confidence:.1%} уверенности)"
            )
            print(
                f"   Вероятности: setosa={prediction[0]:.1%}, "
                f"versicolor={prediction[1]:.1%}, virginica={prediction[2]:.1%}"
            )
            print()

        except ValueError:
            print("❌ Введите 4 числа через пробел!")
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break


if __name__ == "__main__":
    main()
