import random
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# гиперпараметры
INPUT_DIM = 4
OUT_DIM = 3
H_DIM = 10
ALPHA = 0.002
NUM_EPOCHS = 1000
BATCH_SIZE = 50

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    # axis = 1 - сумма эл-ов по строкам
    # на выходе матрица (батч на кол-во класса)
    return out / np.sum(out)

def softmax_batch(t):
    out = np.exp(t)
    # axis = 1 - сумма эл-ов по строкам
    # на выходе матрица (батч на кол-во класса)
    return out / np.sum(out, axis = 1, keepdims = True)

def sparse_cross_entropy_batch(z, y):
    # считаем кросс-энтропию для каждого эл-та батча
    # в полном векторе правильного ответа мы имеем везде 0 и только в 1 месте 1, то от суммы останется одна компонента в-ра z
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

def to_full_batch(y, num_classes):
    # matrix из 0
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

def relu_deriv(t):
    return (t >= 0).astype(float)

iris = datasets.load_iris()
# подготовка данных (в-р строка с признаками и индекс правильного класса)
# [None,...] - позволяет сделать именно в-р строку
dataset = [(iris.data[i][None,...], iris.target[i]) for i in range(len(iris.target))]
# print(dataset)
# exit()

# np.random.randn - массив случайных зн-ий с нормальным распределением
# ввиду наличия 2х полносвязных слоев, нам надо 2 матрицы весов и 2 вектора сдвига
# перед обучением инициализация случайными значениями
W1 = np.random.rand(INPUT_DIM, H_DIM)  # матрица
b1 = np.random.rand(1, H_DIM)  # вектор - строка
W2 = np.random.rand(H_DIM, OUT_DIM)  # матрица
b2 = np.random.rand(1, OUT_DIM)  # вектор - строка

W1 = (W1 - 0.5) * 2 * np.square(1 / INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.square(1 / INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.square(1 / H_DIM)
b2 = (b2 - 0.5) * 2 * np.square(1 / H_DIM)

loss_arr = []

for ep in range(NUM_EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset) // BATCH_SIZE):
        # есть список кортежей, которые нужно разделить
        # i * BATCH_SIZE : BATCH_SIZE * i + BATCH_SIZE - от и до какого
        batch_x, batch_y = zip(*dataset[i * BATCH_SIZE : BATCH_SIZE * i + BATCH_SIZE])
        x = np.concatenate(batch_x, axis = 0)
        y = np.array(batch_y)
        # x - вектор с признаками (при работе с batch: матрица - набор вектора строк)
        # y - правильный ответ (при работе с batch: вектор, размерв-ра = р-ру batch)

        # прямое распространение (Forward)

        # @ - матричное умножение
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        # вектор вероятности
        z = softmax_batch(t2)
        # ошибка - разреженная кросс-энтропия (у - не вектор, а индекс правильного класса)
        # финальная ошибка - сумма ошибок на различных эл-тах batch
        E = np.sum(sparse_cross_entropy_batch(z, y))
        loss_arr.append(E)

        # обратное распространение (Backward)

        # превращает индекс правильного класса в соответствующее распределение (в-р из 0 и 1)
        y_full = to_full_batch(y, OUT_DIM)
        # выводим все найденные параметры (https://www.youtube.com/watch?v=bW4dKxtUFpg)
        # при использовании batch все формулы: https://www.youtube.com/watch?v=bXGBeRzM87g
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        # axis = 0 - сумма эл-ов в столбцах
        dE_db2 = np.sum(dE_dt2, axis = 0, keepdims = True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis = 0, keepdims = True)

        # делаем один шаг градиентного спуска и обновляем параметры

        # мы делаем смещение в сторону антиградиента, поэтому знак "-"
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    # вектор вероятности
    z = softmax(t2)
    return z

# вычисление точности (кол-во правильно угаданных образцов / длину dataset)
def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        # получаем индекс класса
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc

accuracy = calc_accuracy()
print("Accuracy: ", accuracy)
# график ошибки по итерациям
plt.plot(loss_arr)
plt.show()
