import tensorflow as tf
from matplotlib import pyplot as plt
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import random
import seaborn as sns
from collections import Counter

random.seed(2023)

MODEL_SAVE_NAME = "data/handwritten.model"


def neural_network() -> None:
    mnist = tf.keras.datasets.mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = tf.keras.utils.normalize(train_X, axis=1)
    test_X = tf.keras.utils.normalize(test_X, axis=1)

    if not (os.path.exists(MODEL_SAVE_NAME)):
        model = tf.keras.models.Sequential()

        model.add(
            tf.keras.layers.Flatten(input_shape=(28, 28), name="input_layer")
        )

        model.add(
            tf.keras.layers.Dense(128, activation="relu", name="layer_2")
        )

        model.add(
            tf.keras.layers.Dense(128, activation="relu", name="layer_3")
        )

        model.add(
            tf.keras.layers.Dense(
                10, activation="softmax", name="output_layer")
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(train_X, train_y, epochs=3)

        model.save(MODEL_SAVE_NAME)

    model = tf.keras.models.load_model(MODEL_SAVE_NAME)
    if model is not None:
        model.summary()
        loss, accuracy = model.evaluate(test_X, test_y)
        print(loss)
        print(accuracy)
    else:
        raise ValueError(f"Could not load model {MODEL_SAVE_NAME}")


def k_means_clustering(k: int) -> tuple[float, float, float, float]:
    KNN_train_X = np.array(train_X).reshape(60000, 784)
    NUM_OF_CENTROIDS = k
    centroids = KNN_train_X[np.random.choice(KNN_train_X.shape[0], NUM_OF_CENTROIDS, replace = False)]
    delta = 10000
    counter = 0
    while np.all(delta > 1e-7):
        counter += 1
        distances = KNN_train_X.dot(centroids.T)
        labels = np.argmax(distances, axis=1)
        new_centroids = np.zeros((NUM_OF_CENTROIDS, KNN_train_X.shape[1]))

        for i in range(NUM_OF_CENTROIDS):
            indices = np.where(labels == i)[0]
            if len(indices) > 0:
                mean_vec = np.sum(KNN_train_X[indices], axis=0) / len(indices)
            else:
                mean_vec = np.zeros(KNN_train_X.shape[1])
            new_centroids[i] = mean_vec

        delta = np.linalg.norm(new_centroids - centroids, axis=1)
        centroids = new_centroids

    label_names = [i for i in range(0,10)]
    labels_in_centroids = [[0]*NUM_OF_CENTROIDS for _ in range(len(label_names))]

    distances = KNN_train_X.dot(centroids.T)
    labels = np.argmax(distances, axis=1)

    for i, centroid in enumerate(centroids):
        indices = np.where(labels == i)[0]
        for idx in indices:
            labels_in_centroids[train_y[idx]][i] += 1

    # label_counts = np.array(labels_in_centroids)
    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.heatmap(label_counts, annot=True, fmt=".0f", ax=ax)
    # plt.xlabel("Cluster")
    # plt.ylabel("Number")
    # plt.show()
    # print(label_counts)

    KMN_test_X = np.array(test_X).reshape(10000, 784)
    distances = KMN_test_X.dot(centroids.T)
    labels = np.argmax(distances, axis=1)
    y_pred = np.zeros(KMN_test_X.shape[0], dtype=int)
    for i in range(NUM_OF_CENTROIDS):
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            majority_label = np.argmax(np.bincount(train_y[indices]))
            y_pred[indices] = majority_label

    accuracy = accuracy_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred, average='macro')
    precision = precision_score(test_y, y_pred, average='macro', zero_division=0)
    f1 = f1_score(test_y, y_pred, average='macro')
    return accuracy, precision, recall, f1 # type: ignore

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = tf.keras.utils.normalize(train_X, axis=1)
    test_X = tf.keras.utils.normalize(test_X, axis=1)

    # neural_network()
    best_k = 0
    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for k in range(1, 21):
        print(f"Run: {k}")
        accuracy, precision, recall, f1 = k_means_clustering(k)
        if f1 > best_f1:
            best_k = k
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    print(f'Best results: k={best_k}, accuracy={best_accuracy:.2f}, precision={best_precision:.2f}, recall={best_recall:.2f}, f1={best_f1:.2f}')
