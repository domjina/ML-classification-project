import tensorflow as tf
from matplotlib import pyplot
import scipy
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy
import os

MODEL_SAVE_NAME = "data/handwritten.model"

def main() -> None:
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
            tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
        )

        model.compile(
            optimizer="adam", 
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(train_X, train_y, epochs=3)

        model.save(MODEL_SAVE_NAME)
    
    model = tf.keras.models.load_model(MODEL_SAVE_NAME)
    model.summary()
    loss, accuracy = model.evaluate(test_X, test_y)
    print(loss)
    print(accuracy)


if __name__ == "__main__":
    main()
