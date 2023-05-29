from keras.datasets import mnist
import tensorflow as tf
from matplotlib import pyplot
import scipy
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy


def main() -> None:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

if __name__ == "__main__":
    main()
