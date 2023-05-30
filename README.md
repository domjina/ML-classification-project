# machine-learning-project

This is a small project in which I create, test and evaluate machine learning models on the "mnist" dataset to classify handwritten digits.

## neural network

Here, I used the TensorFlow sequential model. This allows me to specify a neural network from input to output, passing through a series of neural layers, one after the other.

The input layer is a the gray-scale values of each pixel of each image (784 neurons). 
There are then two dense layers with 128 neurons, using the "Rectified Linear Unit" (ReLU) activation function. A dense layer means that all neurons are connected to all other neurons in the previous layer. ReLU is a piecewise linear function that outputs the input directly if positive and zero otherwise.

The final layer (output) is a dense layer using the softmax activation function. Softmax maps the input from the previous layer to the classes specified in the dataset.

Following on, I compile the neural network and fit the data, using the "adam optimizer" (Adaptive Moment Estimation). This is an extension to stochastic gradient descent which is used to minimize the error of the model. We then calculate the loss using "sparse categorical cross entropy" and calculate the accuracy of the model. Additionally, we use 3 epochs to train the model (3 iterations of training).

The results are as follows:

loss: 0.09478... (how well the NN models the training data)

accuracy: 0.97 (ratio between correct predictions and total predictions)
