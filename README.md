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


## K-means-clustering

For K-means-clustering, I created my own solution based on a similar, previous exercise that I had done. K-means-clustering works by creating k-many centroids (or manually selecting points), calculating the distance from all points to the centroids and then assigning you data points to the nearest centroid. That group of data points then becomes a cluster. The centroids are then recalculated, taking the mean point of the cluster. The data points are then all reassigned to clusters based on the nearest centroids. The process is done iteratively until the centroids no longer change (or move a very small distance).

Once I had programmed that, I calculated the distance between my test set to the centroids. I counted the majority labels in each cluster and that then became the label of the cluster. Then a data point had been assigned correctly if the label of the testing data point was that of the cluster label. Finally, I iterated this process with different numbers of centroids (from 1 to 20). I used the same random seed (2023) to keep this fiar.

The results are as follows:

training dataset size: (60000, 784)

test dataset size: (10000, 784)

Best number of centroids: 19

Accuracy: 0.20

Recall: 0.19

Precision: 0.16

F1-score: 0.15

As you can see, the results are not great. This is likely due to the input size of each image, being 784 features. As there are so many dimensions, the data points become very sparse and therefore become hard to cluster as all data points can appear equally far apart from each other making it hard to group similar points.

After getting the previous results, I looked for ways to reduce dimensionality of the dataset to try and improve the previous metrics. I came across "principle component analysis" (PCA). PCA works by subtracting the mean of each feature from the data points, computing the covariance matrix of the centered data, calculating eigenvectors and eigenvalues, choosing the top k eigenvectors with the largest eigenvalues and then projects the data onto the new basis by computing the dot product between the data points and the top k eigenvectors. This gives a lower dimensional representation of your original data whilst capturing most of the variation.

The results are as follows:

training dataset size: (60000, 50)

test dataset size: (10000, 50)

Best number of centroids: 7

Accuracy: 0.31

Recall: 0.29

Precision: 0.17

F1-score: 0.20

Unfortunately, accuracy, recall, precision and f1 scores are still quite low. This could mean that k-means clustering is not well suited for this particular dataset. k-means algorithm assumes that clusters are spherical and equally sized which may not be the case.
