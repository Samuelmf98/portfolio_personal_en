#Convolutional_Neural_Network_Dog_Cat_

# Purpose
The purpose of this project is the construction of a convolutional neural network to classify images of dogs and cats.
# Input
The entrance is two folders, made up of 4,000 images of dogs and 4,000 images of cats.

# Parameters
The parameters used to read the images, both for train/test and for prediction, are found in the file "prm_Convolutional_Neural_Network_Dog_Cat_".

# Neural Network
For the construction of the Convolutional Neural Network, Tensorflow was used. The input layer was configured with 32 3x3 feature maps. Then a "Max_Pooling" is added to calculate the maximum of the subsets of pixels. To flatten the images into a vector, the "Flattening" layer is used and then an artificial neural network is created for the fully connected layer with a "relu" (unitary linear rectifier) ​​activation function. For the output layer, the sigmoid activation function was used to show the probability that an image is classified as a dog or a cat.

The compilation was done with the "rmsprop" optimizer, the chosen loss function is "binary_crossentropy" to minimize the error and the "accuracy" metric to measure the percentage of accuracy. During training a "batch_size" of 32 was used to update the weights after each iteration. 25 epochs were carried out.

# Results
An accuracy of 81% was obtained with the chosen parameters.

A test is carried out with random images taken from the Internet to verify the efficiency of our model for classifying images of dogs and cats. A label is applied to the results obtained, resulting in 1 for dogs and 0 for cats. Of 10 displayed images, it correctly predicts 8 and fails in 2.

# Observations


# Technician
Samuel Marcos Fernandez