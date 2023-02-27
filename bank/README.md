#Artificial_Neural_Network_Bank

# purpose
The purpose of this project is the construction of an artificial neural network to predict the probability that a client abandons the bank.

# Input
The input is a dataset in .csv format that contains the data of 10,000 customers whose target column called exited returns 1 if the customer leaves the bank and 0 if they stay.

# Parameters
The parameters used to read the file are found in the "prm_Artificial_Neural_Network_Bank" file.

# Neural Network
For the construction of the Artificial Neural Network, Tensorflow was used. The input layer was configured with a uniform weight distribution and the activation function "relu" (unitary linear rectifier). The settings from the previous layer were kept in the second hidden layer. For the output layer, the sigmoid activation function was used to show the probability that a customer will leave the bank. A 10% dropout is placed after each layer to avoid overfitting.

The compilation was done with the "rmsprop" optimizer, the chosen loss function is "binary_crossentropy" to minimize the error and the "accuracy" metric to measure the percentage of accuracy. During training a "batch_size" of 32 was used to update the weights after each iteration. 150 epochs were carried out.

# Results
A precision of 83.1% was obtained with the chosen parameters.
An accuracy of 83.4% was obtained with cross validation.

A threshold is applied to the results obtained, thus resulting in only two results, "True" for those who have more than a 50% probability of leaving the bank and "False" for those who have less than 50%.

A test is performed with the data of a client that does not appear in the input dataset to verify the effectiveness of the ANN and provides a 9% probability of abandoning the bank.

# Observations
The necessary code is provided to elaborate a "GridSearchCV" in order to find the parameters that maximize the precision. Due to the high costs of time and computational consumption, this step has not been carried out.

# Technician
Samuel Marcos Fernandez