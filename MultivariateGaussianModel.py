#********************Imports*****************
import numpy as np
from scipy.stats import norm, multivariate_normal

#********************Helper Functions*****************
def mean(X) :
    """Calculates and returns the means of the given data.
    X = a NumPy matrix containing the data [ no. of training inputs X no. of features per example ]"""

    return np.sum(X, axis = 0) / X.shape[0]

def co_variance(X, mu) :
    """Calculates the co_variances for the features.
    X = a NumPy matrix containing the data [ no. of training inputs X no. of features per example ]
    mu = a NumPy matrix containing the means for the features [ 1 X no. of features per example ]"""

    sigma = np.dot((X - mu).T, X - mu) / X.shape[0]

    return sigma

def return_multivariate_gaussian_value(X, mu, sigma) :
    """Returns the value of probability using Gaussian Distribution for the given input\s.
    X = a NumPy Matrix containing the inputs [ no. of training inputs X no. of features per example ]
    mu = a NumPy matrix containing the means for the features [ 1 X no. of features per example ]
    sigma = a NumPy matrix containing the variances for the features [ 1 X no. of features per example ]"""

    return multivariate_normal.pdf(X, mean = mu, cov = sigma)

#********************Classes*****************
class MultivariateGaussianModel(object) :
    def __init__(self) :
        """Initializes the model."""

        #Initializing the model parameters
        self.parameters = []
        self.epsilon = 0.05

    def train(self, X, Y, epsilon = 0.05) :
        """Trains the model on the training data on training data.
        X = a NumPy matrix containing the training data [ no. of training inputs X no. of features per example ]
        Y = a NumPy matrix containing the expected labels for the training inputs [ no. of training inputs X 1 ] 
        epsilon = the max probability of anomaly
        return (F1, Precision, Recall)"""

        #Saving the value of epsilon used for training
        self.epsilon = epsilon

        #Calculating the model parameters
        self.calculate_parameters(X)

        #Calculating and returning the model's accuracy parameters
        return self.get_accuracy_parameters(X, Y, epsilon)

    def calculate_parameters(self, X) :
        """Calculates the parameters for the distribution function.
        X = a NumPy matrix containing the training data [ no. of training inputs X no. of features per example ]
        """

        #Getting the means for the features
        self.parameters.append(mean(X))
        self.parameters[0].shape = (1, self.parameters[0].shape[0])

        #Getting the variances for the features
        self.parameters.append(co_variance(X, self.parameters[0]))


    def get_accuracy_parameters(self, X, Y, epsilon) :
        """Returns the model's accuracy parameters i.e F1, Precision and Recall, for the given inputs
        X = a NumPy matrix containing the training data [ no. of training inputs X no. of features per example ]
        Y = a NumPy matrix containing the expected labels for the training inputs [ no. of training inputs X 1 ] 
        return (F1, Precision, Recall)
        """
        
        #Getting the input probabilities
        input_probabilities = self.pdf(X)

        pos = input_probabilities < epsilon #The inputs which have been classified as anomalies
        pos.shape = (pos.size,1)

        #Calculating the True Positives, False Positives, False Negatives 
        tp = np.sum(np.logical_and(pos == 1, Y == 1)) #The True Positives i.e examples correctly classified as anomalies
        fp = np.sum(np.logical_and(pos == 1, Y == 0)) #The False Positives i.e examples incorrectly classified as anomalies
        fn = np.sum(np.logical_and(pos == 0, Y == 1)) #The False Negatives i.e examples misclassified as normal

        #Calculating Precision and Recall
        precision = tp / (tp + fp) 
        recall = tp / (tp + fn)

        #Calculating the models F1 score
        f1 = (2 * precision * recall) / (precision + recall)

        return (f1, precision, recall)

    def pdf(self, X) :
        """"Returns the value of the probability-density-function for the given inputs. Assumes that the mean and variance have already been calculated.
        X = a NumPy matrix containing the training data [ no. of training inputs X no. of features per example ]
        returns inputs_probabilities = a NumPy matrix containing the probabilities of the given inputs [ no. of training inputs X 1 ]
        """

        return return_multivariate_gaussian_value(X, self.parameters[0].ravel(), self.parameters[1])

    def predict(self, X, epsilon=None) :
        """Returns the model's prediction for whether the given inputs represent an anomaly.
        X = a NumPy matrix containing the inputs [ no. of inputs X no. of features per input ]
        epsilon = the max probability of anomaly"""

        #Checking if a value has been provided for epsilon, else using the value set during training
        if(epsilon == None) :
            epsilon = self.epsilon

        probabilities = self.pdf(X) #Getting the probabilities for the inputs

        return probabilities < epsilon