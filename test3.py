#********************Imports*****************
import numpy as np
from UnivariateGaussianModel import UnivariateGaussianModel
from MultivariateGaussianModel import MultivariateGaussianModel
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import isnan

#********************Functions*****************
def use_univariate_model(X_train, X_val, Y_val) :

    print("Using Univariate model")

    #Initializing the model
    model = UnivariateGaussianModel()

    #Training the model
    model.calculate_parameters(X_train)

    #Selecting the ideal value of epsilon
    p_val = model.pdf(X_val) #Getting the model's probability values for the cross-validation set
    min_pval = np.min(p_val) #The min probability of any input
    max_pval = np.max(p_val) #The max probability of any input
    stepsize = (max_pval - min_pval) / 1000
    ideal_parameters = [0, 0, 0]
    ideal_epsilon = 0
    for epsilon in np.arange(min_pval, max_pval, stepsize) :
        params = list(model.get_accuracy_parameters(X_val, Y_val, epsilon))

        if(not isnan(params[0]) and params[0] > ideal_parameters[0]) :
            ideal_parameters = params
            ideal_epsilon = epsilon

    #Printing the ideal parameters and epsilon
    print(f"The ideal epsilon is {ideal_epsilon} for F1 = {ideal_parameters[0]}, Precision = {ideal_parameters[1]}, Recall = {ideal_parameters[2]}")
    

def use_multivariate_model(X_train, X_val, Y_val) :
    
    print("Using Multivariate model")

    #Initializing the model
    model = MultivariateGaussianModel()

    #Training the model
    model.calculate_parameters(X_train)

    #Selecting the ideal value of epsilon
    p_val = model.pdf(X_val) #Getting the model's probability values for the cross-validation set
    min_pval = np.min(p_val) #The min probability of any input
    max_pval = np.max(p_val) #The max probability of any input
    stepsize = (max_pval - min_pval) / 1000
    ideal_parameters = [0, 0, 0]
    ideal_epsilon = 0
    for epsilon in np.arange(min_pval, max_pval, stepsize) :
        params = list(model.get_accuracy_parameters(X_val, Y_val, epsilon))

        if(not isnan(params[0]) and params[0] > ideal_parameters[0]) :
            ideal_parameters = params
            ideal_epsilon = epsilon

    #Printing the ideal parameters and epsilon
    print(f"The ideal epsilon is {ideal_epsilon} for F1 = {ideal_parameters[0]}, Precision = {ideal_parameters[1]}, Recall = {ideal_parameters[2]}")


#********************Script Commands*****************

#Loading the data
data_file = loadmat("test3_data.mat")
X_train = data_file["X"]
X_val = data_file["Xval"]
Y_val = data_file["yval"]

#Applying a anomaly detection model
use_univariate_model(X_train, X_val, Y_val)
use_multivariate_model(X_train, X_val, Y_val)
