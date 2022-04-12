from math import inf
import math
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
import numpy as np
import itertools

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def mse(y, h):
    sum = 0
    for i in range(len(y)):
        sum += (y[i] - h[i]) ** 2
    mse = sum / len(y)
    return mse

def lambdas2alpha(params):
    a_l1 = []
    for i in range(len(params)):
        if params[i][0] == 0 or params[i][0] > params[i][1]: a_l1.append(None)
        else: 
            alpha = params[i][0] + params[i][1]
            l1 = params[i][0]/alpha
            a_l1.append([alpha,l1])
    return a_l1

def bestMSE(X, y, models):
    MSE = []
    hypothesis = []
    smallestError = inf
    bestIndex = None
    for i in range(len(models)):
        model = models[i]
        if model is not None:
            h = model.predict(X)
            hypothesis.append(h)
            error = mse(y, h)
            MSE.append(error)
            if error < smallestError: 
                smallestError = error
                bestIndex = i
        else:
            MSE.append(None)
            hypothesis.append(None)
    return bestIndex, MSE, hypothesis  

def trainModels(X, y, a_l1):
    models = []
    for i in range(len(a_l1)):
        if a_l1[i] is not None:
            alpha = a_l1[i][0]
            l1_ratio = a_l1[i][1]
            model = ElasticNet(alpha = alpha, l1_ratio=l1_ratio, tol = .1)
            model.fit(X, y)
            models.append(model)
        else: models.append(None)
    return models

def printMSE(params, MSE):
    for i in range(len(MSE)):
        if MSE[i] is not None: print("Lambda1:",params[i][0],"Lambda2:",params[i][1],"MSE",MSE[i])

def printBestMSE(params, MSE, bestIndex):
    print("Best - Lambda1:",params[bestIndex][0],"Lambda2:",params[bestIndex][1],"MSE:",MSE[bestIndex])

def getXY(data, num_features):
    X = data[:,0:num_features-1]
    y = data[:,num_features]
    return X, y

def exercise(training_data, validation_data, testing_data, num_features):
    #alpha * l1_ratio
    lambda1 = [0,.000001,.00001,.0001,.001,.01,.1,1]
    #alpha * (1-l1_ratio)
    lambda2 = [0,.000001,.00001,.0001,.001,.01,.1,1]

    ##Get valid a_l1 values
    params = list(itertools.product(lambda1, lambda2))
    a_l1 = lambdas2alpha(params)

    #a) Fit an elastic net model to the training data with each possible combination of the following L1 and L2 regularization weights.
    X, y = getXY(training_data, num_features)
    training_Models = trainModels(X, y, a_l1)

    #b) For each model trained in step (a), make a prediction for each training example, using the
    # predict method for sklearn.linear_model.ElasticNet and calculate the mean squared
    # error (MSE) on the training examples.
    print("---Training Data---")
    training_bestIndex, training_MSE, training_h = bestMSE(X, y, training_Models)
    printMSE(params, training_MSE)

    #c) Make a prediction for each validation example
    # and calculate the mean squared error on the validation examples
    print("---Validation Data---")
    X, y = getXY(validation_data, num_features)

    validation_Models = trainModels(X, y, a_l1)
    validation_bestIndex, validation_MSE, validation_h = bestMSE(X, y, validation_Models)
    printMSE(params, validation_MSE)  

    #d) Which model (i.e., pair of λ1 and λ2) performed best on the training data?
    print("---Best MSE---")
    print("Training Data:")
    printBestMSE(params, training_MSE, training_bestIndex)
    #Which model performed best on the validation data?
    print("Validation Data:")
    printBestMSE(params, validation_MSE, validation_bestIndex)

    #e)Find the best hyperparameter set (pair of λ1 and λ2) on the validation data. Train a model with
    #the same λ1 and λ2 on both the training and validation data. 
    #Using this model, make predictions for each testing example and calculate the mean squared error on the test examples.

    combined_data = np.append(training_data, validation_data, axis = 0)
    X, y = getXY(combined_data, num_features)

    best_a_l1 = [a_l1[validation_bestIndex]]
    combined_model = trainModels(X, y, best_a_l1)
    
    #Testing Data
    X, y = getXY(testing_data, num_features)
    testing_bestIndex, testing_MSE, testing_h = bestMSE(X, y, combined_model)
    print("---Testing Data---")
    print("MSE:", testing_MSE[0])

def printConfusionMatrix(cm):
    print("True Negative:",cm[0][0])
    print("False Positive:",cm[0][1])
    print("False Negative:",cm[1][0])
    print("True Positive:", cm[1][1])

#Exercise 2

#Get data from CSVs
sc_training_data = np.loadtxt("SCTraining.csv",skiprows=1,delimiter=",")
sc_validation_data = np.loadtxt("SCValidation.csv",skiprows=1,delimiter=",")
sc_testing_data = np.loadtxt("SCTesting.csv",skiprows=1,delimiter=",")

exercise(sc_training_data,sc_validation_data,sc_testing_data, 81)

#Exercise 3

#Get data from CSVs
stab_training_data = np.loadtxt("StabTraining.csv",skiprows=1,delimiter=",")
stab_validation_data = np.loadtxt("StabValidation.csv",skiprows=1,delimiter=",")
stab_testing_data = np.loadtxt("StabTesting.csv",skiprows=1,delimiter=",")

exercise(stab_training_data,stab_validation_data,stab_testing_data, 11)

#Exercise 4
def loss(y, h):
    if y == h: return 1
    else: return 0

def emprisk(y, h):
    sum = 0
    for i in range(len(y)):
        sum += loss(y[i], h[i])
    return sum / len(y)

def confusionMatrix(y, h):
    true_neg = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    for i in range(len(y)):
        if y[i] == h[i]:
            if h[i] == 0: true_neg += 1
            else: true_pos += 1
        else:
            if h[i] == 0: false_neg += 1
            else: false_pos += 1
    return [[true_neg, false_pos], [false_neg, true_pos]]

print("---Validation Data Logistic Regression---")
#a)
regularModel = LogisticRegression()
nonRegularModel = LogisticRegression(penalty='none')
X, y = getXY(stab_training_data, 11)
regularModel.fit(X, y)
nonRegularModel.fit(X, y)

#b)

X, y = getXY(stab_validation_data, 11)
#Using the two models created in part (a) make a prediction for each validation example. W
h_reg = regularModel.predict(X)
h_non = nonRegularModel.predict(X)

reg_risk = emprisk(y, h_reg)
non_risk = emprisk(y, h_non)

print("L2 regularization model empirical risk:", reg_risk)
print("No regularization model empirical risk:", non_risk)

reg_cm = confusionMatrix(y, h_reg)
non_cm = confusionMatrix(y, h_non)

print("---Confusion Matrix---")
print("L2 Regularization Model Confusion Matrix")
printConfusionMatrix(reg_cm)
print("No Regularization Model Confusion Matrix")
printConfusionMatrix(non_cm)

print("---Best performance---")
if reg_risk < non_risk: print("L2 Regularized model is better with risk of :", reg_risk," < ", non_risk)
else: print("Non Regularized model is better with risk of :", non_risk," < ", reg_risk)

print("---Training Data Logistic Regression using L2 Regularized Model---")
#c part 2)
stab_combined_data = np.append(stab_training_data, stab_validation_data, axis = 0)
X, y = getXY(stab_combined_data, 11)
combined_reg_model = LogisticRegression()
regularModel.fit(X, y)

X, y = getXY(stab_training_data, 11)
h_test_reg = regularModel.predict(X)
test_risk = emprisk(y, h_test_reg)
test_cm = confusionMatrix(y, h_test_reg)

print("L2 regularization model empirical risk:", test_risk)
print("L2 Regularization Model Confusion Matrix")
printConfusionMatrix(test_cm)