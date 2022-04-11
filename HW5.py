from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
import numpy as np
import itertools

def mse(y, h):
    sum = 0
    for i in range(len(y)):
        sum += (y[i] - h[i]) ** 2
    mse = sum / len(y)
    return mse

training_data = np.loadtxt("train.csv",skiprows=1,delimiter=",")
validation_data = np.loadtxt("validate.csv",skiprows=1,delimiter=",")
testing_data = np.loadtxt("test.csv",skiprows=1,delimiter=",")
#alpha * l1_ratio
lambda1 = [0,.000001,.00001,.0001,.001,.01,.1,1]
#alpha * (1-l1_ratio)
lambda2 = [0,.000001,.00001,.0001,.001,.01,.1,1]
params = list(itertools.product(lambda1, lambda2))

##Get valid a_l1 values
a_l1 = []
for i in range(len(params)):
    if params[i][0] == 0 or params[i][0] < params[i][1]: continue
    else: 
        alpha = params[i][0] + params[i][1]
        l1 = params[i][0]/alpha
        a_l1.append([alpha,l1])

X = training_data[:,0:80]
y = training_data[:,81]

models = []
for i in range(len(a_l1)):
    alpha = a_l1[i][0]
    l1_ratio = a_l1[i][1]

    model = ElasticNet(alpha = alpha, l1_ratio=l1_ratio)
    model.fit(X, y)
    models.append(model)

#a) MSE for training examples
for model in models:
    h = model.predict(X)
    print(mse(y, h))

#b) MSE for validation examples
X = validation_data[:,0:80]
y = validation_data[:,81]
for model in models:
    h = model.predict(X)
    #print(mse(y, h))