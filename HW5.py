from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
import numpy as np

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
X = training_data[:,0:80]
y = training_data[:,81]

models = []
for i in range(len(lambda1)):
    model = ElasticNet(alpha = (lambda1[i]/2), l1_ratio=.5, random_state=0)
    model.fit(X, y)
    models.append(model)

#X = validation_data[:,0:80]
#y = validation_data[:,81]

for model in models:
    h = model.predict(X)
    print(mse(y, h))

    


# regr = ElasticNet(random_state=0)
# regr.fit(X, y)
# print(regr.coef_)
