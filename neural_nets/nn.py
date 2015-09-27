## beautiful toy neural network from http://iamtrask.github.io/2015/07/12/basic-python-network/
import numpy as np
X = np.array([
    [1,0,1],
    [1,1,1],
    [0,0,1],
    [0,1,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
    ])
y = np.array([
    [1,0,1,0,1,1,0]
    ]).T
syn0 = 2*np.random.random(X.shape).T - 1
syn1 = 2*np.random.random(y.shape) - 1
for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
    if j%5000==0:
        print(l2)