## beautiful toy neural network from http://iamtrask.github.io/2015/07/12/basic-python-network/
import numpy as np
import csv
f = open('data/titanic3.csv','r')
w = csv.reader(f)
titanic = [r for r in w]
titanic = [dict(zip(titanic[0], titanic[i])) for i in range(1, len(titanic))]
titanic = [x for x in titanic if x['age'] != '']
f.close()
mean_age = np.mean([float(x['age']) for x in titanic])
std_age = np.std([float(x['age']) for x in titanic])
X = np.vstack([ np.array([
        (float(x['age']) - mean_age)/std_age,
        float(x['sex']=='female'), 
        float(x['pclass']=='1')
        ]) for x in titanic])
y = np.array([float(x['survived']) for x in titanic]).T
syn0 = 2*np.random.random(X.shape).T - 1
syn1 = 2*np.random.random(y.shape) - 1
for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
    if j%50==0:
        print(np.mean([a==b for a,b in zip(y, l2>0.5)]))
