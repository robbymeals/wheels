import math
import csv


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def hypothesis(theta, x):
    z = sum([theta[i]*x[i] for i in xrange(len(theta))])
    return sigmoid(z)


def cost(theta, x_i, y_i):
    h_i = hypothesis(theta, x_i)
    j_i = -1*y_i * log(h_i) - (1-y_i * log(1-h_i))
    return j_i


def costAll(theta, X, y):
    J = sum([cost(theta, X[i], y[i]) for i in xrange(len(X))])/len(X)
    return J


def gradient(theta, X, y):
    grad = [sum([(hypothesis(theta, X[i]) - y[i])*X[i][j] 
        for i in xrange(len(X))])/float(len(X)) 
        for j in xrange(len(theta))]
    return grad 


def gradient_descent_step(alpha, theta, X, y):
    update = [alpha*val for val in gradient(theta, X, y)]
    theta = [theta[j] - update[j] for j in xrange(len(update))]
    return theta


def get_features(r):
    male = float(r['sex'] == 'male')
    alone = float(all([r['sibsp'] == '0', r['parch'] == '0']))
    pclass_2 = float(r['pclass'] == '2')
    pclass_3 = float(r['pclass'] == '3')
    features = [1, male, alone, pclass_2, pclass_3]
    return features


def get_performance(pred_act):
    acc = float(sum([pa[0]==pa[1] for pa in pred_act]))/len(pred_act)
    p_0 = float(sum([all([pa[0]==0.0, pa[1]==0.0]) 
        for pa in pred_act]))/sum([pa[0]==0.0 for pa in pred_act])
    r_0 = float(sum([all([pa[0]==0.0, pa[1]==0.0]) 
        for pa in pred_act]))/sum([pa[1]==0.0 for pa in pred_act])
    f_0 = (2*(p_0*r_0))/(p_0+r_0)
    p_1 = float(sum([all([pa[0]==1.0, pa[1]==1.0]) 
        for pa in pred_act]))/sum([pa[0]==1.0 for pa in pred_act])
    r_1 = float(sum([all([pa[0]==1.0, pa[1]==1.0]) 
        for pa in pred_act]))/sum([pa[1]==1.0 for pa in pred_act])
    f_1 = (2*(p_1*r_1))/(p_1+r_1)
    metrics = [acc, p_0, r_0, f_0, p_1, r_1, f_1]
    return metrics


if __name__ == '__main__':
    f = open('data/titanic3.csv','rb')
    w = csv.reader(f)
    h = w.next()
    titanic = [dict(zip(h,r)) for r in w]
    X = [get_features(r) for r in titanic]
    y = [float(r['survived']) for r in titanic]
    theta = [0 for i in xrange(len(X[0]))]
    iterations = 5000
    alpha = 0.02
    for i in xrange(iterations):
        theta = gradient_descent_step(0.01, theta, X, y)

    y_preds = [float(hypothesis(theta, X[i])>0.5) for i in xrange(len(X))]
    pred_act = zip(y, y_preds)
    performance = get_performance(pred_act)
    print [round(t,2) for t in theta]
    print [round(m,2) for m in performance]




