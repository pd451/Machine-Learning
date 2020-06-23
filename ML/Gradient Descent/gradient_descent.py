import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pickle
import random

class gradient_descent():
    def __init__(self, n_features, out_dimension):
        self.Theta = np.random.rand(out_dimension,n_features);
        self.bias = np.zeros((out_dimension,1));
        self.num_features = n_features;

    def get_gradient(self,X,Y):
        T_gradient = np.zeros(self.Theta.shape);
        b_gradient = np.zeros(self.bias.shape);
        TS = T_gradient.shape;
        BS = b_gradient.shape;
        m = X.shape[0];
        n = self.num_features;
        out = TS[0];

        for i in range(0,TS[0]):
            vec_Y = Y[:,i];
            vec_W = self.Theta[i,:];

            vec_Y = vec_Y.reshape((m,1));
            vec_W = vec_W.reshape((1,n));

            vec_T = np.matmul(X,np.transpose(vec_W));


            assert(vec_Y.shape == (m,1));
            assert(vec_W.shape == (1,n));
            assert(vec_T.shape == (m,1));

            for j in range(0,TS[1]):
                vec_T1 = X[:,j].reshape((m,1));
                vec_T2 = vec_Y - vec_T - self.bias[i];
                vec_T3 = np.multiply(vec_T2, vec_T1);

                assert(vec_T1.shape == (m,1));
                assert(vec_T2.shape == (m,1));
                assert(vec_T3.shape == (m,1));

                T_gradient[i][j] = -2 * np.sum(vec_T3, axis=0);

            vec_T1 = vec_Y - vec_T - self.bias[i];

            assert(vec_T1.shape == (m,1));

            b_gradient[i] = -2 * np.sum(vec_T1, axis=0);


        return T_gradient, b_gradient

    def compute_cost(self, X,Y):
        m = X.shape[0];
        out = Y.shape[1];

        temp = np.transpose(self.bias);
        vec_T1 = Y - np.matmul(X,np.transpose(self.Theta)) - temp;
        vec_T1 = np.square(vec_T1);

        assert(temp.shape == (1,out));
        assert(vec_T1.shape == (m,out));

        T1 = np.sum(vec_T1,axis=0);
        T2 = np.sum(T1,axis=0);

        assert(T2.shape == ());

        return T2;

    def predict(self,X):
        vec_T1 = np.matmul(X,np.transpose(self.Theta)) + np.transpose(self.bias);
        return vec_T1;


    def train(self,X,Y,num_iter,learning_rate):
        num_examples = X.shape[0];

        for i in range(0,num_iter):
            cost = self.compute_cost(X,Y);

            T_gradient, b_gradient = self.get_gradient(X,Y);
            self.Theta -= learning_rate * T_gradient;
            self.bias -= learning_rate * b_gradient;

            print("Iteration " + str(i) +  ": Cost = " + str(cost));


def main():
    radius = 1;
    x_train, y_train, x_test, y_test = load_data();

    X_shape = x_train.shape;


    m = X_shape[0];
    m1 = x_test.shape[0];

    y_train = y_train.reshape((m,1));
    y_test = y_test.reshape((m1,1));

    Y_shape = y_train.shape;

    nf = X_shape[1];
    out_dim = Y_shape[1];

    epochs = 50000;
    learning_rate = .000002;

    x_train = x_train / (100 * radius);
    x_test = x_test / (100 * radius);

    model = gradient_descent(nf,out_dim);

    model.train(x_train, y_train, epochs, learning_rate);

    Ypred = model.predict(x_test);

    limit = 10;

    acc = evaluate_accuracy(Ypred, y_test);
    print(acc);
    print(Ypred[:limit].astype(int));
    print(Ypred[:limit]);
    print(y_test[:limit]);

def train_test_split(X,Y,frac):
    if frac > 1 or frac <= 0:
        raise Exception('Improper Value for frac');
    m = len(X);
    k = (int) (frac * m);
    S = random_subset(m,k);
    Sc = [];
    for i in range(0,m):
        if i not in S:
            Sc.append(i);

    return np.array([X[i] for i in S]) , np.array([Y[i] for i in S]) , np.array([X[i] for i in Sc]) , np.array([Y[i] for i in Sc]);

def random_subset(n,k):
    if k <= 0 or n <= 0:
        print("k must be >= 0");
        return [];

    if k >= n:
        return [i for i in range(0,n)];

    result = [0 for i in range(0,k)];

    temp = [i for i in range(0,n)];

    for i in range(0,k):
        p = random.uniform(0,1);
        j = (int)((n-i) * p);
        t1 = temp[j];
        temp2 = [0 for i in range(0,n-i-1)];
        for i1 in range(0,j):
            temp2[i1] = temp[i1];
        for i2 in range(j+1,n-i):
            temp2[i2-1] = temp[i2];
        temp = temp2;
        result[i] = t1;
    return list(np.sort(result));

def generate_data_clusters(cluster_size, num_cluster, dimension, radius, points):
    n = len(points);
    if n != num_cluster:
        raise Exception('Number of Points and Clusters do not match');

    X,Y = [],[];
    count = 0;
    delta = np.divide(radius, np.sqrt(dimension));
    for i in range(0,num_cluster):
        for j in range(0,cluster_size):
            temp = [0 for i1 in range(0,dimension)];
            for k in range(0,dimension):
                p = 2 * random.uniform(0,1) - 1;
                temp[k] = points[i][k] - np.multiply(p , delta);
            X.append(np.array(temp));
            Y.append(i);
    return np.array(X), np.array(Y);

def generate_points(num_points, dimension, radius):
    result = [];
    delta = np.divide(radius, np.sqrt(dimension));
    for i in range(0,num_points):
        temp = [0 for i in range(0,dimension)];
        for k in range(0,dimension):
            p = 2 * random.uniform(0,1) - 1;
            temp[k] = np.multiply(p , delta);
        result.append(np.array(temp));
    return result;

def load_data():
    radius = 1;
    num_points = 10;
    num_features = 10;
    cluster_size = 300;

    points = generate_points(num_points, num_features, 100 * radius);
    X,Y = generate_data_clusters(cluster_size, num_points, num_features, radius, points);
    return train_test_split(X,Y,frac=0.8);

def evaluate_accuracy(Y_pred,Y_true):
    n = len(Y_pred);
    count = 0;
    for i in range(0,n):
        pred = np.rint(Y_pred[i]);
        if pred != Y_true[i]:
            count += 1;
    return 1 - (float) (count) / (float) (n);

if __name__ == '__main__':
    main();
