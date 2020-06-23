from sklearn.datasets.samples_generator import make_blobs;
import matplotlib.pyplot as plt;
from sklearn.svm import SVC; # "Support Vector Classifier"
import numpy as np;
import random;

class multi_SVC:
    def __init__(self,num_classes):
        self.num_classes = num_classes;
        self.classifiers = [];
        for i in range(0,num_classes):
            t1 = SVC(kernel='linear');
            self.classifiers.append(t1);

    def train(self,X,Y):
        n = self.num_classes;
        for i in range(0,n):
            Ty = split_helper(Y,i);
            self.classifiers[i].fit(X,Ty);

    def predict(self,X):
        n = self.num_classes;
        m = len(X);
        temp = [[1 for i in range(0,n)] for j in range(0,m)];

        for i in range(0,n):
            T = self.classifiers[i].predict(X);
            for j in range(0,m):
                if T[j] == 0:
                    temp[j][i] = 0;

        result = [];
        for i in range(0,m):
            t1 = predict_helper(temp[i],n);
            result.append(t1);

        return np.array(result);

#if part of multiple class return predict with equal probability
def predict_helper(temp,num_classes):
    temp2 = [];
    for i in range(0,num_classes):
        if temp[i] == 1:
            return i;

    return (int) (random.uniform(0,1) * num_classes);

#Splits class by i and (not i)
def split_helper(Y,i):
    Tx = [];
    Ty = [];
    n = len(Y);
    for j in range(0,n):
        if Y[j] == i:
            Ty.append(1);
        else:
            Ty.append(0);

    return np.array(Ty);
# Generate clusters in given dimension such that diameter of cluster in R^d <= 2 * radius
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

def evaluate_accuracy(Y_pred,Y_true):
    n = len(Y_pred);
    count = 0;
    for i in range(0,n):
        if Y_pred[i] != Y_true[i]:
            count += 1;
    return 1 - (float) (count) / (float) (n);

def main():
    cluster_size = 50;
    num_cluster = 4;
    dimension = 2;
    radius = 0.5;
    frac = 0.8;

    points = generate_points(num_cluster,dimension, 200 * radius);
    X,Y = generate_data_clusters(cluster_size, num_cluster, dimension, 50 * radius, points);

    Xtrain,Ytrain,Xtest,Ytest = train_test_split(X,Y,frac);
    model = multi_SVC(num_cluster);

    model.train(Xtrain,Ytrain);

    Ypred = model.predict(Xtest);

    acc = evaluate_accuracy(Ypred,Ytest);

    print(len(Xtest));
    print(len(Xtrain));

    print("Accuracy is : " + str(acc));


    # plotting scatters
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=Ytest, s=50, cmap='spring');
    plt.show()

    # plotting scatters
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=Ypred, s=50, cmap='spring');
    plt.show()






if __name__ == '__main__':
    main();
