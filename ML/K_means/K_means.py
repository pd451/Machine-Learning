from sklearn.datasets.samples_generator import make_blobs;
import matplotlib.pyplot as plt;
from sklearn.svm import SVC; # "Support Vector Classifier"
import numpy as np;
import random;
import seaborn as sns;


def initialization(X,K):
    n = len(X);
    S = random_subset(n,K);
    return [X[i] for i in S];

def assignment(X,clusters):
    n = len(X);
    K = len(clusters);

    ids = [-1 for i in range(0,n)];

    X = np.array(X);
    clusters = np.array(clusters);

    for i in range(0,n):
        min_dist = float('inf');
        index = -1;
        for j in range(0,K):
            d = compute_distance(X[i,:], clusters[j,:]);
            if d < min_dist:
                min_dist = d;
                index = j;
        assert(index != -1);
        ids[i] = index;


    count = [0 for i in range(0,K)];

    for i in range(0,n):
        count[ids[i]] += 1;

    for i in range(0,K):
        if count[i] == 0:
            index = int(n * random.uniform(0,1));
            ids[index] = i;
            clusters[i,:] = X[index,:];
            count[i] = 1;

    return ids;

def update(X,ids,K):
    m = len(X[0]);
    n = len(X);
    clusters = [[0 for i in range (0,m)] for j in range(0,K)];
    count = [0 for i in range(0,K)];
    for i in range(0,n):
        for j in range(0,m):
            clusters[ids[i]][j] += X[i][j];
        count[ids[i]] += 1;

    for i in range(0,K):
        for j in range(0,m):
            clusters[i][j] /= count[i];

    return clusters;

def get_clusters(X, num_clusters):
    best_clusters = np.array(initialization(X,num_clusters));
    ids = assignment(X,best_clusters);
    new_clusters = np.array(update(X,ids,num_clusters));

    while np.sum(np.sum(best_clusters - new_clusters, axis=1),axis=0) != 0:
        best_clusters = new_clusters;
        ids = assignment(X,best_clusters);
        new_clusters = np.array(update(X,ids,num_clusters));

    return new_clusters;

def eval_clusters(X,clusters):
    ids = assignment(X,clusters);
    result = 0;
    X = np.array(X);
    clusters = np.array(clusters);
    n = len(X);
    for i in range(0,n):
        result += np.sqrt(np.sum( np.square(clusters[ids[i],:] - X[i,:]) , axis=0));
    return result;


def main():
    num_clusters = 6;
    cluster_size = 30;
    dimension = 2;
    radius = 1;
    num_iter = 5;
    n = num_clusters * cluster_size;

    points = generate_points(num_clusters, dimension, 20 * radius);
    X,Y = generate_data_clusters(cluster_size, num_clusters, dimension, radius, points);


    best_clusters = get_clusters(X, num_clusters);
    min_var = float('inf');
    for i in range(0,1000):
        temp_clusters = get_clusters(X, num_clusters);
        r = eval_clusters(X,temp_clusters);
        if r < min_var:
            min_var = r;
            best_clusters = temp_clusters;
        print(i);

    X = np.array(X);


    sns.scatterplot(x=X[:,0], y=X[:,1]);
    sns.scatterplot(x=best_clusters[:,0],y=best_clusters[:,1], hue = [i for i in range(0,num_clusters)]);
    plt.show();

def compute_distance(x,y):
    result = 0;
    x1 = np.array(x);
    y1 = np.array(y);
    result += np.sum(np.square(x-y),axis=0);
    return np.sqrt(result);


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



if __name__ == '__main__':
    main();
