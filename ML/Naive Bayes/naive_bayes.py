from sklearn.datasets.samples_generator import make_blobs;
import matplotlib.pyplot as plt;
from sklearn.svm import SVC; # "Support Vector Classifier"
import numpy as np;
import random;
import pickle;




class naive_bayes:
    #features_type is an array storing a string of each feature's type
    def __init__(self,num_features, num_classes):
        self.n_features = num_features;
        self.num_classes = num_classes;
        self.P = [[{} for i in range(0,self.n_features)] for j in range(0,self.num_classes)];
        self.class_count = [0 for i in range(0,num_classes)];

    # Y-values are in [k] - 1 (namely integer values)
    def train(self,X,Y):
        nc = self.num_classes;
        nf = self.n_features;

        m = len(X);
        for i in range(0,m):
            row = X[i];
            val = Y[i];

            self.class_count[val] += 1;
            for j in range(0,nf):

                t1 = row[j];
                map = self.P[val][j];
                if t1 not in map:
                    map[t1] = 0;
                map[t1] += 1;


    def predict(self,X):
        result = [];
        m = len(X);
        nc = self.num_classes;
        nf = self.n_features;

        for i in range(0,m):
            row = X[i];
            max_prob = -1;
            best_choice = -1;

            for j in range(0,nc):
                temp = (float)(self.class_count[j]) / (float)(m);
                for k in range(0,nf):
                    val = row[k];
                    map = self.P[j][k];
                    if val not in map:
                        temp = 0;
                    else:
                        temp *= (float) (map[val]) / (float) (self.class_count[j]);

                if temp >= max_prob:
                    max_prob = temp;
                    best_choice = j;

            assert(best_choice != -1);
            result.append(best_choice);

        return np.array(result);

def evaluate_accuracy(Y_pred,Y_true):
    n = len(Y_pred);
    count = 0;
    for i in range(0,n):
        if Y_pred[i] != Y_true[i]:
            count += 1;
    return 1 - (float) (count) / (float) (n);

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

def main():
    infile = open("C:\\gmail_api\\read_data.txt",'rb')
    XT = pickle.load(infile)
    infile.close()
    infile = open("C:\\gmail_api\\unread_data.txt",'rb')
    XF = pickle.load(infile)
    infile.close()
    Y = [1 for i in range(0,len(XT))];
    for row in XF:
        XT.append(row);
        Y.append(0);
    X = XT;

    Xtrain, Ytrain, Xtest, Ytest = train_test_split(X,Y,frac=0.8);

    bayes = naive_bayes(num_features=5,num_classes=3);
    bayes.train(Xtrain,Ytrain);


    Ypred = bayes.predict(Xtest);

    acc = evaluate_accuracy(Ypred,Ytest);
    print(acc);

if __name__ == '__main__':
    main();
