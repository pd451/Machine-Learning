import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pickle
import random

def compute_cost_example():
    a = np.log([[0.04, 0.13, 0.96, 0.12],    # correct prediction
            [0.01, 0.93, 0.06, 0.07]])   # incorrect prediction
    b = np.array([[ 0,    0,    1,    0],
                  [ 1,    0,    0,    0]])   # labels

    print(-a *b);
    r_sum = np.sum(-a * b, axis=1)
    r_mean = np.mean(r_sum)

    print("With Numpy");

    print(f' sum = {r_sum}')
    print(f'mean = {r_mean:.5f}')

    with tf.Session() as sesh:
        tf_sum = sesh.run(-tf.reduce_sum(a * b, axis=1))
        tf_mean = sesh.run(tf.reduce_mean(tf_sum))

    print("With tensorflow");

    print(f' sum = {tf_sum}')
    print(f'mean = {tf_mean:.5f}')

def visual_train_data(x_train, y_train):
    fig, axes = plt.subplots(1, 4, figsize=(7, 3))
    for img, label, ax in zip(x_train[:4], y_train[:4], axes):
        ax.set_title(label)
        ax.imshow(img)
        ax.axis('off')
    plt.show()

def plot_log_function():
    x = np.linspace(1/100, 1, 100)
    fig, ax = plt.subplots(1, figsize=(4.7, 3))
    ax.plot(x, np.log(x), label='$\ln(x)$')
    ax.legend()
    plt.show()

def plot_softmax():
    x = np.arange(30)
    fig, ax = plt.subplots(1, figsize=(4.7, 3))
    ax.plot(x, softmax(x), label='Softmax')
    ax.legend()
    plt.show()

def plot_results(x_train, pred):
    fig, axes = plt.subplots(1, 10, figsize=(8, 4))
    for img, ax in zip(x_train[:10], axes):
        guess = np.argmax(sesh.run(pred, feed_dict={X: [img]}))
        ax.set_title(guess)
        ax.imshow(img.reshape((28, 28)))
        ax.axis('off')

    plt.show();

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

def load_data():
    infile = open("C:\\gmail_api\\read_data.txt",'rb')
    XT = pickle.load(infile)
    infile.close()
    infile = open("C:\\gmail_api\\unread_data.txt",'rb')
    XF = pickle.load(infile)
    infile.close()
    Y = [];
    X = [];
    for row in XF:
        X.append(row[1:]);
        Y.append(0);
    for row in XT:
        X.append(row[1:]);
        Y.append(1);

    Xtrain, Ytrain, Xtest, Ytest = train_test_split(X,Y,frac=0.8);

    return np.array(Xtrain), np.array(Ytrain), np.array(Xtest), np.array(Ytest);

def main():
    # Retrieve Train and Test Data
    x_train, y_train, x_test, y_test = load_data()
    num_categories = 2;
    num_examples = 1000;
    num_features = len(x_train[0]);
    print(x_train);
    assert(num_features == 4);

    with tf.Session() as sesh:
        y_train = sesh.run(tf.one_hot(y_train, num_categories))
        y_test = sesh.run(tf.one_hot(y_test, num_categories))

    x_train = np.reshape(x_train,(int(num_examples * 0.8), num_features));
    x_test = np.reshape(x_test,(int(num_examples * 0.2), num_features));

    #Declare Hyperparamters
    learning_rate = 0.01
    epochs = 20
    batch_size = 100
    batches = int(x_train.shape[0] / batch_size)

    X = tf.placeholder(tf.float32, [None, num_features])
    Y = tf.placeholder(tf.float32, [None, num_categories])

    W = tf.Variable(0.1 * np.random.randn(num_features, num_categories).astype(np.float32))
    B = tf.Variable(0.1 * np.random.randn(num_categories).astype(np.float32))

    pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for i in range(batches):
                offset = i * epoch
                x = x_train[offset: offset + batch_size]
                y = y_train[offset: offset + batch_size]
                sesh.run(optimizer, feed_dict={X: x, Y:y})
                c = sesh.run(cost, feed_dict={X:x, Y:y})

            if not epoch % 2:
                print(f'epoch:{epoch:2d} cost={c:.4f}')

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acc = accuracy.eval({X: x_test, Y: y_test})
        print(f'Accuracy: {acc * 100:.2f}%')




def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


if __name__ == '__main__':
    main();
