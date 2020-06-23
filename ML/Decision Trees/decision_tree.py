from __future__ import print_function
import numpy as np;
import random;
#training data is T = [X | y], last column is y-value

training_data = [
    [84,320,"Basketball","B",10],
    [82,260,"Basketball","B",20],
    [79,220,"Basketball","B",30],
    [80,250,"Basketball","B",40],
    [83,220,"Basketball","B",50],
    [78,230,"Basketball","B",60],
    [77,230,"Basketball","B",61],
    [73,200,"Basketball","B",63],
    [73,220,"Football","B",240],
    [76,220,"Football","W",243],
    [78,265,"Football","W",260],
    [67,220,"Football","B",240],
    [68,230,"Football","B",230],
    [71,179,"Tennis","W",1000],
    [77,195,"Tennis","W",1020],
    [75,185,"Tennis","W",1204],
    [76,185,"Tennis","B",1203]
];

training_data1 = [

    ['Green', 3, 5, 'Apple'],
    ['Yellow', 3, 5,'Apple'],
    ['Red', 1, 4,'Grape'],
    ['Red', 1, 5, 'Grape'],
    ['Yellow', 3, 2, 'Lemon'],
];
header = ["Height", "Weight", "Salary","Race"];

"""
Random Forest Implementation > Decision Tree Implementation
2 Methods : Regression and classification
Construction of Decision Trees uses recursive binary splitting until all rows in a node have the same output value or
is under a limit (hyper param, for larger training sets) 
"""

#Class Declaration

class Random_Forest:
    def __init__(self, rows, num_trees, examples_frac, features_frac, regression, limit): #regression 1/0 = Y/N, rows = training_data, examples_frac,features_frac \in (0,1)
        n_features = len(rows[0]) - 1;
        n_examples = len(rows);
        features_size = max(int(n_features * features_frac),1);
        examples_size = max(int(n_examples * examples_frac),1);
        trees = [[] for i in range(0,num_trees)];
        self.regression = regression;
        self.limit = limit;
        feature_sets = [];
        for i in range(0,num_trees):
            f_rset = random_subset(n_features, features_size);
            e_rset = random_subset(n_examples, examples_size);
            f_rset.append(n_features);
            feature_sets.append(f_rset);
            temp = [rows[col] for col in e_rset];
            temp_rows = [];
            for row in temp:
                temp_rows.append([row[col] for col in f_rset]);

            if regression == 1:
                trees[i] = build_tree_regression(temp_rows,limit);
            else:
                trees[i] = build_tree(temp_rows);
        self.trees = trees;
        self.num_trees = num_trees;
        self.feature_sets = feature_sets;

    def predict(self,row,regression):
        if regression == 1:
            n = self.num_trees;
            result = 0;
            for i in range(0,n):
                temp = classify_regression([row[col] for col in self.feature_sets[i]],self.trees[i]);
                result += temp.predictions;
            result /= n;
            return result;
        else:
            t1 = {};
            n = self.num_trees;

            for i in range(0,n):
                temp = classify([row[col] for col in self.feature_sets[i]],self.trees[i]);
                occ = 0;
                for key in temp:
                    occ += temp[key];
                temp2 = {};
                for key in temp:
                    temp2[key] = (float) (temp[key]) / (float) (occ);

                for key in temp2:
                    if key not in t1:
                        t1[key] = temp2[key];
                    else:
                        t1[key] += temp2[key];

            max = -1;
            result = [];

            for key in t1.keys():
                if t1[key] >= max:
                    max = t1[key];
                    result = key;
            return result;


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows);

class Reg_Leaf:
    def __init__(self,rows):
        average = 0;
        n = (float) (len(rows));
        for row in rows:
            average += row[-1];
        stdev = 0;
        average = (float) (average) / n;
        for row in rows:
            stdev += (row[-1] - average) ** 2;
        stdev /= n;
        stdev = np.sqrt(stdev);
        self.predictions = average;
        self.stdev = stdev;
        self.n = n;


class Decision_Node:
    def __init__(self,question,true_branch,false_branch):
        self.question = question;
        self.true_branch = true_branch;
        self.false_branch = false_branch;

class Question:
    def __init__(self, column, value):
        self.column = column;
        self.value = value;

    def match(self, example):
        if is_numeric(example[self.column]):
            return self.value >= example[self.column];
        else:
            return self.value == example[self.column];

    def __repr__(self):
        condition = "==";
        if is_numeric(self.value):
            condition = ">=";
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value));

# Print Tree/Leaves

def print_tree(node, spacing=""):
    if isinstance(node, Leaf) or isinstance(node, Reg_Leaf):
        print (spacing + "Predict", node.predictions);
        return;
    print (spacing + str(node.question));
    print (spacing + '--> True:');
    print_tree(node.true_branch, spacing + "  ");
    print (spacing + '--> False:');
    print_tree(node.false_branch, spacing + "  ");




def print_leaf(counts):
    total = sum(counts.values()) * 1.0;
    probs = {};
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%";
    return probs;

#Utility Function

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


def class_counts(rows):
    counts = {};
    for row in rows:
        label = row[-1];
        if label not in counts.keys():
            counts[label] = 1;
        else:
            counts[label] += 1;
    return counts;

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float);

def partition(rows, question):
    true_rows = [];
    false_rows = [];
    for row in rows:
        if question.match(row):
            true_rows.append(row);
        else:
            false_rows.append(row);
    return true_rows, false_rows;


#Compute Relevant Parameters

def gini(rows):
    counts = class_counts(rows);
    n = len(rows);
    result = 1;
    for label in counts:
        result -= ((float) (counts[label]) / (float) (len(rows))) ** 2;
    return result;

def info_gain(left, right, current_uncertainty):
    p = (float) (len(left)) / (float) (len(left) + len(right));
    return current_uncertainty - p * gini(left) - (1-p) * gini(right);

def square_residuals(left, right):
    left_average = 0;
    left_count = len(left);
    right_average = 0;
    right_count = len(right);

    result = 0;

    for row in left:
        left_average += row[-1];
    for row in right:
        right_average += row[-1];

    left_average /= left_count;
    right_average /= right_count;

    for row in left:
        result += (row[-1] - left_average) ** 2;
    for row in right:
        result += (row[-1] - right_average) ** 2;

    return result;

#Build Tree Regression

def find_split_regression(rows):
    best_loss = float('inf');
    best_question = None;
    n_features = len(rows[0]) - 1;
    for i in range(0,n_features):
        values = set([row[i] for row in rows]);
        for val in values:
            question = Question(i,val);
            true_rows, false_rows = partition(rows,question);
            if true_rows == [] or false_rows == []:
                continue;
            loss = square_residuals(false_rows, true_rows);
            if loss <= best_loss:
                best_loss = loss;
                best_question = question;
    return best_loss, best_question;

def build_tree_regression(rows,limit):
    best_loss, best_question = find_split_regression(rows);
    if best_loss == float('inf') or len(rows) <= limit:
        return Reg_Leaf(rows);
    true_rows, false_rows = partition(rows, best_question);
    true_branch = build_tree_regression(true_rows,limit);
    false_branch = build_tree_regression(false_rows,limit);
    return Decision_Node(best_question, true_branch, false_branch);

def classify_regression(row,node):
    if isinstance(node, Reg_Leaf):
        return node;
    if node.question.match(row):
        return classify_regression(row, node.true_branch);
    else:
        return classify_regression(row, node.false_branch);

#Build Tree Classification

def find_best_partition(rows):
    best_gain = 0;
    best_question = None;
    n_features = len(rows[0]) - 1;
    current_uncertainty = gini(rows);
    for i in range(0,n_features):
        values = set([row[i] for row in rows]);
        for val in values:
            question = Question(i,val);
            true_rows, false_rows = partition(rows,question);
            if true_rows == [] or false_rows == []:
                continue;
            gain = info_gain(false_rows, true_rows, current_uncertainty);
            if gain >= best_gain:
                best_gain = gain;
                best_question = question;
    return best_gain, best_question;

def build_tree(rows):
    best_gain, best_question = find_best_partition(rows);
    if best_gain == 0:
        return Leaf(rows);
    true_rows, false_rows = partition(rows, best_question);
    true_branch = build_tree(true_rows);
    false_branch = build_tree(false_rows);
    return Decision_Node(best_question, true_branch, false_branch);

def classify(row,node):
    if isinstance(node, Leaf):
        return node.predictions;

    if node.question.match(row):
        return classify(row, node.true_branch);
    else:
        return classify(row, node.false_branch);

#Evaluation

def evaluate(tree,node,regression): #regression = 1 => regression, regression = 0 => classification
    if regression == 1:
        leaf = classify_regression(row,node);
    else:
        leaf = classify(row,node);

def main():
    #def __init__(self, rows, num_trees, examples_frac, features_frac, regression, limit)
    limit = 2;
    t1 = training_data1;
    num_trees = 5;
    examples_frac = 0.5;
    features_frac = 0.5;
    regression = 0;

    testing_data = [["Red", 1, 5, "Apple"]];

    forest = Random_Forest(t1,num_trees,examples_frac,features_frac,regression,limit);

    result = forest.predict(testing_data[0],regression);

    print(result);

def test_data(testing_data,tree):
    for row in testing_data:
        leaf = classify_regression(row,tree);
        print("Actual = " + str(row[-1]) + ", Prediction = " + str(leaf.predictions));
        print("Standard Deviation = " + str(leaf.stdev) + ", Number of Sample in Class = " + str(leaf.n));
        print();

def print_tree2(node,spacing=""):
    if isinstance(node, Reg_Leaf):
        print (spacing + "Predict", node.predictions);
        return;
    print (spacing + str(node.question));
    print (spacing + '--> True:');
    print_tree2(node.true_branch, spacing + "  ");
    print (spacing + '--> False:');
    print_tree2(node.false_branch, spacing + "  ");

def swap_cols(A,a,b):
    n = len(A);
    for i in range(0,n):
        temp = A[i][a];
        A[i][a] = A[i][b];
        A[i][b] = temp;
    return A;


if __name__ == '__main__':
    main();
