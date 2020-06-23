from __future__ import print_function

training_data = [
    [84,320,"Basketball","B"],
    [82,260,"Basketball","B"],
    [79,220,"Basketball","B"],
    [80,250,"Basketball","B"],
    [83,220,"Basketball","B"],
    [78,230,"Basketball","B"],
    [77,230,"Basketball","B"],
    [73,200,"Basketball","B"],
    [73,220,"Football","B"],
    [76,220,"Football","W"],
    [78,265,"Football","W"],
    [67,220,"Football","B"],
    [68,230,"Football","B"],
    [71,179,"Tennis","W"],
    [77,195,"Tennis","W"],
    [75,185,"Tennis","W"],
    [76,185,"Tennis","B"]
];
header = ["Height", "Weight", "Sport"];

#unique values of rows with column
def unique_vals(rows,col):
    return set([row[col] for row in rows]);

#counts is a dictionary with label -> occurrences of label
def class_counts(rows):
    counts = {};
    for row in rows:
        label = row[-1];
        if label not in counts:
            counts[label] = 0;
        counts[label] += 1;
    return counts;

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float);

class Question:
    def __init__(self, column, value):
        self.column = column;
        self.value = value;

    def match(self, example):
        val = example[self.column];
        if is_numeric(val):
            return val >= self.value;
        else:
            return val == self.value;

    def __repr__(self):
        condition = "==";
        if is_numeric(self.value):
            condition = ">=";
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value));

#partition remaining into true and false based on question
def partition(rows,question):
    true_rows, false_rows = [], [];
    for row in rows:
        if question.match(row):
            true_rows.append(row);
        else:
            false_rows.append(row);
    return true_rows, false_rows;

#Calculate Gini_impurity
def gini(rows):
    counts = class_counts(rows);
    result = 1;
    n = len(rows);
    for label in counts:
        result -= (counts[label] / float(n)) ** 2;
    return result;

#checks the amount of information gained by question
def info_gain(left, right , current_uncertainty):
    p = float(len(left) / (len(left) + len(right)));
    return current_uncertainty - p * gini(left) - (1-p) * gini(right);

def find_best_split(rows):
    best_gain = 0;
    best_question = None;
    current_uncertainty = gini(rows);
    n_features = len(rows[0]) - 1;
    for col in range(n_features):
        values = set([row[col] for row in rows])

        for val in values:
            question = Question(col,val);
            true_rows , false_rows = partition(rows,question);
            if true_rows == [] or false_rows == []:
                continue;
            gain = info_gain(true_rows , false_rows , current_uncertainty);

            if gain >= best_gain:
                best_gain = gain;
                best_question = question;

    return best_gain, best_question;

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows);

class Decision_Node:
    def __init__(self, question , true_branch , false_branch):
        self.question = question;
        self.true_branch = true_branch;
        self.false_branch = false_branch;

def build_tree(rows):
    gain, question = find_best_split(rows);

    if gain == 0:
        return Leaf(rows);

    true_rows , false_rows = partition(rows,question);

    true_branch = build_tree(true_rows);
    false_branch = build_tree(false_rows);

    return Decision_Node(question, true_branch , false_branch);

def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
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


def classify(row,node):

    if isinstance(node, Leaf):
        return node.predictions;

    if node.question.match(row):
        return classify(row, node.true_branch);
    else:
        return classify(row, node.false_branch);

"""
    Overview of the Construction of a Decision Tree:
    1.) Start with remaining rows = all rows of training data
    2.) Find the best question to use to partition the remaining rows
    - Best is defined as most information gained, minimize weighted gini impurity of true and false sub branch
    - iterated through all features, all possible values
    3.) Partition remaining rows according to the true/false branches and recursively compute rest of tree
    - Binary Recursive Partitioning
    4.) Evaluate on testing data
"""

if __name__ == '__main__':
    t1 = training_data;
    n = len(t1);

    for i in range(0,n):
        temp = t1[i][2];
        t1[i][2] = t1[i][3];
        t1[i][3] = temp;
    print(t1);
    tree = build_tree(t1);
    print_tree(tree);

    testing_data = [
        [71,170,"W","Tennis"]
    ]

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %(row[-1], print_leaf(classify(row, tree))));
