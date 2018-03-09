
"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
import os
sys.path.append(os.getcwd() + "\\codeone\\tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print("features_train:",len(features_train))
print("features_test:",len(features_test))
print("labels_train:",len(labels_train))
print("labels_test:",len(labels_test))
#########################################################
### your code goes here ###

from sklearn import tree
clf = tree.DecisionTreeClassifier()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
#########################################################
t1 = time()
labels_predict = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print "accuracy_score:", accuracy_score(labels_predict, labels_test)
print "precision_score:", precision_score(labels_predict, labels_test)
print "recall_score:", recall_score(labels_predict, labels_test)
print "f1_score:", f1_score(labels_predict, labels_test)
