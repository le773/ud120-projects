"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
import os
sys.path.append(os.getcwd() + "\\codeone\\tools")
from email_preprocess import preprocess

print sys.path
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

from sklearn import svm
t0 = time()
linear_svc = svm.SVC(kernel='linear')
print "training time:", round(time()-t0, 3), "s"

linear_svc.fit(features_train, labels_train)

t1 = time()
labels_predict = linear_svc.predict(features_test)
print "training time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print "accuracy_score:", accuracy_score(labels_predict, labels_test)
print "precision_score:", precision_score(labels_predict, labels_test)
print "recall_score:", recall_score(labels_predict, labels_test)
print "f1_score:", f1_score(labels_predict, labels_test)

#########################################################

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

labels_test = true_labels
pred = predictions
true_positive=0.0
true_negative=0.0
false_positive=0.0
false_negative=0.0
precision=0.0
recall=0.0

for i in range(len(labels_test)):

  if (labels_test[i] == 1.0) and (labels_test[i] == pred[i]):
     true_positive+=1.0
  if (labels_test[i] == 0.0) and (labels_test[i] == pred[i]):
     true_negative+=1.0
  if (labels_test[i] == 0.0) and (labels_test[i] != pred[i]):
     false_positive+=1.0
  if (labels_test[i] == 1.0) and (labels_test[i] != pred[i]):
     false_negative+=1.0

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print "precision:", precision
print "recall:", recall
