""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


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
