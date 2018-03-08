

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.naive_bayes import GaussianNB


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t2 = time()
features_predict = clf.predict(features_test)
print "predicting time:", round(time()-t2, 3), "s"

# acc = clf_pf.predict(features_predict, labels_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print "accuracy_score:", accuracy_score(features_predict, labels_test)
print "precision_score:", precision_score(features_predict, labels_test)
print "recall_score:", recall_score(features_predict, labels_test)
print "f1_score:", f1_score(features_predict, labels_test)

#########################################################
