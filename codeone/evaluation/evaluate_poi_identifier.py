"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
import os
from time import time
sys.path.append(os.getcwd() + "\\codeone\\tools")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open(os.getcwd() + "\\codeone" + "\\final_project\\final_project_dataset.pkl", "r"))

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

print 'labels:', len(labels)
print 'features:', len(features)

### your code goes here

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"


t1 = time()
labels_predict = clf.predict(features_test)
print "training time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print "accuracy_score:", accuracy_score(labels_predict, labels_test)
print "precision_score:", precision_score(labels_predict, labels_test)
print "recall_score:", recall_score(labels_predict, labels_test)
print "f1_score:", f1_score(labels_predict, labels_test)
