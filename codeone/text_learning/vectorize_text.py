
import os
import pickle
import re
import sys

# sys.path.append( "../tools/" )
sys.path.append(os.getcwd() + "\\codeone\\tools")
from parse_out_email_text import parseOutText
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

pwd = os.getcwd()
from_sara  = open(pwd + "\\codeone\\text_learning\\from_sara.txt", "r")
from_chris = open(pwd + "\\codeone\\text_learning\\from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

mail_root = pwd + "\\codeone"

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        if temp_counter < 200:
            path = os.path.join(mail_root, path[:-1])
            print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            words = parseOutText(email)

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            remove_words = ["sara", "shackleton", "chris", "germani"]
            for word in remove_words:
                words.replace(word, '')
            # remove backspace
            words = ' '.join(words.split())
            print 'words:', words
            ### append the text to word_data
            word_data.append(words)
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == 'sara':
                from_data.append(0)
            elif name == 'chris':
                from_data.append(1)

            email.close()
# print 'debug os.path', os.path.join('A','B')
print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

### in Part 4, do TfIdf vectorization here

print 'word_data:', word_data
print 'from_data:', len(from_data)
print 'word_data:', len(word_data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, from_data, test_size=0.1, random_state=42)


### use TfidfTransformer
'''
vectorizer=CountVectorizer()
w_matrix = vectorizer.fit_transform(features_train)
print(w_matrix.toarray())

from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = TfidfTransformer()
features_train_transformed = vectorizer.fit_transform(w_matrix)
'''

### text vectorization--go from strings to lists of numbers
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)

feature_names  = vectorizer.get_feature_names()
print(len(feature_names))

print(features_train_transformed.toarray())
print(features_train_transformed.toarray().shape)
