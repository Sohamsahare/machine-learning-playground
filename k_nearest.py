import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# many algorithms take in -99999 as NA values and skip it
df.replace('?', -99999, inplace=True)
# unnecessary data which affects accuracy if included
# id of data is irrelevant 
df.drop(['id'], 1, inplace=True)
 
# features are everything except the class that it belongs to
X = np.array(df.drop(['class'],1))
# result is the class that a data point belongs to
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# KNN
clf = neighbors.KNeighborsClassifier()
# SVM Support Vector Classifier
# clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# # testing on random data
# example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,3,7,9,1,7,2,1]])
# # reshaping 1d array to 2d array because sci-kit learn demands so
# example_measures = example_measures.reshape(len(example_measures),-1)
# prediction = clf.predict(example_measures)

# print(prediction)