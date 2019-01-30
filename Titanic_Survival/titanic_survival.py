# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from sklearn import preprocessing, model_selection, neighbors, svm

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier



df = pd.read_csv('train.csv')
# many algorithms take in -99999 as NA values and skip it
# df.replace('?', -99999, inplace=True)
# unnecessary data which affects accuracy if included
# id of data is irrelevant 

df['FamilyMembers'] = df['SibSp'] + df['Parch']
df = df[['Pclass','Sex','Age','FamilyMembers','Survived']]
df.fillna( -99999, inplace = True )
# print(df.head(20))
# features are everything except the class that it belongs to
X = np.array(df.drop(['Survived'],1))
# result is the class that a data point belongs to
y = np.array(df['Survived'])

train_df = df
test_df = pd.read_csv('test.csv')
test_df['FamilyMembers'] = test_df['SibSp'] + test_df['Parch']

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df[['Pclass','Sex','Age','FamilyMembers']]

X_test.fillna( -99999, inplace = True )
	
# # Logistic Regression
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print('LR => ',acc_log)
# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
# print(coeff_df.sort_values(by='Correlation', ascending=False))

# # Support Vector Machines
# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# print('SVM -> ',acc_svc)

# # KNN
# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)
# acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# print('KNN -> ', acc_knn)

# # Gaussian Naive Bayes
# gaussian = GaussianNB()
# gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
# acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
# print('Gaussian Naive Bayer -> ',acc_gaussian)

# # Perceptron
# perceptron = Perceptron()
# perceptron.fit(X_train, Y_train)
# Y_pred = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# print('Perceptron -> ',acc_perceptron)

# # Linear SVC
# linear_svc = LinearSVC()
# linear_svc.fit(X_train, Y_train)
# Y_pred = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# print('Linear SVC -> ',acc_linear_svc)

# # Stochastic Gradient Descent
# sgd = SGDClassifier()
# sgd.fit(X_train, Y_train)
# Y_pred = sgd.predict(X_test)
# acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# print('Stochastic Gradient Descent -> ',acc_sgd)

# Decision Tress
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print('Decision Trees -> ',acc_decision_tree)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Random Forest -> ',acc_random_forest)

# models = pd.DataFrame({
    # 'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              # 'Random Forest', 'Naive Bayes', 'Perceptron', 
              # 'Stochastic Gradient Decent', 'Linear SVC', 
              # 'Decision Tree'],
    # 'Score': [acc_svc, acc_knn, acc_log, 
              # acc_random_forest, acc_gaussian, acc_perceptron, 
              # acc_sgd, acc_linear_svc, acc_decision_tree]})
# print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
print(submission.shape)
# submission.to_csv('submission_titanic.csv', index=False)