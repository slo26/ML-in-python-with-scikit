#Classification task: Predicting the species of an unknown iris
#Used three classification models: KNN (K=1), KNN (K=5), logistic regression
#Need a way to choose between the models

#Solution: Model evaluation procedures

#Evaluation procedure #1: Train and test on the entire dataset
#1.Train the model on the entire dataset.
#2.Test the model on the same dataset, and evaluate how well we did by comparing the predicted response values with the true response values.
# read in the iris data
from sklearn.datasets import load_iris
iris = load_iris()
# create X (features) and y (response)
X = iris.data
y = iris.target

#Logistic regression
# import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X, y)
# predict the response values for the observations in X
y_pred = logreg.predict(X)
print(y_pred)
print(len(y_pred))
# compute classification accuracy for the logistic regression model# compu 
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))
#KNN (K=5)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))
#KNN (K=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))

#Evaluation procedure #2: Train/test split
#Split the dataset into two pieces: a training set and a testing set.
#Train the model on the training set.
#Test the model on the testing set, and evaluate how well we did.
# print the shapes of X and y
print(X.shape)
print(y.shape)
# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
# print the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)
# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)
# STEP 2: train the model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# STEP 3: make predictions on the testing set# STEP 3 
y_pred = logreg.predict(X_test)
# compare actual response values (y_test) with predicted response values (y_pred)
print(metrics.accuracy_score(y_test, y_pred))
print("Repeat for KNN with K=5:")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print("Repeat for KNN with K=1:")
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print("try K=1 through K=25 and record testing accuracy")
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print("# import Matplotlib (scientific plotting library)")
import matplotlib.pyplot as plt
# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

#Making predictions on out-of-sample data
# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=11)
# train the model with X and y (not X_train and y_train)
knn.fit(X, y)
# make a prediction for an out-of-sample observation
print(knn.predict([[3, 5, 4, 2]]))