#150 observations
#4 features (sepal length, sepal width, petal length, petal width)
#Response variable is the iris species
#Classification problem since response is categorical

#K-nearest neighbors (KNN) classification
#Pick a value for K.
#Search for the K observations in the training data that are "nearest" to the measurements of the unknown iris.
#Use the most popular response value from the K nearest neighbors as the predicted response value for the unknown iris.

# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
# store feature matrix in "X"
X = iris.data
# store response vector in "y"
y = iris.target

# print the shapes of X and y
print(X.shape)
print(y.shape)

#Step 1: Import the class you plan to use
from sklearn.neighbors import KNeighborsClassifier
#Step 2: "Instantiate" the "estimator"
#"Estimator" is scikit-learn's term for model
#"Instantiate" means "make an instance of"
knn = KNeighborsClassifier(n_neighbors=1)
#Name of the object does not matter
#Can specify tuning parameters (aka "hyperparameters") during this step
#All parameters not specified are set to their defaults
print(knn)
#Step 3: Fit the model with data (aka "model training")
#Model is learning the relationship between X and y
#Occurs in-place
knn.fit(X, y)
#Step 4: Predict the response for a new observation
#New observations are called "out-of-sample" data
#Uses the information it learned during the model training process
print(knn.predict([[3, 5, 4, 2]]))
#Returns a NumPy array
#Can predict for multiple observations at once
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X_new))
print("Using a different value for K=5")
# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)
# fit the model with data
knn.fit(X, y)
# predict the response for new observations
print(knn.predict(X_new))
print("Using a different classification model=LogisticRegression")
# import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X, y)
# predict the response for new observations
print(logreg.predict(X_new))