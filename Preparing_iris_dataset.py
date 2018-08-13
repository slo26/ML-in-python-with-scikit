#
#The iris dataset can be used on classification algorithm for surprivise learning. 
#because it does have dataset of mapping between features as data and response as correct type of iris
#

# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
print(type(iris))
# print the iris dataset
print(iris.data)

#Machine learning terminology
#Each row is an observation (also known as: sample, example, instance, record)
#Each column is a feature (also known as: predictor, attribute, independent variable, input, regressor, covariate)

# print the names of the four features
print(iris.feature_names)
# print integers representing the species of each observation, it is a response
print(iris.target)

# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)

#Each value we are predicting is the response (also known as: target, outcome, label, dependent variable)
#Classification is supervised learning in which the response is categorical
#Regression is supervised learning in which the response is ordered and continuous

#Requirements for working with data in scikit-learn
#Features and response are separate objects
#Features and response should be numeric
#Features and response should be NumPy arrays
#Features and response should have specific shapes
# check the types of the features and response
print(type(iris.data))
print(type(iris.target))
# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)
# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)
# store feature matrix in "X"
X = iris.data
# store response vector in "y"
y = iris.target