import pandas as pd
import csv
# import tensorflow as tf
import numpy as np
import sklearn
import imblearn

# from tensorflow 
import keras
from keras import activations
from keras.models import *
from keras.layers import *
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

fileLocation ='C:/Graduate-Studies/Biomedical-Computation/FDI/Docker/food-drug-ssp-all.csv'

#Dataset Preparation
def readDataFromFile(fileLocation):
  interactionData = pd.read_csv(fileLocation,
      names=['Food-id','Drug-id','Tanimoto','Dice','Cosine','Sokal','Tversky','Result'])
  interactionArray = np.array(interactionData)
  # print(interactionArray[0])
  return interactionArray

interactionArray = readDataFromFile(fileLocation)

#this function calculates the F1 score using y_test and y_pred values
def calculateF1Score(y_test, y_pred):
  macro = f1_score(y_test, y_pred, average='macro')
  micro = f1_score(y_test, y_pred, average='micro')
  weighted = f1_score(y_test, y_pred, average='weighted')
  noavg = f1_score(y_test, y_pred, average=None)
  zerodiv = f1_score(y_test, y_pred, zero_division=1)
  return weighted



from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold

def prepareDataset(interactionArray):
  array = np.copy(interactionArray)
  X = array[:,2:7]
  y = interactionArray[:,7]
  y = y.astype(int)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

  return X_train, X_test, y_train, y_test

""" Prepare a dataset with k-folds. Currently taking k = 4.
K-Fold cross validation is a technique where the dataset is divided into k equal parts and train and test data
is split equally, and each part is considered to be a test dataset in turns.
Parameters: an array consisting of all the interactions and their coefficients
Returns: X_train, X_train, y_train, y_test are the array of training and testing datasets. Counter is number of k-splits."""
def prepareDatasetKFold(interactionArray):
  # Copy interactionArray into other array
  array = np.copy(interactionArray)
  # Splitting the array and taking only coeffecients, Tanimoto, Dice, Cosine, Sokal, Tversky indexes
  X = array[:,2:7]
  # Splitting the array and taking the labels as 'TRUE' or 'FALSE' values
  y = interactionArray[:,7]
  # Converting the 'TRUE' or 'FALSE' values into binary labels.
  y = y.astype(int)
  
  # Applying K-Fold cross validation
  kf = KFold(n_splits=4)
  kf.get_n_splits(X)

  #Initializing X_train, X_test, y_train, y_test as empty arrays 
  X_train, X_test, y_train, y_test = [], [], [], []

  # For each train and test index in split of X, y : Append X[train_index], y[train_index] to train dataset and X[test_index], y[test_index] to test dataset
  for train_index, test_index in kf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train.append(X[train_index])
    X_test.append(X[test_index])
    y_train.append(y[train_index])
    y_test.append(y[test_index])
  # Return the test and train dataset with k-fold (currently 4)
  return X_train, X_test, y_train, y_test, 4

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

def kMeansMethod(interactionArray):
  print('K-Means Method \n')
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  model = KMeans(n_clusters=2)
  model.fit(X_train)
  y_pred=model.predict(X_test) 
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('Precision Score : ', metrics.precision_score(y_test, y_pred))
  print('MSE : ', metrics.mean_squared_error(y_test, y_pred))
  print('Mean Squared Log Error : ', metrics.mean_squared_log_error(y_test, y_pred))
  print('Median Abs Error : ', metrics.median_absolute_error(y_test, y_pred))
  print('F1 Score : ', calculateF1Score(y_test, y_pred))
  

from sklearn.mixture import BayesianGaussianMixture
def bgm(interactionArray):
  print('\n Bayesian Gaussian Method \n')
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  model = BayesianGaussianMixture(n_components=2)
  model.fit(X_train)
  y_pred=model.predict(X_test) 
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('Precision Score : ', metrics.precision_score(y_test, y_pred))
  print('MSE : ', metrics.mean_squared_error(y_test, y_pred))
  print('Mean Squared Log Error : ', metrics.mean_squared_log_error(y_test, y_pred))
  print('Median Abs Error : ', metrics.median_absolute_error(y_test, y_pred))
  print('F1 Score : ', calculateF1Score(y_test, y_pred))


from sklearn.mixture import GaussianMixture
def gaussian(interactionArray):
  print('\n Gaussian Method \n')
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  model = GaussianMixture(n_components=2)
  model.fit(X_train)
  y_pred=model.predict(X_test) 
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('Precision Score : ', metrics.precision_score(y_test, y_pred))
  print('\nF1 Score : ', metrics.f1_score(y_test,y_pred))
  calculateF1Score(y_test, y_pred)

# gaussian(interactionArray)
def kNeighbors(interactionArray):
  print('\nkNeighbors\n')
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  model=KNeighborsClassifier(n_neighbors = 1)
  model.fit(X_train,y_train)
  y_pred=model.predict(X_test)
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('Precision Score : ', metrics.precision_score(y_test, y_pred))
  print('\nF1 Score : ', metrics.f1_score(y_test,y_pred))
  calculateF1Score(y_test, y_pred)

def supportVectorMachine(interactionArray):
  print('\nSVM\n')
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  
  X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)
  for i in range(counter):
    print(i)
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    model.fit(X_train[i],y_train[i])
    y_pred=model.predict(X_test[i])
    print('Accuracy Score : ', metrics.accuracy_score(y_test[i],y_pred))
    print('Precision Score : ', metrics.precision_score(y_test[i], y_pred))
    print('F1 Score : ', metrics.f1_score(y_test[i],y_pred))
    calculateF1Score(y_test[i], y_pred)
    print('\n')

from sklearn.neural_network import MLPClassifier

def multiLayerPerceptron(interactionArray):
  print('\nMulti Layer Perceptron\n')
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  model = MLPClassifier(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(X_train, y_train)
  y_pred=model.predict(X_test)
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('Precision Score : ', metrics.precision_score(y_test, y_pred))
  print('\nF1 Score : ', metrics.f1_score(y_test,y_pred))
  calculateF1Score(y_test, y_pred)

from sklearn.linear_model import LinearRegression
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.linear_model import HuberRegressor
def HR(interactionsArray):
  print("Huber Regressor")
  # X_train, X_test, y_train, y_test are arrays having the k-fold datasets. Counter is the number of k that are used
  X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)
  # Initialize precision, accuracy and f1-score as empty arrays
  precision, accuracy, f1_score = [], [], []
  # For each k in k-fold, fit the model, predict, and calculate metrics
  for i in range(counter):
    # Original dataset shape
    print('Original dataset shape %s' % Counter(y_train[i]))

    # Using BorderlineSMOTE for upsampling the dataset
    sm = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train[i], y_train[i])

    # Resampled dataset shape
    print('Resampled dataset shape %s' % Counter(y_train_res))

    # Train the model using fit() with X_train_res and y_train_res
    model = HuberRegressor(max_iter=100000).fit(X_train_res, y_train_res) 
    # Predict a Response
    y_pred = model.predict(X_test[i])
    
    y_pred = np.where(y_pred > 0, 1, 0)
    #Calculate the metrics using functions in sklearn
    precision.append(metrics.precision_score(y_test[i], y_pred))
    accuracy.append(metrics.accuracy_score(y_test[i], y_pred))
    f1_score.append(metrics.f1_score(y_test[i], y_pred))
  
  # Print accuracy, precision and f1-scores by taking the mean of the respective arrays
  print('Accuracy Score : ', np.mean(accuracy))
  print('Precision Score : ', np.mean(precision))
  print('F1 Score : ', np.mean(f1_score))

# HR(interactionArray)
""" LinearRegression fits a linear model with coefficients to minimize the residual sum of squares 
between the observed targets in the dataset, and the targets predicted by the linear approximation.
X_train and y_train are upsampled using Borderline-SMOTE.The model is trained with the X_train and y_train
using fit() and predicts the X_test using predict(). Evaluation metrics are calculated using metrics functions
in sklearn using the y_pred that has been obtained from the predict() with y_test as a parameter.
Parameters: an array consisting of all the interactions and their coefficients """
def LR(interactionArray):
  print('\n Linear Regression \n')

  # X_train, X_test, y_train, y_test are arrays having the k-fold datasets. Counter is the number of k that are used
  X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)
  # Initialize precision, accuracy and f1-score as empty arrays
  precision, accuracy, f1_score = [], [], []

  # For each k in k-fold, fit the model, predict, and calculate metrics
  for i in range(counter):
    # Original dataset shape
    print('Original dataset shape %s' % Counter(y_train[i]))

    # Using BorderlineSMOTE for upsampling the dataset
    sm = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train[i], y_train[i])

    # Resampled dataset shape
    print('Resampled dataset shape %s' % Counter(y_train_res))

    # Train the model using fit() with X_train_res and y_train_res
    model = LinearRegression().fit(X_train_res, y_train_res) 
    # Predict a Response
    y_pred = model.predict(X_test[i])
    
    y_pred = np.where(y_pred > 0, 1, 0)
    #Calculate the metrics using functions in sklearn
    precision.append(metrics.precision_score(y_test[i], y_pred))
    accuracy.append(metrics.accuracy_score(y_test[i], y_pred))
    f1_score.append(metrics.f1_score(y_test[i], y_pred))
  
  # Print accuracy, precision and f1-scores by taking the mean of the respective arrays
  print('Accuracy Score : ', np.mean(accuracy))
  print('Precision Score : ', np.mean(precision))
  print('F1 Score : ', np.mean(f1_score))

# LR(interactionArray)

""" Gaussian Naive Bayes is a probabilistic classification algorithm based on applying Bayes' theorem
with strong independent assumption. X_train and y_train are upsampled using BorderLine-SMOTE. The model is then
trained using the X_train and y_train using fit() and predicts the X_test using predict().Evaluation metrics are 
calculated using metrics functions in sklearn using the y_pred that has been obtained from the predict() with
y_test as a parameter.
Parameters: an array consisting of all the interactions and their coefficients
"""
def NB(interactionArray):
  print('\n Gaussian Naive Bayes \n')

  # X_train, X_test, y_train, y_test are arrays having the k-fold datasets. Counter is the number of k that are used
  X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)

  # Initialize precision, accuracy and f1-score as empty arrays
  precision, accuracy, f1_score = [], [], []
  
  # For each k in k-fold, fit the model, predict, and calculate metrics
  for i in range(counter):
    # Original dataset shape
    print('Original dataset shape %s' % Counter(y_train[i]))
    # Using BorderlineSMOTE for upsampling the dataset
    sm = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train[i], y_train[i])
    # Resampled dataset shape
    print('Resampled dataset shape %s' % Counter(y_train_res))
    # Train the model using fit() with X_train_res and y_train_res and predict a response
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train_res, y_train_res).predict(X_test[i])

    #Calculate the metrics using functions in sklearn
    precision.append(metrics.precision_score(y_test[i], y_pred))
    accuracy.append(metrics.accuracy_score(y_test[i], y_pred))
    f1_score.append(metrics.f1_score(y_test[i], y_pred))
  
  # Print accuracy, precision and f1-scores by taking the mean of the respective arrays
  print('Accuracy Score : ', np.mean(accuracy))
  print('Precision Score : ', np.amax(precision))
  print('F1 Score : ', np.mean(f1_score))
  print('\n')
# NB(interactionArray)

def KNN(interactionArray):
  X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)
  precision, accuracy, f1_score = [], [], []
  for i in range(counter):
    print(i)
    print('Original dataset shape %s' % Counter(y_train[i]))
    sm = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train[i], y_train[i])
    print('Resampled dataset shape %s' % Counter(y_train_res))

    model=KNeighborsClassifier(n_neighbors = 5)
    model.fit(X_train_res,y_train_res)
    y_pred=model.predict(X_test[i])
    precision.append(metrics.precision_score(y_test[i], y_pred))
    accuracy.append(metrics.accuracy_score(y_test[i], y_pred))
    f1_score.append(metrics.f1_score(y_test[i], y_pred))

  print('Accuracy Score : ', np.mean(accuracy))
  print('Precision Score : ', np.amax(precision))
  print('F1 Score : ', np.mean(f1_score))
  print('\n') 
# KNN(interactionArray)

""" Mini-Batch K-means is a version of the standard K-means algorithm in machine learning. It uses small, random, 
fixed-size batches of data to store in memory, and then with each iteration, a random sample of the data is collected 
and used to update the clusters.
Parameters: an array consisting of all the interactions and their coefficients """
from sklearn.cluster import MiniBatchKMeans
def MiniBatchKMeans1(interactionArray):
  #  X_train, X_test, y_train, y_test are variables havind training data and testing data
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)

  # Initializing MiniBatch K-means with 5 clusters, and batch size 6
  kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=6, n_init=3)
  # Train the model using fit() with X_train as a parameter
  kmeans = kmeans.fit(X_train, y_train)

  # Predict the response
  y_pred = kmeans.predict(X_test)
  # Calculate metrics using the metrics functions in sklearn
  print('Accuracy Score : ', metrics.accuracy_score(y_test, y_pred))
  print('Precision Score : ', metrics.precision_score(y_test, y_pred, average='weighted'))
  print('F1 Score : ', metrics.f1_score(y_test, y_pred, average='weighted'))

from sklearn.linear_model import SGDClassifier
def SGD(interactionArray):
  #  X_train, X_test, y_train, y_test are variables havind training data and testing data
  X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)
  precision, accuracy, f1_score = [], [], []

  for i in range(counter):
    print(i)
    print('Original dataset shape %s' % Counter(y_train[i]))
    sm = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train[i], y_train[i])
    print('Resampled dataset shape %s' % Counter(y_train_res))

    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100000)
    clf.fit(X_train_res, y_train_res)
    y_pred=clf.predict(X_test[i])
    precision.append(metrics.precision_score(y_test[i], y_pred))
    accuracy.append(metrics.accuracy_score(y_test[i], y_pred))
    f1_score.append(metrics.f1_score(y_test[i], y_pred))

  print('Accuracy Score : ', np.mean(accuracy))
  print('Precision Score : ', np.amax(precision))
  print('F1 Score : ', np.mean(f1_score))
  print('\n') 


# SGD(interactionArray)
from sklearn.neural_network import MLPClassifier

def multiLayerPerceptron(interactionArray):
  print('\nMulti Layer Perceptron\n')
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  print('Original dataset shape %s' % Counter(y_train))
  sm = BorderlineSMOTE(random_state=42)
  X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
  print('Resampled dataset shape %s' % Counter(y_train_res))
  model = MLPClassifier(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(X_train_res, y_train_res)
  y_pred=model.predict(X_test)
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('Precision Score : ', metrics.precision_score(y_test, y_pred))
  print('\nF1 Score : ', metrics.f1_score(y_test,y_pred))
  calculateF1Score(y_test, y_pred)
# import calendar
# import time
# import datetime
# def multiLayerPerceptron(interactionArray):
#   print('\nMulti Layer Perceptron\n')
#   X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)
#   precision, accuracy, f1_score = [], [], []

#   for i in range(counter):
#     print(i)
#     # print('Original dataset shape %s' % Counter(y_train[i]))
#     # sm = BorderlineSMOTE(random_state=42)
#     # X_train_res, y_train_res = sm.fit_resample(X_train[i], y_train[i])
#     # print('Resampled dataset shape %s' % Counter(y_train_res))
#     time_start = datetime.datetime.now()
#     print(time_start)
#     model = MLPClassifier(hidden_layer_sizes=(8,8,8),activation="relu" ,random_state=1, max_iter=500).fit(X_train[i], y_train[i])
#     y_pred=model.predict(X_test[i])
#     time_end = datetime.datetime.now()
#     print(time_end)
#     print((time_end - time_start).total_seconds() / 60)
#     precision.append(metrics.precision_score(y_test[i], y_pred))
#     accuracy.append(metrics.accuracy_score(y_test[i], y_pred))
#     f1_score.append(metrics.f1_score(y_test[i], y_pred))

#   print('Accuracy Score : ', np.mean(accuracy))
#   print('Precision Score : ', np.mean(precision))
#   print('F1 Score : ', np.mean(f1_score))
#   print('\n') 
multiLayerPerceptron(interactionArray)


# MiniBatchKMeans1(interactionArray)
# AP(interactionArray)
# LR(interactionArray)
# kMeansMethod(interactionArray)
# bgm(interactionArray)


# dbScan(interactionArray)

# gaussianNaivesBayes(interactionArray)
# randomForest(interactionArray)
# kNeighbors(interactionArray)
# supportVectorMachine(interactionArray)
# multiLayerPerceptron(interactionArray)
