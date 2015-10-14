import matplotlib
matplotlib.use('Agg')
import csv as csv
import numpy as np
import pandas as pandas
import sklearn
import matplotlib
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from solutionHelper import *
from plot import plotClassifier


###################################
# learnning some python and Data Sceince on kaggle titanic data set
##################################

##############################
# fix train data + predict on train data with cross validation
##############################
trainData = pandas.read_csv("input/train.csv")
normalizeData(trainData)

#analyze features
analysFeatures(trainData)

# Estimate prediction rate
classifiers = getClassifiers()
names = [ i[0]  for i in classifiers]
print ("solution using : " + "".join(names))
value_column="Survived"

for name,clf,p,w in classifiers: 
        score = sklearn.cross_validation.cross_val_score(clf, trainData[p], trainData[value_column], cv=3)
        print (score.mean())

###########################################
# fix test data + predict on test data
###########################################
testData = pandas.read_csv("input/test.csv") 
normalizeData(testData)

#train + predict
full_predictions = [0]*len(testData)
classifiers = getClassifiers()
for name,clf,p,w in classifiers:
    clf = clf.fit(trainData[p], trainData[value_column])
    plotClassifier(clf,name)
    predictions = clf.predict_proba(testData[p].astype(float))[:,1]
    full_predictions = full_predictions +  predictions * w

totalW = sum([ i[3] for i in classifiers ])
full_predictions = full_predictions / totalW 
full_predictions[full_predictions <= .5] = 0 
full_predictions[full_predictions > .5] = 1
full_predictions = full_predictions.astype(int)

#submit
submission = pandas.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": full_predictions
    })

submission.to_csv("output/prediction.csv",index=False)
