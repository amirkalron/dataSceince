
import csv as csv
import numpy as np
import pandas as pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from solutionHelper import normalizeData
from solutionHelper import getPredictors
from solutionHelper import getAlgs

###################################
# learnning some python and Data Sceince on kaggle titanic data set
##################################

##############################
# fix train data + predict on train data with cross validation
##############################
trainData = pandas.read_csv("train.csv")
normalizeData(trainData)

# Estimate prediction rate
algs = getAlgs()
names = [ i[0]  for i in algs]
print ("solution using : " + "".join(names))
value_column="Survived"

for name,alg,p,w in algs: 
        score = sklearn.cross_validation.cross_val_score(alg, trainData[p], trainData[value_column], cv=3)
        print (score.mean())

###########################################
# fix test data + predict on test data
###########################################
testData = pandas.read_csv("test.csv") 
normalizeData(testData)

#train + predict
full_predictions = [0]*len(testData)
algs = getAlgs()
for name,alg,p,w in algs:
	alg.fit(trainData[p], trainData[value_column])
	predictions = alg.predict_proba(testData[p].astype(float))[:,1]
	full_predictions =full_predictions +  predictions * w

totalW = sum([ i[3] for i in algs ])
full_predictions = full_predictions / totalW 
full_predictions[full_predictions <= .5] = 0 
full_predictions[full_predictions > .5] = 1
full_predictions = full_predictions.astype(int)

#submit
submission = pandas.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": full_predictions
    })

submission.to_csv("prediction.csv",index=False)
