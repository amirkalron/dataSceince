import csv as csv
import numpy as np
import pandas as pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier
from solutionHelper import normalizeData
from solutionHelper import getPredictors

#init predicators
predictors = getPredictors()
algType = "Random forest"
print "solution using : " + algType

##############################
# fix train data + predict on train data with cross validation
##############################
titanic = pandas.read_csv("train.csv")
normalizeData(titanic)

#just estimate prediction rate
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = sklearn.cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print scores.mean()

###########################################
# fix test data + predict on test data
###########################################
titanic_test = pandas.read_csv("test.csv") 
normalizeData(titanic_test)

#train + predict
#alg = LogisticRegression(random_state=1)
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
alg.fit(titanic[predictors], titanic["Survived"])
predictions = alg.predict(titanic_test[predictors])

#submit
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("prediction.csv",index=False)
