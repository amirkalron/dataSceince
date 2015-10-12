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


###################################
# learnning some python and Data Sceince on kaggle titanic data set
##################################

##############################
# fix train data + predict on train data with cross validation
##############################
trainData = pandas.read_csv("train.csv")
# d = trainData.groupby("Survived").count()['Sex']/len(trainData.index)
# print d

normalizeData(trainData)
g1 = trainData.groupby(["FareCategory","Pclass","Sex"],sort=True).agg({'Survived': 'count'})
g2 = trainData.groupby(["FareCategory","Pclass","Sex"],sort=True).agg('size')
print groupsSize[3]


# 
train1=trainData[:692]
# train1.to_csv("test/train1.csv",index=False)
# test1=trainData[693:]
# test1.to_csv("test/test1.csv",index=False)


