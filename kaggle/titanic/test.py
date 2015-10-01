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
print trainData['SibSp']

