import re
import operator
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib.backends.backend_pdf import PdfPages

###########################################
# Common Data Sceince functions ##########
##########################################

def normalizeData(data):
	data["Age"]=data["Age"].fillna(data["Age"].median())
	data["Fare"]=data["Fare"].fillna(data["Fare"].median())
	data.loc[data["Sex"] == "male", "Sex"] = 0
	data.loc[data["Sex"] == "female", "Sex"] = 1
	data["Embarked"] = data["Embarked"].fillna("S")
	data.loc[data["Embarked"] == "S", "Embarked"] = 0
	data.loc[data["Embarked"] == "C", "Embarked"] = 1
	data.loc[data["Embarked"] == "Q", "Embarked"] = 2
	data["FamilySize"] = data["SibSp"] + data["Parch"]	
	data["NameLength"] = data["Name"].apply(lambda x: len(x))
	data["Title"] = data.apply(lambda x: getTitle(x,False),axis=1)
	data["ShortTitle"] = data.apply(lambda x: getTitle(x,True),axis=1)	
	
 	fId = data.apply(getFamilyId,axis=1)
        fId[data["FamilySize"] < 3]=-1
        data["FamilyId"]=fId

allFeatures = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize", "Title","ShortTitle","FamilyId"]

def getAlgs():
	return  [["ranodom forest",RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2),
			["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize", "Title","FamilyId"],10],
	  	 ["gradient boosting",GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
			["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize", "ShortTitle","FamilyId"],4],
	  	 ["logistic regression",LogisticRegression(random_state=1),
			["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize", "ShortTitle","FamilyId"],1]]
def analysFeatures(data):
	predictors = allFeatures;
	selector = SelectKBest(f_classif, k=5)
	selector.fit(data[predictors], data["Survived"])
	scores = -np.log10(selector.pvalues_)
	with PdfPages('predictors.pdf') as pdf:
		plt.bar(range(len(predictors)), scores) 
		plt.xticks(range(len(predictors)), predictors, rotation='vertical')
		pdf.savefig()
		plt.close()

#################################
# data specific functions ####### 
#################################
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2} 
def getTitle(person,isShort):
    name  = person["Name"];
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        title = title_search.group(1)
        if isShort:
        	title = getShortTitle(title,person['Sex'])
        if title in title_mapping:
          return title_mapping.get(title)
        else:
          return 0
    return 0

def getShortTitle(title,sex):
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if sex=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

family_id_mapping = {}
def getFamilyId(row):
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]
