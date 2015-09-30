import re
import operator
import pandas as pandas
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

###########################################
# Common Data Sceince functions ##########
##########################################
def normalizeData(titanic):
	titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
	titanic["Fare"]=titanic["Fare"].fillna(titanic["Fare"].median())
	titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
	titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
	titanic["Embarked"] = titanic["Embarked"].fillna("S")
	titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
	titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
	titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
	titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]	
	titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
        titanic["Title"] = titanic["Name"].apply(lambda x: getTitle(x))	
	
 	fId = titanic.apply(getFamilyId,axis=1)
        fId[titanic["FamilySize"] < 3]=-1
        titanic["FamilyId"]=fId

def getPredictors():
	return ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize", "Title", "FamilyId"]

def getAlgs():
	return  [[RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2),getPredictors(),2],
	  	 [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),getPredictors(),1],
	  	 [LogisticRegression(random_state=1),getPredictors(),1]]

#################################
# data specific functions ####### 
#################################
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2} 
def getTitle(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        title = title_search.group(1)
        if title in title_mapping:
          return title_mapping.get(title)
        else:
          return 0
    return 0

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
