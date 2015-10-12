import re
import operator
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib.backends.backend_pdf import PdfPages
from pandas.core.frame import DataFrame
from tools import *
from enum import Enum



### features ###
allFeatures = ["Pclass", "Sex", "Parch", "Fare", "Embarked","Title","ShortTitle","Deck","AgeGroup","Age*Class","FareCategory","falimiy_freq"]
selectedFeatures = ["Age*Class", "Sex","ShortTitle","Fare","Deck","AgeGroup","falimiy_freq" ] 

### variables ###
age_groups = [[1,0,1],[2,1,5],[3,5,16],[4,16,25],[5,25,40],[6,40,60],[7,60,999]]
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2} 
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

###########################################
# Common Data Sceince functions ##########
##########################################

def normalizeData(data):
	data["Age"]=data["Age"].fillna(data["Age"].median())
	data["AgeGroup"]=data["Age"].apply(lambda age: getAgeGroup(age))
	data["Fare"]=data["Fare"].fillna(data["Fare"].median())
	data.loc[data["Sex"] == "male", "Sex"] = 0
	data.loc[data["Sex"] == "female", "Sex"] = 1
	data["Embarked"] = data["Embarked"].fillna("S")
	data.loc[data["Embarked"] == "S", "Embarked"] = 0
	data.loc[data["Embarked"] == "C", "Embarked"] = 1
	data.loc[data["Embarked"] == "Q", "Embarked"] = 2
	data["FamilySize"] = data["SibSp"] + data["Parch"] + 1	
	data["Age*Class"] = data["AgeGroup"] * data["Pclass"]
	data["NameLength"] = data["Name"].apply(lambda x: len(x))
	data["Title"] = data.apply(lambda x: getTitle(x,False),axis=1)
	data["ShortTitle"] = data.apply(lambda x: getTitle(x,True),axis=1)
 	getDecks(data)	
 	data["FamilyId"] = data.apply(getFamilyId,axis=1)
 	family_id_freq = initFamilityIdFreq(data)["FamilyId"]
 	data["falimiy_freq"]=data["FamilyId"].apply(lambda fid : family_id_freq.loc[fid])
 	data.loc[data["falimiy_freq"] <= 2, "FamilyId"] = "TooSmall"
 	data["IsBaby"]=data["Age"].apply(lambda age : isBaby(age))
 	createCategory(data,"Fare","FareCategory",10,40)
 	data["Fare"] = data["Fare"].apply(lambda fare : getFare(fare) )
    
 
def getAlgs():
# 	return [["DecisionTreeClassifier",DecisionTreeClassifier(random_state=0),allFeatures,1]]
	
	return  [["ranodom forest",
			RandomForestClassifier(random_state=1, n_estimators=250,max_features=2, min_samples_split=6, min_samples_leaf=3),
			selectedFeatures,
			2]]
# 	  	 ["gradient boosting",GradientBoostingClassifier(random_state=1, n_estimators=10, max_depth=3),
# 			["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title","FamilyId","Deck","IsBaby"],2],
# 	  	 ["logistic regression",LogisticRegression(random_state=1),
# 			["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","Title","Deck"],1]]

def analysFeatures(data):
	predictors = allFeatures;
	selector = SelectKBest(f_classif,k='all')
	selector.fit(data[predictors], data["Survived"])
	scores = -np.log10(selector.pvalues_)
	with PdfPages('output/predictors.pdf') as pdf:
		plt.bar(range(len(predictors)), scores) 
		plt.xticks(range(len(predictors)), predictors, rotation='vertical')
		pdf.savefig()
		plt.close()

#################################
# data specific functions ####### 
#################################


def getAgeGroup(age):
	for gid,min,max in age_groups: 
		if age > min and age <= max:
			return gid



def isBaby(age):
	if age <= 18:
		return 1;
	return 0;

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
    if title in ['Don','Master', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
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

def getFamilyId(row):
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    return family_id
   
def initFamilityIdFreq(data):
	return data.groupby(["FamilyId"]).agg({'FamilyId': 'size'})


def getDecks(data):
	data['Deck']=data['Cabin'].map(lambda x: substringsLocations_in_string(x, cabin_list))

def getFare(fare):
	if fare > 30:
		return 0
	if fare > 20 and fare <= 30:
		return 1
	if fare > 10 and fare <= 20:
		return 2
	else:
		return 3



	

