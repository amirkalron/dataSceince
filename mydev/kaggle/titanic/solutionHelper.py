import re


def normalizeData(titanic):
	titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
	titanic["Fare"]=titanic["Fare"].fillna(titanic["Fare"].median())
	titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
	titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
	titanic["Embarked"] = titanic["Embarked"].fillna("S")
	titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
	titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1    
	titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    addTitles(titanic);


#set titles
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}


#add person's title as indicator
def addTitles(titanic):
    titles = titanic["Name"].apply(get_title)
    for k,v in title_mapping.items():
    titles[titles == k] = v
    titanic["Title"] = titles

def get_title(name):
    # Use a regular expression to search for a title.  Titles always 
    # consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


def getPredictors():
	return ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

