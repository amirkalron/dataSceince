import re

def get_title(name):
    # Use a regular expression to search for a title.  Titles always 
    # consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def fixData(titanic):
	titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
	titanic["Fare"]=titanic["Fare"].fillna(titanic["Fare"].median())
	titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
	titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
	titanic["Embarked"] = titanic["Embarked"].fillna("S")
	titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
	titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
	titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

def getPredictors():
	return ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

