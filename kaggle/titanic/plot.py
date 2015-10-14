from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot 
import os
from props import  * 

def plotClassifier(clf,name):
    if hasattr(clf, "estimators_") :
        for id,estimator in enumerate(clf.estimators_): 
            __plotTree(estimator,name + str(id) )
    if hasattr(clf, "tree_"):
        __plotTree(clf,name);   
        
    
    
    
def __plotTree(clf,name):  
    tree.export_graphviz(clf,out_file=outputdir + name) 
    dot_data = StringIO() 
    tree.export_graphviz(clf,out_file=dot_data)   
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_pdf(outputdir +  name + '.pdf') 
    os.remove(outputdir + name)  

#plot utilities