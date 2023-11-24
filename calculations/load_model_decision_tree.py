import joblib
import numpy as np
import pandas as pd
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer 
import string
from sklearn.tree import export_graphviz, plot_tree
import graphviz


import matplotlib.pyplot as plt


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]



model = joblib.load("./models/decision_tree_model_2.pkl")

best_model = model.best_estimator_

model = best_model.named_steps['estimator']
tokenizer = best_model.named_steps["tokenizer"]

# Assuming 'clf' is your fitted decision tree classifier
#export_graphviz(model, out_file="tree.dot") #  feature_names=tokenizer.get_feature_names_out()

fig = plt.figure(figsize=(25,20))
_ = plot_tree(model,                 #    feature_names=iris.feature_names,  
                #    class_names=iris.target_names,
                   filled=True)
plt.show()
# graph = graphviz.Source.from_file("tree.dot", format="png")
# graph.render("decision_tree")