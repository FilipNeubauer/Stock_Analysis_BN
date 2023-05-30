import pandas as pd
import json
import pandas_ta as ta
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import matplotlib.pyplot as plt
from pgmpy.factors.discrete.CPD import TabularCPD
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from pgmpy.inference import VariableElimination
from pgmpy.metrics import correlation_score
from pgmpy.metrics import log_likelihood_score
import math

from tech_analysis.priceChange import priceChange
from tech_analysis.changeRSI import changeRSI
from confusion_matrix import my_confusion_matrix
from tech_analysis.preprocessData import preprocessData


# ---counting occurences of specif combination---
def count(local_df, **kwargs):
    titles = ""
    for col, value in kwargs.items():
        local_df = local_df[local_df[col] == value]
        titles += col + "=" + str(value) + ", "
    return titles + ":" + str(len(local_df))


# ---prints the whole table---
def print_full(cpd, f):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd, file=f)
    TabularCPD._truncate_strtable = backup



train, test = preprocessData("MSFT")

# print(train)
# print(train.info())
# print(test.info())

# with open("output.txt", "w") as f:
#     print(df.to_string(), file=f)



model = BayesianModel([("change_discretize30", "change_discretize7"), ("SMA_7_discretize", "change_discretize7"), ("rsi_discretize", "change_discretize7")]) # ("rsi_discretize", "change_discretize7") ("cross_back_over_bought", "change_discretize7") ("cross_back_over_sold", "change_discretize7"),
# mle = MaximumLikelihoodEstimator(model, df)
model.fit(train, estimator=MaximumLikelihoodEstimator)


with open("output.txt", "w") as f:
    # ---printing the whole dataset or value_counts of some column---
    # print(df.to_string(), file=f)
    # print(df["cross_back_over_sold"].value_counts(), file=f)
    print(model, file=f)

    for cpd in model.get_cpds():
        print_full(cpd, f)
    # print(count(train, cross_back_over_sold=True), file=f)



test_cor = test[["change_discretize30", "change_discretize7",   "SMA_7_discretize", "rsi_discretize"]] #"rsi_discretize", "cross_back_over_bought", "cross_back_over_sold",
# print(test.columns)
# print(model.nodes())

# print(log_likelihood_score(model, test))

true_values = test_cor["change_discretize7"]

# print(true_values)
test_cor.drop("change_discretize7", inplace=True, axis=1)

y_pred = model.predict(test_cor)
test_cor["predicted_change_7"] = y_pred
test_cor["true_change_7"] = true_values

# with open("output.txt", "a") as f:
    # print(test_cor.to_string(), file=f)

my_confusion_matrix(test_cor, "predicted_change_7", "true_change_7")



# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(test["date"], test["rsi"])
# ax1.axhline(y=OVER_SOLD, color="red", linestyle = '--')
# ax1.axhline(y=OVER_BOUGHT, color="red", linestyle = '--')
# ax2.plot(test["date"], test["adjusted close"])
# ax1.set_title('RSI Over Time')
# ax2.set_title('Stock Price Over Time')
# ax1.set_ylabel('RSI')
# ax2.set_ylabel('Price')
# ax1.legend(['RSI'])
# ax2.legend(['Price'])

# plt.show()


# equal frenquency binning
# logistická regrese
# Welkoksonův problém

# try EMA
# try 30 day MA