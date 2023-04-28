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

from priceChange import priceChange
from changeRSI import changeRSI
from confusion_matrix import confusion_matrix


ticker = "AAPL"
OVER_SOLD = 40
OVER_BOUGHT = 80

# ---counting occurences of specif combination---
def count(local_df, **kwargs):
    titles = ""
    for col, value in kwargs.items():
        local_df = local_df[local_df[col] == value]
        titles += col + "=" + str(value) + ", "
    return titles + ":" + str(len(local_df))


with open("./data/data"+ ticker + ".json", "r") as f:
    data = json.load(f)


# ---prints the whole table---
def print_full(cpd, f):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd, file=f)
    TabularCPD._truncate_strtable = backup


# ---getting the data from json and creating dataframe + converting strings to float---
my_list = []

for date, info in data["Time Series (Daily)"].items():
    row = [date] + [item for item in info.values()]
    my_list.append(row)

df = pd.DataFrame(my_list, columns=["date","open", "high", "low", "close", "adjusted close", "volume", "dividend amount", "split coefficient"])
df[["open", "high", "low", "close", "adjusted close","volume", "dividend amount", "split coefficient"]] = df[["open", "high", "low", "close", "adjusted close","volume", "dividend amount", "split coefficient"]].astype(str).astype(float)


# ---data from the oldest to the newest---
df = df.iloc[::-1]


df["SMA_7"] = df["adjusted close"].rolling(7).mean()
df = priceChange(df, "SMA_7", 7, name="SMA_change")
df["SMA_7_discretize"] = pd.cut(x=df["SMA_change7"], bins=[-1, 0, 1], labels=["down", "up"])


# ---creating the 7 days change column + discretizing (column to predict)---
df = priceChange(df, "adjusted close", 7)
df["change_discretize7"] = pd.cut(x=df["change7"], bins=[-1, 0, 1], labels=["down", "up"]) # maybe issue if the cahnge is higher/lower than 100 %


# ---creating the 30 days change column + discretizing---
df = priceChange(df, "adjusted close", 30)
df["change_shifted30"] = df["change30"].shift(7) # shifting it the down 7 bcs to predict the next 7 days change
df["change_discretize30"] = pd.cut(x=df["change_shifted30"], bins=[-1, 0, 1], labels=["down", "up"]) # maybe issue if the cahnge is higher/lower than 100 %


# ---creating RSI column + discretizing---
df["rsi"] = ta.rsi(close=df["adjusted close"], length=14)
df["rsi_shifted"] = df["rsi"].shift(7) # shifting it the down 7 bcs to predict the next 7 days change
df["rsi_discretize"] = pd.cut(x=df["rsi_shifted"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=["0-9%", "10-19%", "20-29%", "30-39%", "40-49%", "50-59%", "60-69%", "70-79%", "80-89%", "90-100%"]) 


# ---RSI crossing the 30 % line up or 70 % line down---
df = changeRSI(df, OVER_SOLD, OVER_BOUGHT) # columns: cross_back_over_bought, cross_back_over_sold


# ---droping all NAN values---
df.dropna(inplace=True)


# ---using only 200 data---
# df = df.iloc[len(df)-200:]

# print(df)
split_num = math.floor(len(df)*0.8)
train = df.iloc[:split_num, :] # older
test = df.iloc[split_num:, :] # newer

# print(train)
# print(train.info())
# print(test.info())

# with open("output.txt", "w") as f:
#     print(df.to_string(), file=f)



model = BayesianModel([("change_discretize30", "change_discretize7"), ("cross_back_over_bought", "change_discretize7"), ("cross_back_over_sold", "change_discretize7"), ("SMA_7_discretize", "change_discretize7")]) # ("rsi_discretize", "change_discretize7")
# mle = MaximumLikelihoodEstimator(model, df)
model.fit(train, estimator=MaximumLikelihoodEstimator)


with open("output.txt", "w") as f:
    # ---printing the whole dataset or value_counts of some column---
    # print(df.to_string(), file=f)
    # print(df["cross_back_over_sold"].value_counts(), file=f)
    for cpd in model.get_cpds():
        print_full(cpd, f)
    print(count(train, cross_back_over_sold=True), file=f)



test_cor = test[["change_discretize30", "change_discretize7",  "cross_back_over_bought", "cross_back_over_sold", "SMA_7_discretize"]] #"rsi_discretize",
# print(test.columns)
# print(model.nodes())
print("correlation score:", correlation_score(model, test_cor, test="chi_square", significance_level=0.05)) # , return_summary=True

# print(log_likelihood_score(model, test))

true_values = test_cor["change_discretize7"]

# print(true_values)
test_cor.drop("change_discretize7", inplace=True, axis=1)

y_pred = model.predict(test_cor)
test_cor["predicted_change_7"] = y_pred
test_cor["true_change_7"] = true_values

# with open("output.txt", "a") as f:
    # print(test_cor.to_string(), file=f)

confusion_matrix(test_cor, "predicted_change_7", "true_change_7")



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(test["date"], test["rsi"])
ax1.axhline(y=OVER_SOLD, color="red", linestyle = '--')
ax1.axhline(y=OVER_BOUGHT, color="red", linestyle = '--')
ax2.plot(test["date"], test["adjusted close"])
ax1.set_title('RSI Over Time')
ax2.set_title('Stock Price Over Time')
ax1.set_ylabel('RSI')
ax2.set_ylabel('Price')
ax1.legend(['RSI'])
ax2.legend(['Price'])

plt.show()


# equal frenquency binning
# logistická regrese
# Welkoksonův problém