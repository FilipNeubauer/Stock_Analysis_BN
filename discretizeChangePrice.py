
import pandas as pd
import json
import pandas_ta as ta
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import matplotlib.pyplot as plt
from pgmpy.factors.discrete.CPD import TabularCPD

from priceChange import priceChange
from changeRSI import changeRSI


ticker = "AAPL"

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


# ---creating the 7 days change column + discretizing (column to predict)---
df = priceChange(df, "adjusted close", 7)
df["7change_discretize"] = pd.cut(x=df["7change"], bins=[-1, 0, 1], labels=["down", "up"]) # maybe issue if the cahnge is higher/lower than 100 %


# ---creating the 30 days change column + discretizing---
df = priceChange(df, "adjusted close", 30)
df["30change_shifted"] = df["30change"].shift(7) # shifting it the down 7 bcs to predict the next 7 days change
df["30change_discretize"] = pd.cut(x=df["30change_shifted"], bins=[-1, 0, 1], labels=["down", "up"]) # maybe issue if the cahnge is higher/lower than 100 %


# ---creating RSI column + discretizing---
df["rsi"] = ta.rsi(close=df["adjusted close"], length=14)
df["rsi_shifted"] = df["rsi"].shift(7) # shifting it the down 7 bcs to predict the next 7 days change
df["rsi_discretize"] = pd.cut(x=df["rsi_shifted"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=["0-9%", "10-19%", "20-29%", "30-39%", "40-49%", "50-59%", "60-69%", "70-79%", "80-89%", "90-100%"]) 


# ---RSI crossing the 30 % line up or 70 % line down---
df = changeRSI(df, 30, 70) # columns: cross_back_over_bought, cross_back_over_sold


# ---droping all NAN values---
df.dropna(inplace=True)


# ---using only 200 data---
# df = df.iloc[:400]


model = BayesianModel([("30change_discretize", "7change_discretize"),  ("cross_back_over_bought", "7change_discretize"), ("cross_back_over_sold", "7change_discretize")]) #("rsi_discretize", "7change_discretize"),
# mle = MaximumLikelihoodEstimator(model, df)
model.fit(df, estimator=MaximumLikelihoodEstimator)


with open("output.txt", "w") as f:
    # ---printing the whole dataset or value_counts of some column---
    # print(df.to_string())
    # print(df["cross_back_over_sold"].value_counts())
    for cpd in model.get_cpds():
        print_full(cpd, f)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(df["date"], df["rsi"])
ax2.plot(df["date"], df["adjusted close"])
ax1.set_title('RSI Over Time')
ax2.set_title('Stock Price Over Time')
ax1.set_ylabel('RSI')
ax2.set_ylabel('Price')
ax1.legend(['RSI'])
ax2.legend(['Price'])

plt.show()

