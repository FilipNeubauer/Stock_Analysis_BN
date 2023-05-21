import json
import pandas as pd
from priceChange import priceChange
import pandas_ta as ta
import math






def preprocessData(ticker):
    with open("./data/data"+ ticker + ".json", "r") as f:
            data = json.load(f)

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
    df["SMA_change7_shifted"] = df["SMA_change7"].shift(7)
    # df["SMA_7_discretize"] = pd.cut(x=df["SMA_change7"], bins=[-1, 0, 1], labels=["down", "up"])
    df["SMA_7_discretize"] = pd.qcut(x=df["SMA_change7_shifted"], q=3, labels=["down","middle", "up"])


    # ---creating the 7 days change column + discretizing (column to predict)---
    df = priceChange(df, "adjusted close", 7)
    df["change_discretize7"] = pd.cut(x=df["change7"], bins=[-1, 0, 1], labels=["down", "up"]) # maybe issue if the cahnge is higher/lower than 100 %


    # ---creating the 30 days change column + discretizing---
    df = priceChange(df, "adjusted close", 30)
    df["change_shifted30"] = df["change30"].shift(7) # shifting it the down 7 bcs to predict the next 7 days change
    # df["change_discretize30"] = pd.cut(x=df["change_shifted30"], bins=[-1, 0, 1], labels=["down", "up"]) # maybe issue if the cahnge is higher/lower than 100 %
    df["change_discretize30"] = pd.qcut(x=df["change_shifted30"], q=3, labels=["down","middle", "up"]) 


    # ---creating RSI column + discretizing---
    df["rsi"] = ta.rsi(close=df["adjusted close"], length=14)
    df["rsi_shifted"] = df["rsi"].shift(7) # shifting it the down 7 bcs to predict the next 7 days change
    df["rsi_discretize"] = pd.qcut(x=df["rsi_shifted"], q=3, labels=["low", "middle", "high"]) 


    # ---droping all NAN values---
    df.dropna(inplace=True)


    # ---using only 200 data---
    # df = df.iloc[len(df)-200:]

    print(df["change_discretize7"].value_counts())


    split_num = math.floor(len(df)*0.8)
    train = df.iloc[:split_num, :] # older
    test = df.iloc[split_num:, :] # newer

    return train, test

