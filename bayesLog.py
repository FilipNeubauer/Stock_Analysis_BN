import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import json
from priceChange import priceChange
import pandas_ta as ta
import math


# doesnt work


if __name__ == '__main__':
    # Load the stock data
    ticker = "AAPL"


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

    df["direction7"] = [1 if i > 0 else 0 for i in df["change7"]]


    df.dropna(inplace=True)


    # ---creating RSI column + discretizing---
    df["rsi"] = ta.rsi(close=df["adjusted close"], length=14)
    df["rsi_shifted"] = df["rsi"].shift(7) # shifting it the down 7 bcs to predict the next 7 days change
    df["rsi_discretize"] = pd.qcut(x=df["rsi_shifted"], q=3, labels=["low", "middle", "high"]) 

    df.dropna(inplace=True)


    split_num = math.floor(len(df)*0.8)
    train = df.iloc[:split_num, :] # older
    test = df.iloc[split_num:, :] # newer

    # Define the predictor and response variables
    X = train[['rsi_shifted', 'change_shifted30', 'SMA_change7_shifted']].values
    y = train['direction7'].values



    print(X)
    print(y)



    # Standardize the predictor variables
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    print(X)

    


    # Define the model
    with pm.Model() as logistic_model:
        pm.glm.GLM(x=X, labels=['rsi_shifted', 'change_shifted30', 'SMA_change7_shifted'],
        y=y, family=pm.glm.families.Binomial())
        # Priors on the parameters
        # alpha = pm.Normal('alpha', mu=0, sd=10)
        # beta = pm.Normal('beta', mu=0, sd=10, shape=X.shape[1])

        # Likelihood function
        # likelihood = pm.Bernoulli('y', p=pm.math.sigmoid(alpha + pm.math.dot(beta, X.T)), observed=y)

        # Run the model
        trace = pm.sample(2000, chains=4, tune=1000)

    # Plot the posterior distribution of the parameters
    pm.plot_posterior(trace, var_names=['alpha', 'beta'])


    new_data = test[['rsi_shifted', 'change_shifted30', 'SMA_change7_shifted']].values
    with logistic_model:
        # inputs={'beta': beta, 'alpha': alpha, 'X': new_data}
        # x.set_values
        y_pred = pm.sample_posterior_predictive(trace, samples=1000, var_names=['y'], 
                                                )
        
    # Plot the posterior predictive distribution of the response variable
    plt.hist(y_pred['y'].mean(axis=0), bins=20)
    plt.show()

    print(y_pred["y"])

    
