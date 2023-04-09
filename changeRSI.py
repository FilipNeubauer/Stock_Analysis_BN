def changeRSI(df, over_sold, over_bought):
    df["rsi_yesterday"] = df["rsi"].shift(1) # shifting the rsi down 1

    over_sold = float(over_sold)
    over_bought = float(over_bought)
    
    # ---crossing the over_sold line up---
    df.loc[(df["rsi_yesterday"] < over_sold) & (df["rsi"] > over_sold) & (df["rsi"] > df["rsi_yesterday"]), "cross_back_over_sold"] = True
    df.loc[(df["rsi_yesterday"] >= over_sold) | (df["rsi"] <= over_sold) | (df["rsi"] <= df["rsi_yesterday"]), "cross_back_over_sold"] = False

    # ---crossing the over_bough line down---
    df.loc[(df["rsi_yesterday"] > over_bought) & (df["rsi"] < over_bought) & (df["rsi"] < df["rsi_yesterday"]), "cross_back_over_bought"] = True
    df.loc[(df["rsi_yesterday"] <= over_bought) | (df["rsi"] >= over_bought) | (df["rsi"] >= df["rsi_yesterday"]), "cross_back_over_bought"] = False


    return df
    
