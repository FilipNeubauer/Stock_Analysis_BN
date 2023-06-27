import pandas as pd
from datetime import datetime

def priceChange(data, column, period, name="change"):
    col = data[column]
    mean = col.rolling(period, closed="both").mean()
    data[name + str(period)] = (col - mean) / mean
    return data

def convert_date_format(date_str):
    date_obj = datetime.strptime(date_str, "%m/%d/%Y")
    formatted_date = date_obj.strftime("%Y-%m-%d")
    return formatted_date




sp500 = pd.read_csv("sp500data.csv")


sp500["Date"] = sp500["Date"].apply(lambda x: convert_date_format(x))
sp500 = sp500.iloc[::-1]


sp500["Price"] = sp500["Price"].astype(float)


sp500 = priceChange(sp500, "Price", 1, name="weekChange")
# checking max
# print(sp500["weekChange7"].max()) 
sp500["weekChange_binary"] = pd.cut(x=sp500["weekChange1"], bins=[-1, 0, 1], labels=[0, 1]) # maybe issue if the cahnge is higher/lower than 100 %
sp500.dropna(inplace=True)


df = pd.read_csv("Combined_News_DJIA.csv")
df.fillna("", inplace=True)
df["combined"] = df[['Top1', "Top2", "Top3", 'Top4', 'Top5', 'Top6', 'Top7',
      'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15',
      'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23',
       'Top24', 'Top25']].apply(lambda row: " ".join(row), axis=1) # 


# print(sp500["weekChange_binary"].value_counts())
# print(sp500.head(100))
last = df["Date"].iloc[-1]
first = df["Date"].iloc[0]


sp500["Date"] = pd.to_datetime(sp500["Date"])
start_date = pd.to_datetime(first)
end_date = pd.to_datetime(last)

filtered_df = sp500.loc[(sp500["Date"] >= start_date) & (sp500["Date"] <= end_date)]
filtered_df = filtered_df.reset_index(drop=True)

df["sp500_week_change"] = filtered_df["weekChange_binary"]

if __name__ == "__main__":
    print(df)
