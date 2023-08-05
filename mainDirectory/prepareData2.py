import pandas as pd
from datetime import datetime
import json
import numpy as np

categories = set()
valid_arr = []

N = 2

my_set = set()

with open("mainDirectory/News_Category_Dataset_v3.json", "r") as file:
    for line in file:
        data = json.loads(line)

        my_set.add(data["category"])

        if data["category"]=="BUSINESS" or data["category"]=="POLITICS" or data["category"]=="MONEY" or data["category"]=="IMPACT" or data["category"]=="TECH" or data["category"]=="SCIENCE" or data["category"]=="U.S. NEWS" or data["category"]=="GOOD NEWS":
            categories.add(data["category"])
            valid_arr.append([data["date"], data["short_description"] + " " + data["headline"]])
            
# print(my_set)

# print(len(valid_arr))

# sorted_data = sorted(valid_arr, key=lambda x: x[0])
# for i in sorted_data:
#     print(i)

df_ = pd.DataFrame(valid_arr, columns=["Date", "short_description + headline"])
# print(set(df["Date"]))

# print(df)


arr_ = []
for i in set(df_["Date"]):
    arr_.append([i, ""])


arr_ = sorted(arr_, key=lambda x: x[0], reverse=True)


k = 0

begin = valid_arr[0][0]
# 
for i in range(len(valid_arr)):

    if valid_arr[i][0] == begin:
        arr_[k][1] += " " + valid_arr[i][1]
    else:
        k += 1
        arr_[k][1] += " " + valid_arr[i][1]
        begin = valid_arr[i][0]
    # print(arr_[k])


    

df_ = pd.DataFrame(arr_, columns=["Date", "combined"])
# print(df)




# # print(arr_)

# for i in range(len(df)):

#     for j in range(len(arr_)):
#         if df.iloc[i]["Date"] == arr_[j][0]:
#             arr_[j][1] += df.iloc[i]["short_description + headline"]

# print(pd.DataFrame(data=arr_, columns=["Date", "Title + description"]))



def priceChange(data, column, period, name="change"):
    col = data[column]
    mean = col.rolling(period, closed="both").mean()
    data[name] = (col - mean) / mean
    return data

def convert_date_format(date_str):
    date_obj = datetime.strptime(date_str, "%m/%d/%Y")
    formatted_date = date_obj.strftime("%Y-%m-%d")
    return formatted_date

sp500 = pd.read_csv("mainDirectory/SP500_historic_data.csv")

# # print(sp500)


sp500["Date"] = sp500["Date"].apply(lambda x: convert_date_format(x))
sp500 = sp500.iloc[::-1]


sp500["Price"] = sp500["Price"].apply(lambda x: float(x.replace(",", "")))
sp500["Price"] = sp500["Price"].astype(float)


def percentage_to_float(percentage_str):
    multiplier = -1 if percentage_str.startswith('-') else 1
    return multiplier * float(percentage_str.strip('-%')) / 100

sp500["floatChange"] = sp500["Change %"].apply(percentage_to_float)
sp500["label"] = pd.cut(x=sp500["floatChange"], bins=[-float('inf'), 0, float('inf')], labels=[0, 1])
print(sp500)



#sp500 = priceChange(sp500, "Price", N, name="label")
#sp500["label"] = pd.cut(x=sp500["label"], bins=[-1, 0, 1], labels=[0, 1]) # maybe issue if the cahnge is higher/lower than 100 %

# print(sp500.head(10)) 


sp500["label"] = sp500["label"].shift(-(N-1))




sp500 = sp500.dropna(subset=["label"])
sp500 = sp500.iloc[::-1]

sp500.reset_index(inplace=True, drop=True)



print(sp500)




my_dict = dict()

for i in range(len(sp500)):
    my_dict[sp500.iloc[i]["Date"]] = sp500.iloc[i]["label"]

# print(my_dict)

df_["label"] = np.nan


for i in range(len(df_)):

    date = df_.iloc[i]["Date"]

    if date in my_dict:
        df_.at[i, "label"] = my_dict[date]

    

df_.dropna(inplace=True)

df_["label"] = df_["label"].apply(lambda x: int(x))

df_.drop("Date", axis=1, inplace=True)

df_.to_csv("news_category_data.csv", index=False)
# print(df_)




# print(df)


# print(new_col)
# print(df)

# sp500 = pd.read_csv("sp500data.csv")




# data = json.loads("mainDirectory/News_Category_Dataset_v3.json")

# print(data)