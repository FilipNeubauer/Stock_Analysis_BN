from preprocessData import preprocessData
from confusion_matrix import my_confusion_matrix

import pandas as pd
import sys

from scipy.special import expit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# np.set_printoptions(threshold=np.inf)
# --- input data change_shifted30, rsi_shifted, SMA_change7_shifted
# --- output data change_discretize7

np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=np.inf)




# "change_discretize30", "change_discretize7",   "SMA_7_discretize"
# "change_shifted30", "SMA_change7_shifted"

train, test = preprocessData("MSFT")

with open("output.txt", "w") as f:
    print(train[["change_discretize7"]].value_counts(), file=f)
    # print(train[["change_shifted30", "SMA_change7_shifted", "change_discretize7"]].to_string(), file=f)

df_train_x = train[["change_shifted30", "SMA_change7_shifted"]] #rsi_shifted
# df_train_x["SMA_7_discretize"] = [1 if i == "up" else 0 for i in df_train_x["SMA_7_discretize"]] 
# df_train_x["change_discretize30"] = [1 if i == "up" else 0 for i in df_train_x["change_discretize30"]] 
df_train_y = train[["change_discretize7"]]
df_train_y["change_discretize7"] = [1 if i == "up" else 0 for i in df_train_y["change_discretize7"]]

arr_x = df_train_x.to_numpy()
arr_y = df_train_y.to_numpy()
arr_y = np.array([i for i in df_train_y["change_discretize7"]])
# print("arr_y: ", np.unique(arr_y, return_counts=True))


df_test_x = test[["change_shifted30", "SMA_change7_shifted"]] #rsi_shifted

# df_test_x["SMA_7_discretize"] = [1 if i == "up" else 0 for i in df_test_x["SMA_7_discretize"]] 
# df_test_x["change_discretize30"] = [1 if i == "up" else 0 for i in df_test_x["change_discretize30"]] 

df_test_y = test[["change_discretize7"]]
df_test_y["change_discretize7"] = [1 if i == "up" else 0 for i in df_test_y["change_discretize7"]]

test_x = df_test_x.to_numpy()
test_y = df_test_y.to_numpy()
test_y = np.array([i for i in df_test_y["change_discretize7"]])



model = LogisticRegression(solver='liblinear', random_state=0, ) #class_weight = 'balanced'
model.fit(arr_x, arr_y)



actual = pd.Series(test_y, name="actual")
predicted = pd.Series(model.predict(test_x), name="predicted")

predict =np.array([[1, 0]])

print(model.predict(predict))

df = actual.to_frame().join(predicted)

my_confusion_matrix(df, "predicted", "actual", up=1, down=0)





# print(test_y, model.predict(test_x))


# print(confusion_matrix(test_y, model.predict(test_x)))
# print(model.score(test_x, test_y))
print(model.intercept_)
print(model.coef_)