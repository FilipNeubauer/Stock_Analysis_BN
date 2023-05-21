import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from confusion_matrix import my_confusion_matrix




# 49 %

df = pd.read_csv("Combined_News_DJIA.csv")



# print(df[df.isna().any(axis=1)])


# ---replacing nan values with empty string---
df.fillna("", inplace=True)
# print(df[df.isna().any(axis=1)])


# ---using only column Top1---
df = df.iloc[:, [0, 1, 2]]
# print(df)





# ---text cleaning function---
def text_cleaning(row):
    remove_b = row.replace("b\"", "").replace("b'", "")                   # removing b" and b'
    remove_punc = "".join([char for char in remove_b if char not in string.punctuation])     # removing punctuation
    remove_stop_words = [word for word in remove_punc.split() if word.lower() not in stopwords.words("english")]  # removing stopwords

    return remove_stop_words





transform = CountVectorizer(analyzer=text_cleaning).fit_transform(df["Top1"])     # asigning numbers to words


# print(transform)

tfidf_transformer = TfidfTransformer().fit_transform(transform)     # asigning significance to words


X_train, X_test, y_train, y_test = train_test_split(tfidf_transformer, df["Label"], test_size=0.2, shuffle=False) # splitting data into training and testing datasets



model = MultinomialNB().fit(X_train, y_train)
predict = model.predict(X_test)


conf = pd.DataFrame({"Predicted": predict, "Actual":y_test})

my_confusion_matrix(conf, "Predicted", "Actual", up=1, down=0)


# confusion_matrix = metrics.confusion_matrix(y_test, predict)

# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

# cm_display.plot()
# plt.show()





# df["combined"] = df[['Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7',
#        'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15',
#        'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23',
#        'Top24', 'Top25']].apply(lambda row: " ".join(row), axis=1)

