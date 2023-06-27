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
from nltk.stem import WordNetLemmatizer 
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
np.set_printoptions(threshold=np.inf)
import csv
from sklearn.linear_model import LogisticRegression
from sp500data import df
from sklearn import tree

import nltk
from nltk.stem import PorterStemmer

# nltk.download('punkt')  


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from confusion_matrix import my_confusion_matrix

from imblearn.over_sampling import SMOTE

from plot_roc_curve import plot_roc_curve
from precision_recall import plot_precision_recall


# ---text cleaning function---
def text_cleaning(row):
    remove_b = row.replace("b\"", "").replace("b'", "")                   # removing b" and b'
    remove_punc = "".join([char for char in remove_b if char not in string.punctuation])     # removing punctuation
    remove_stop_words = [word for word in remove_punc.split() if word.lower() not in stopwords.words("english")]  # removing stopwords


    # stemming
    # stemmed_words = [stemmer.stem(word) for word in remove_stop_words]  # Apply stemming to each word
    # stemmed_text = ' '.join(stemmed_words)  # Join the stemmed words back into a single string


    lem_words = [lemmatizer.lemmatize(word) for word in remove_stop_words] # same meanings together
    
    #---removing words that occur only once--- 
    # one_words = pd.read_csv("one_words.csv")
    # without_one_words = [word for word in lem_words if word not in list(one_words.iloc[:, 0])]


    #---manually removing words with possibly low significance (only my perspectif)---
    # remove_us = [word for word in lem_words if word not in ["us", "Us"]]

    # lower_words = [word.lower() for word in remove_us]



    lem_words = " ".join(lem_words)
    return lem_words 


#---oversamplerers---
smote = SMOTE(random_state=42)
oversampler = RandomOverSampler(random_state=42)
undersampler = RandomUnderSampler(random_state=42)

# article data
# df = pd.read_csv("Combined_News_DJIA.csv")

# sp500 data
# sp500 = pd.read_csv("sp500data.csv")



lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()



# ---replacing nan values with empty string---
# df.fillna("", inplace=True)


# df["combined"] = df[['Top1', "Top2", "Top3", 'Top4', 'Top5', 'Top6', 'Top7',
#       'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15',
#       'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23',
#        'Top24', 'Top25']].apply(lambda row: " ".join(row), axis=1) # 



#---testing with only 100 records---
# df = df.iloc[:100]


    
#---cheking imbalances---
# print(df["Label"].value_counts())



#---testing text_cleaning function---
# print(text_cleaning(df["combined"].iloc[0]))




df["combined"] = df["combined"].apply(text_cleaning)


cv = CountVectorizer() # , analyzer=text_cleaning, ngram_range=(2,2)
transform = cv.fit_transform(df["combined"])     # asigning numbers to words
# print(cv.vocabulary_)


feature_names = cv.get_feature_names_out()

# Get the indices of the "0" and "1" texts in the DataFrame
indices_0 = df[df["Label"] == 0].index
indices_1 = df[df["Label"] == 1].index

# Get the occurrences of each word in the "0" texts
word_counts_0 = transform[indices_0].sum(axis=0)

# Get the occurrences of each word in the "1" texts
word_counts_1 = transform[indices_1].sum(axis=0)

# Create a dictionary to store the word counts
word_counts = {
    "word": feature_names,
    "count_0": word_counts_0.tolist()[0],
    "count_1": word_counts_1.tolist()[0]
}

# Create a DataFrame from the word counts dictionary
word_counts_df = pd.DataFrame(word_counts)

# print(word_counts_df)


# ---count of most common words---
word_counts = np.array(transform.sum(axis=0)).flatten()


top_10_indices = np.argsort(word_counts)[::-1]


# Retrieve the top 10 words and their counts
top_10_words = [list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(index)] for index in top_10_indices]
top_10_counts = [word_counts[index] for index in top_10_indices]

# Print the top 10 words and their counts

df_counts = pd.DataFrame(index=top_10_words, columns=["count"])

#---getting words that occur only once---
n = 0

one_words = []
for word, count in zip(top_10_words, top_10_counts):
    # print("Word:", word, "- Count:", count)

    df_counts.loc[word] = count

    # if (count == 1):
    #     n += 1
    #     one_words.append(word)


df_counts = df_counts.sort_index()
# print(df_counts)


def create_csv_from_array(array, filename):
    df = pd.DataFrame(array)
    df.to_csv(filename, index=False)

# Example usage
# csv_filename = 'one_words.csv'

# create_csv_from_array(one_words, csv_filename)

# print(pd.read_csv("one_words.csv"))





# do not use
# transformer = TfidfTransformer().fit_transform(transform)     # asigning significance to words


# sp500_week_change or Label column
X_train, X_test, y_train, y_test = train_test_split(transform, df["sp500_week_change"], test_size=0.2, shuffle=False) # splitting data into training and testing datasets , random_state=42
# print(X_train)

# X_train, y_train = smote.fit_resample(X_train, y_train)
# X_train, y_train = undersampler.fit_resample(X_train, y_train)


# model = LogisticRegression(solver='liblinear', random_state=42).fit(X_train, y_train) 

model = MultinomialNB(alpha=0.1).fit(X_train, y_train)  #
# model = tree.DecisionTreeClassifier().fit(X_train, y_train)
# print(model.predict_proba(X_train))
# print(model)
predict = model.predict(X_test)




feature_log_probs = model.feature_log_prob_

# Convert feature log probabilities to probabilities
feature_probs = np.exp(feature_log_probs)

sorted_vocabulary = sorted(cv.vocabulary_.items(), key=lambda x: x[1])

vocab_arr = [item[0] for item in sorted_vocabulary]

# Create a DataFrame to represent the probabilistic table
prob_table = pd.DataFrame(feature_probs, columns=vocab_arr)



with open("output.txt", "w") as f:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    word_counts_df.set_index("word", inplace=True)

    prob_table = prob_table.transpose().sort_index()
    prob_table["count"] = df_counts["count"]
    prob_table["count_0"] = word_counts_df["count_0"]
    prob_table["count_1"] = word_counts_df["count_1"]

    prob_table["pomer"] =  prob_table.iloc[:,1] / prob_table.iloc[:, 0]


    print(prob_table.sort_values("pomer",  ascending=False), file=f) # sort by biggest impact on 0
    # print(prob_table.transpose().sort_values(1,  ascending=False), file=f) # sort by biggest impact on 0


    pd.reset_option('all')



conf = pd.DataFrame({"Predicted": predict, "Actual":y_test})

my_confusion_matrix(conf, "Predicted", "Actual", up=1, down=0)


plot_roc_curve(y_test, predict)
plot_precision_recall(y_test, predict)