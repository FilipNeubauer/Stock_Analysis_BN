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





from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from confusion_matrix import my_confusion_matrix

from imblearn.over_sampling import SMOTE


# ---text cleaning function---
def text_cleaning(row):
    remove_b = row.replace("b\"", "").replace("b'", "")                   # removing b" and b'
    remove_punc = "".join([char for char in remove_b if char not in string.punctuation])     # removing punctuation
    remove_stop_words = [word for word in remove_punc.split() if word.lower() not in stopwords.words("english")]  # removing stopwords

    lem_words = [lemmatizer.lemmatize(word) for word in remove_stop_words] # same meanings together

    
    # one_words = pd.read_csv("one_words.csv")
    # without_one_words = [word for word in lem_words if word not in one_words[0]]


    # print(without_one_words)
    return lem_words


#---oversamplerers---
smote = SMOTE(random_state=42)
oversampler = RandomOverSampler(random_state=42)
undersampler = RandomUnderSampler(random_state=42)


df = pd.DataFrame(data={"col" : ["Hello, this is a testing sentence.", "Hello, this is a second sentence.","Hi, okay testing", "Hello, this is a testing.", ], "Label":[1, 0, 1, 1]}) #"second"
print(df)

lemmatizer = WordNetLemmatizer()


# ---replacing nan values with empty string---
df.fillna("", inplace=True)
# print(df[df.isna().any(axis=1)])




# df = df.iloc[:100]





# with open("output.txt", "w") as f:
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#     pd.set_option('display.max_colwidth', -1)
    
#     # All dataframes hereafter reflect these changes.
#     print(df["combined"], file=f)
    
#     # Resets the options
#     pd.reset_option('all')

    
# ---cheking imbalances---
# print(df["Label"].value_counts())


# ---using only column Top1---
# df = df.iloc[:, [0, 1, 2]]
# print(df)







########errrorrrrrrr

# print(text_cleaning(df["combined"].iloc[0]))




# 
cv = CountVectorizer(analyzer=text_cleaning)
transform = cv.fit_transform(df["col"])     # asigning numbers to words
# print(cv.vocabulary_)
# print(transform)

# print(transform.sum(axis=0).flatten())
# ---count of most common words---
word_counts = np.array(transform.sum(axis=0)).flatten()
# print(word_counts)


# Get indices of top 10 words with most occurrences
top_10_indices = np.argsort(word_counts)[::-1]
# top_10_indices = np.argsort(word_counts)[-100:][::-1]
# print(top_10_indices)


# Retrieve the top 10 words and their counts
top_10_words = [list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(index)] for index in top_10_indices]
top_10_counts = [word_counts[index] for index in top_10_indices]

# Print the top 10 words and their counts

n = 0
one_words = []
df_counts = pd.DataFrame(index=top_10_words, columns=["count"])


for word, count in zip(top_10_words, top_10_counts):
    # print("Word:", word, "- Count:", count)

    # if (count == 1):
    #     n += 1
    #     one_words.append(word)

    df_counts.loc[word] = count



print(df_counts)
# print(n)


# def create_csv_from_array(array, filename):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(array)

# # Example usage
# csv_filename = 'one_words.csv'

# create_csv_from_array(one_words, csv_filename)

# print(pd.read_csv("one_words.csv"))

def create_csv_from_array(array, filename):
    df = pd.DataFrame(array)
    df.to_csv(filename, index=False)

# Example usage
# csv_filename = 'one_words.csv'

# create_csv_from_array(one_words, csv_filename)

# print(pd.read_csv("one_words.csv"))





# do not use
# tfidf_transformer = TfidfTransformer().fit_transform(transform)     # asigning significance to words





# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# X_train_random, y_train_random = oversampler.fit_resample(X_train, y_train)



# print(transform[:2])
# print()
print(cv.vocabulary_)

# print(transform[:3])
# print(df.loc[:1, ["Label"]])

print(transform[:])
print(df.loc[:,["Label"]])

#model = MultinomialNB(alpha=0).fit(transform[:4], df.loc[:3, ["Label"]]) #alpha=0.1
# print(model.predict_proba(transform[4]))
# print(model)




from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42).fit(transform[:4], df.loc[:3, ["Label"]])

from sklearn import tree

fig = plt.figure(figsize=(12,6))
tree.plot_tree(model)
plt.show()

print(tree.export_text(model))


# feature_log_probs = model.feature_log_prob_

# Convert feature log probabilities to probabilities
# feature_probs = np.exp(feature_log_probs)

print(cv.vocabulary_)
# print(model.feature_names_in_)
# print(feature_probs)

sorted_vocabulary = sorted(cv.vocabulary_.items(), key=lambda x: x[1])

vocab_arr = [item[0] for item in sorted_vocabulary]

# Create a DataFrame to represent the probabilistic table
# prob_table = pd.DataFrame(feature_probs, columns=vocab_arr)

# Print the probabilities
# print(X_train)


# with open("output.txt", "w") as f:
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#     pd.set_option('display.max_colwidth', -1)
#     # print(np.exp(model.feature_log_prob_),file=f)
#     # print(model.feature_names_in_, file=f)

#     # prob_table.iloc[[0, 1]] = prob_table.iloc[[1, 0]].values
    
#     # prob_table = prob_table.transpose().sort_index()
#     # prob_table["count"] = df_counts["count"]


#     print(df, file=f)
#     # print(prob_table, file=f) # sort by biggest impact on 0

#     # print(prob_table.sort_values(0, ascending=False), file=f)


#     # print(prob_table.transpose().sort_values(0,  ascending=False), file=f) # sort by biggest impact on 0
#     # print(prob_table.transpose().sort_values(1,  ascending=False), file=f) # sort by biggest impact on 0

#     # print(model.predict_proba(transform[2]),file=f)
#     # print(transform.toarray(), file=f)

    

#     pd.reset_option('all')














