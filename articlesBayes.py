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



from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from confusion_matrix import my_confusion_matrix

from imblearn.over_sampling import SMOTE


#---oversamplerers---
smote = SMOTE(random_state=42)
oversampler = RandomOverSampler(random_state=42)
undersampler = RandomUnderSampler(random_state=42)




df = pd.read_csv("Combined_News_DJIA.csv")

lemmatizer = WordNetLemmatizer()


# ---replacing nan values with empty string---
df.fillna("", inplace=True)
# print(df[df.isna().any(axis=1)])


df["combined"] = df[['Top1', "Top2", "Top3", 'Top4', 'Top5', 'Top6', 'Top7',
      'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15',
      'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23',
       'Top24', 'Top25']].apply(lambda row: " ".join(row), axis=1) # 




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





# ---text cleaning function---
def text_cleaning(row):
    remove_b = row.replace("b\"", "").replace("b'", "")                   # removing b" and b'
    remove_punc = "".join([char for char in remove_b if char not in string.punctuation])     # removing punctuation
    remove_stop_words = [word for word in remove_punc.split() if word.lower() not in stopwords.words("english")]  # removing stopwords

    lem_words = [lemmatizer.lemmatize(word) for word in remove_stop_words] # same meanings together


    return lem_words






# 
cv = CountVectorizer(analyzer=text_cleaning)
transform = cv.fit_transform(df["combined"])     # asigning numbers to words
print(cv.vocabulary_)


# ---count of most common words---
word_counts = np.array(transform.sum(axis=0)).flatten()

# Get indices of top 10 words with most occurrences
top_10_indices = np.argsort(word_counts)[-100:][::-1]

# Retrieve the top 10 words and their counts
top_10_words = [list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(index)] for index in top_10_indices]
top_10_counts = [word_counts[index] for index in top_10_indices]

# Print the top 10 words and their counts
for word, count in zip(top_10_words, top_10_counts):
    print("Word:", word, "- Count:", count)




# do not use
# tfidf_transformer = TfidfTransformer().fit_transform(transform)     # asigning significance to words



X_train, X_test, y_train, y_test = train_test_split(transform, df["Label"], test_size=0.2, random_state=42) # splitting data into training and testing datasets , 

# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# X_train_random, y_train_random = oversampler.fit_resample(X_train, y_train)


model = MultinomialNB().fit(X_train, y_train)
# print(model.predict_proba(X_train))
# print(model)
predict = model.predict(X_test)




feature_log_probs = model.feature_log_prob_

# Convert feature log probabilities to probabilities
feature_probs = np.exp(feature_log_probs)

# Create a DataFrame to represent the probabilistic table
prob_table = pd.DataFrame(feature_probs, columns=cv.vocabulary_)

# Print the probabilities
# print(X_train)


with open("output.txt", "w") as f:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    
    # print(prob_table.transpose(), file=f)
    # print(transform.toarray(), file=f)

    

    pd.reset_option('all')



conf = pd.DataFrame({"Predicted": predict, "Actual":y_test})

my_confusion_matrix(conf, "Predicted", "Actual", up=1, down=0)









