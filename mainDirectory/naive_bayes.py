import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from confusion_matrix import my_confusion_matrix
import inflect
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
import numpy as np


from prepareData2 import df_
undersampler = RandomUnderSampler(random_state=42)



# tokenizer object
cv = CountVectorizer() # ngram_range=(2,2)

# stemmer object
""""
porter_stemmer = PorterStemmer()
"""

# lemmatizer object
lemmatizer = WordNetLemmatizer()


my_stop_words = []       # stop words without punctuation
for word in stopwords.words("english"):
    my_word = ""
    for char in word:
        if char not in string.punctuation:
            my_word += char
    my_stop_words.append(my_word)

other_not_included = ["hed", "itd", "theyd", "youd"] # shed, wed, id are common words but he had, it had, they had, you had
for word in other_not_included:
    my_stop_words.append(word)


# plural word -> singular word
def singularize_word(word):
    p = inflect.engine()
    singular_word = p.singular_noun(word)
    if singular_word:
        return singular_word
    else:
        return word


# prepare dataset
def prepare_dataset(df):
    df.fillna("", inplace=True) # filling empty fields
    df["combined"] = df[['Top1', "Top2", "Top3", 'Top4', 'Top5', 'Top6', 'Top7',
      'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15',
      'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23',
       'Top24', 'Top25']].apply(lambda row: " ".join(row), axis=1) # combining all columns into one
    
    new_df = pd.DataFrame(data={"combined": df["combined"], "label": df["Label"]})
    
    return new_df


# text cleaning 
def text_cleanning(row):
    remove_b = row.replace("b\"", "").replace("b'", "")   # removing b" and b'
    remove_punc = "".join([char for char in remove_b if char not in string.punctuation])     # removing punctuation !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    remove_stop_words = [word for word in remove_punc.split() if word.lower() not in my_stop_words]  # removing stopwords

    # comparing different techniques for removing suffixis
    """
    for i in remove_stop_words:
        print(i, " : ", porter_stemmer.stem(i, to_lowercase=False), " : ", lemmatizer.lemmatize(i), " : ", singularize_word(i))
    """
    
    lem_words = [lemmatizer.lemmatize(word) for word in remove_stop_words] # base of the words
    singular_words = [singularize_word(word) for word in lem_words]         # plural -> singular

    return " ".join(singular_words)


def get_word_counts():
    feature_names = cv.get_feature_names_out()

    # Get the indices of the "0" and "1" texts in the DataFrame
    indices_0 = df[df["label"] == 0].index
    indices_1 = df[df["label"] == 1].index

    # Get the occurrences of each word in the "0" texts
    word_counts_0 = tokenized_words[indices_0].sum(axis=0)

    # Get the occurrences of each word in the "1" texts
    word_counts_1 = tokenized_words[indices_1].sum(axis=0)

    # Create a dictionary to store the word counts
    word_counts = {
        "word": feature_names,
        "count_0": word_counts_0.tolist()[0],
        "count_1": word_counts_1.tolist()[0]
    }

    # Create a DataFrame from the word counts dictionary
    word_counts_df = pd.DataFrame(word_counts)



    # ---count of most common words---
    word_counts = np.array(tokenized_words.sum(axis=0)).flatten()


    top_10_indices = np.argsort(word_counts)[::-1]


    # Retrieve the top 10 words and their counts
    top_10_words = [list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(index)] for index in top_10_indices]
    top_10_counts = [word_counts[index] for index in top_10_indices]

    # Print the top 10 words and their counts

    df_counts = pd.DataFrame(index=top_10_words, columns=["count"])

    for word, count in zip(top_10_words, top_10_counts):
    # print("Word:", word, "- Count:", count)

        df_counts.loc[word] = count

    # if (count == 1):
    #     n += 1
    #     one_words.append(word)


    df_counts = df_counts.sort_index()



    return word_counts_df, df_counts




# printing the prob table
def print_probabilities():
    feature_log_probs = model.feature_log_prob_

    # Convert feature log probabilities to probabilities
    feature_probs = np.exp(feature_log_probs)

    sorted_vocabulary = sorted(cv.vocabulary_.items(), key=lambda x: x[1])

    vocab_arr = [item[0] for item in sorted_vocabulary]

    # Create a DataFrame to represent the probabilistic table
    prob_table = pd.DataFrame(feature_probs, columns=vocab_arr)



    with open("mainDirectory/output.txt", "w") as f:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)

        word_counts_df, df_counts = get_word_counts()

        word_counts_df.set_index("word", inplace=True)

        prob_table = prob_table.transpose().sort_index()
        prob_table["count"] = df_counts["count"]
        prob_table["count_0"] = word_counts_df["count_0"]
        prob_table["count_1"] = word_counts_df["count_1"]

        prob_table["pomer"] =  prob_table.iloc[:,1] / prob_table.iloc[:, 0]


        print(prob_table.sort_values("pomer",  ascending=False), file=f) # sort by biggest impact on 0
        # print(prob_table.transpose().sort_values(1,  ascending=False), file=f) # sort by biggest impact on 0


        pd.reset_option('all')
    

# load data
df = pd.read_csv("Combined_News_DJIA.csv")

# prepare dataset
df = prepare_dataset(df)

# adding data2
df = pd.concat([df, df_], ignore_index=True)

# getting how many 0 and 1
"""
print(df["label"].value_counts())
"""

# clean each row
df["combined"] = df['combined'].apply(text_cleanning)

# tokenizing
tokenized_words = cv.fit_transform(df["combined"])     # asigning numbers to words

# getting vocabulary
"""
print(cv.vocabulary_)
"""

# spliting data into training and testing date sets
X_train, X_test, y_train, y_test = train_test_split(tokenized_words, df["label"], test_size=0.1, shuffle=True, random_state=42) # 

# random under samplerer
"""
X_train, y_train = undersampler.fit_resample(X_train, y_train)
"""


# creating the model and fitting the data
model = MultinomialNB(alpha=1, fit_prior=False)
model.fit(X_train, y_train)

# making predictions for testing
predict = model.predict(X_test)

# printing the confusion matrix
conf = pd.DataFrame({"Predicted": predict, "Actual":y_test})
my_confusion_matrix(conf, "Predicted", "Actual", up=1, down=0)

# print the probabilities to output.txt
print_probabilities()


