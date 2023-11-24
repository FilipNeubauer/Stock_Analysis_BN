import joblib
import numpy as np
import pandas as pd
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer 


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]



model = joblib.load("./bayes_model_n_gram_3.pkl") # bayes_model_10.pkl

best_model = model.best_estimator_

model = best_model.named_steps['estimator']
tokenizer = best_model.named_steps["tokenizer"]


def print_probabilities():
    feature_log_probs = model.feature_log_prob_

    # Convert feature log probabilities to probabilities
    feature_probs = np.exp(feature_log_probs)

    sorted_vocabulary = sorted(tokenizer.vocabulary_.items(), key=lambda x: x[1])

    vocab_arr = [item[0] for item in sorted_vocabulary]


    # Create a DataFrame to represent the probabilistic table
    prob_table = pd.DataFrame(feature_probs, columns=vocab_arr)

    prob_table = prob_table.transpose()

    
    # probability ratio
    prob_table["prob_ratio"] = prob_table.iloc[:, 0] / prob_table.iloc[:,1]
    

    # priting the dataframe to a txt file
    with open("bayes_words_n_gram_3.txt", "w") as f:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.expand_frame_repr', False)

        prob_table = prob_table.sort_values("prob_ratio", ascending=False)

        print(prob_table, file=f)

        prob_table.to_csv("words_n_gram_3.csv")
        
        
        pd.reset_option('all')






print_probabilities()