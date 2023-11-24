import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer 
import numpy as np
import math
import matplotlib


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]




model = joblib.load("./bayes_model_n_gram_3.pkl")

results = model.cv_results_

#print(results)

param_grid = {
    "tokenizer__ngram_range": [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
    "tokenizer__min_df": [i for i in range(1, 100)],
}

param_grid_keys = list(param_grid.keys())
param_values = [results[f'param_{key}'] for key in param_grid_keys]

# print(results)


mean_test_scores = results['mean_test_score']
results_df = pd.DataFrame({'ngram': param_values[0], 'min_df': param_values[1], 'Mean_Test_Score': mean_test_scores})

results_df['ngram'] = results_df['ngram'].apply(lambda x: f"{x[0]}_{x[1]}")
replacement_dict = {
    "1_1": "1",
    "1_2" : "1, 2",
    "2_2" : "2",
    "2_3" : "2, 3",
    "3_3" : "3",
    "2_4" : "2, 3, 4",
    "3_4": "3, 4",
    "4_4": "4",
    "1_4": "1, 2, 3, 4"
}

print(results_df["min_df"])

results_df["ngram"] = results_df["ngram"].replace(replacement_dict)

heatmap_data = results_df.pivot(index='ngram', columns='min_df', values='Mean_Test_Score')

matplotlib.rcParams.update({'font.size': 14})



# Create the heatmap
plt.figure(figsize=(8, 4))
ax = sns.heatmap(heatmap_data, annot=False, cmap='viridis', fmt='.1f', cbar=True)


cbar = ax.collections[0].colorbar
cbar.set_label("Balanced Accuracy validační data")
# ax.figure.axes[1].tick_params(axis="x", labelsize=18) 
cbar.locator = plt.MaxNLocator(integer=True, prune="both", nbins=5)

plt.title('Grid Search')
plt.xlabel('Minimální počet výskytů slov a slovních spojení')
plt.ylabel('N-gram')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.yticks(rotation=0)
plt.xticks(ticks=[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99], labels=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], rotation=0)

plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")  # Replace "heatmap.png" with your desired file name
plt.show()
