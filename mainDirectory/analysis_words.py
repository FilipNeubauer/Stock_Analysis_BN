import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# the highest occruence => trump


df = pd.read_csv("words_for_analysis.csv")


# df[df["count"] <1000]["count"]
# Create histogram
plt.hist(x=df[df["count"] <100]["count"], edgecolor='black', bins=100) # bins=np.logspace(np.log10(0.1), np.log10(10), 50)

# plt.yscale('log')

# hist, bins = np.histogram(df["count"], bins=10)

# Specify the number of bins to plot
# num_bins_to_plot = 5

# Plot only the first few bins
# plt.bar(bins[:num_bins_to_plot], hist[:num_bins_to_plot])




# Set labels and title
plt.xlabel('Frequency of occurrence')
plt.ylabel('Count of words')
plt.title('Occurence of words')

# Display the histogram
plt.show()









print(df)
# print(df["count"].max())
print(df["count"].nlargest(2))
print(df[df["count"] > 6000])