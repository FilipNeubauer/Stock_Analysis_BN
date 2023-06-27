import pandas as pd

def my_confusion_matrix(data, predicted_col, true_col, up="up", down="down"):
    tp = data.loc[(data[predicted_col] == up) & (data[true_col] == up)]
    fp = data.loc[(data[predicted_col] == up) & (data[true_col] == down)]
    fn = data.loc[(data[predicted_col] == down) & (data[true_col] == up)]
    tn = data.loc[(data[predicted_col] == down) & (data[true_col] == down)]

    df = pd.DataFrame({"Actual up":[len(tp), len(fn)], "Actual Down":[len(fp), len(tn)]}, index=["Predicted up", "Predicted down"])

    performance = (len(tp) + len(tn)) / (len(tp) + len(tn) + len(fp) + len(fn))
    accuracyP = (len(tp) / (len(tp) + len(fp)))

    if (len(tn) + len(fn)) == 0:
        accuracyN = "no negatives predicted"
    else:
        accuracyN = (len(tn) / (len(tn) + len(fn)))



    print(df)
    print("Performance: ", performance)
    print("Accuracy Positives: ", accuracyP)
    print("Accuracy Negatives: ", accuracyN)