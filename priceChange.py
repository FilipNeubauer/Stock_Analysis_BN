
def priceChange(data, column, period):
    col = data[column]
    mean = col.rolling(period, closed="both").mean()
    data[str(period) + "change"] = (col - mean) / mean
    return data