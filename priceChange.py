
def priceChange(data, column, period, name="change"):
    col = data[column]
    mean = col.rolling(period, closed="both").mean()
    data[name + str(period)] = (col - mean) / mean
    return data