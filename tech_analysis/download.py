import json
import requests

# request + saving data

ticker = "MSFT"
api_key = "0FY5ULXATOYL3FAM"
CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol='+ticker+'&apikey=' + api_key


r = requests.get(CSV_URL)
data = r.json()
with open("./data/data"+ ticker + ".json", "w") as f:
    json.dump(data, f)
