# from psaw import PushshiftAPI
# import datetime as dt
# import requests


# api = PushshiftAPI()

# start_epoch=int(dt.datetime(2017, 1, 1).timestamp())

# result = list(api.search_submissions(after=start_epoch,
#                             subreddit='politics',
#                             filter=['url','author', 'title', 'subreddit'],
#                             limit=10))

# print(result)

from bs4 import BeautifulSoup as soup
import requests

from datetime import date

url = "https://www.bbc.com/news/business"

# url = "https://www.bbc.co.uk/%7BassetUri%7D/page/50"

html = requests.get(url)


bsobj = soup(html.content, "lxml")


# lx-stream__post-container

for li in bsobj.find_all("li", {"class":"lx-stream__post-container"}):
    # for title in soup(li, "lxml").find_all("span", {"class":"lx-stream-post__header-text"}):
    #     print(title.text)
    title = li.find("span", class_="lx-stream-post__header-text").text
    print(title)
    time_date = li.find("span", class_="qa-post-auto-meta").text
    # print(time_date)



# for li in bsobj.find_all("span", {"class":"lx-stream-post__header-text"}):
#     print(li.text)

# for li in bsobj.find_all("span", {"class":"lx-stream-post__header-text"}):
#     print(li.text)

# print(bsobj)