from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas import Series, DataFrame

url = 'https://www.studentnewsdaily.com/archive/daily-news-article/'

result = requests.get(url)
soup = BeautifulSoup(result.content, 'lxml')
content = soup.findAll('section')
summary = content[1].find(class_='col-md-10 col-lg-10 col-xl-10')
l = summary.findAll('h4')
series = [ele.get_text() for ele in l]
print(series)