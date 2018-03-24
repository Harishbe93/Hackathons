# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:00:25 2018

@author: S5RXCY
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import gc
from wordcloud import WordCloud,STOPWORDS
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
import time

os.getcwd()
os.chdir("\\\\CHRB1067.CORP.GWPNET.COM\\homes\\C\\S5RXCY\\Documents\\\\HactionLab\\Task 2")
train = pd.read_excel("Task2 new.xlsx")
os.chdir("DBpedia")

### Get the date, description, Lat and Long from DB pedia
for i in range(len(train["Disaster Name"])):
    try:
       with open(train["Wikipedia Link "][i].replace("https://en.wikipedia.org/wiki/","")+".html","r",encoding = "utf-8") as f:
        data = f.read()
        soup = BeautifulSoup(data)
        if soup.find("span",{'property':'dbp:date'}):
            train.loc[i,"Date"] = soup.find("span",{'property':'dbp:date'}).text
        if soup.find("span",{'property':'dbo:abstract','xml:lang':'en'}):
            train.loc[i,"Description"] = soup.find("span",{'property':'dbo:abstract','xml:lang':'en'}).text
        if soup.find("span",{'property':'geo:lat'}):
            train.loc[i,"Latitude"] = soup.find("span",{'property':'geo:lat'}).text
            train.loc[i,"Longitude"] = soup.find("span",{'property':'geo:long'}).text
        if soup.find("span",{'property':'dbp:latd'}):
            train.loc[i,"Latitude"] = soup.find("span",{'property':'dbp:latd'}).text
            train.loc[i,"Longitude"] = soup.find("span",{'property':'dbp:longd'}).text
    except OSError as e:
        pass
    finally:
        pass

### Get the Number of Dead and Number of Injured from DB pedia
for i in range(len(train["Disaster Name"])):
    try:
       with open(train["Wikipedia Link "][i].replace("https://en.wikipedia.org/wiki/","")+".html","r",encoding = "utf-8") as f:
        data = f.read()
        soup = BeautifulSoup(data)
        if soup.find("span",{'property':'dbp:casualties'}):
            train.loc[i,"Number of Dead"] = soup.find("span",{'property':'dbp:casualties'}).text
        if soup.find("span",{'property':'dbp:fatalities'}):
            train.loc[i,"Number of Dead"] = soup.find("span",{'property':'dbp:fatalities'}).text
        if soup.find("span",{'property':'dbp:reportedDeaths'}):
            train.loc[i,"Number of Dead"] = soup.find("span",{'property':'dbp:reportedDeaths'}).text
        if soup.find("span",{'property':'dbp:deaths'}):
            train.loc[i,"Number of Dead"] = soup.find("span",{'property':'dbp:deaths'}).text
        if soup.find("span",{'property':'dbp:injuries'}):
            train.loc[i,"Number of Injured"] = soup.find("span",{'property':'dbp:injuries'}).text
    except OSError as e:
        pass
    finally:
        pass

### Hitting wiki pages for those links which are not available in DBpedia
train["Description"].fillna("Not available",inplace = True)
for i in range(len(train["Disaster Name"])):
    if train.loc[i,"Description"] == "Not available":
        print(i)
        try:
           with urlopen(train["Wikipedia Link "][i]) as f:
            data = f.read()
            soup = BeautifulSoup(data)
            if soup.table.find("th",text = "Cost"):
                train.loc[i,"Economic Loss"] = soup.table.find("th",text = "Cost").parent.text.replace("Cost","")
        except OSError as e:
            pass
        except AttributeError as e:
            pass
        finally:
            pass
### Location

for i in range(len(train["Disaster Name"])):
    if np.isnan(float(train.loc[i,"Latitude"])):
        try:
           with open(train["Wikipedia Link "][i].replace("https://en.wikipedia.org/wiki/","")+".html","r",encoding = "utf-8") as f:
            data = f.read()
            soup = BeautifulSoup(data)
            if soup.find("span",{'property':'dbp:Location'}):
                train.loc[i,"Location"] = soup.find("span",{'property':'dbp:Location'}).text
        except OSError as e:
            pass
        finally:
            pass

### Get Economic loss For all docs
train.drop("Economic Loss",axis = 1,inplace = True)
for i in range(len(train["Disaster Name"])):
    print(i)
    try:
       with urlopen(train["Wikipedia Link "][i]) as f:
        data = f.read()
        soup = BeautifulSoup(data)
        if soup.table.find("th",text = "Cost"):
            train.loc[i,"Economic Loss"] = soup.table.find("th",text = "Cost").parent.text.replace("Cost","")
    except OSError as e:
        pass
    except AttributeError as e:
        pass
    finally:
        pass

### Using google API to get the country
from urllib.request import Request
from requests import get
import json
def getplace(lat, lon):
    url = "https://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (lat, lon)
    url += "&key=YOUR API KEY"
    v = get(url,verify = False)
    j = json.loads(v.content.decode('utf-8'))
    
    country = None    
    if not np.isnan(lat):     
        components = j['results'][0]['address_components']
        for c in components:
            if "country" in c['types']:
                country = c['long_name']
    return country

for i in tqdm(range(len(train["Disaster Name"]))):
    train.loc[i,"Country"] = getplace(float(train.loc[i,"Latitude"]),float(train.loc[i,"Longitude"]))


def get_date(desc):
    one = desc.str.extract(r'((?:\d{,2}\s)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:-|\.|\s|,)\s?\d{,2}[a-z]*(?:-|,|\s)?\s?\d{2,4})')
    # Get the dates in the form of numbers
    return one   
def get_year(desc):
    one = desc.str.extract(r'(\d{2,4})')
    # Get the dates in the form of numbers
    return one
train["New Date"] = get_date(train["Description"])
train["Year"] = get_year(train["Disaster Name"])



#import dateutil.parser as dparser
#train["Date1"] = train["Description"].apply(lambda x: dparser.parse(x,fuzzy=True))
train["New Date"] = train["New Date"].astype(str)
train["year in Date"] = train["New Date"].apply(lambda x: "yes" if re.search('[1-3][0-9]{3}',x) else "no")


for i in range(len(train["Disaster Name"])):
    if train.loc[i,"year in Date"] == "no":
        train.loc[i,"Final Date"] = str(train.loc[i,"New Date"]) + "," + str(train.loc[i,"Year"])
    else:
        train.loc[i,"Final Date"] = train.loc[i,"New Date"]

train["Final Date"].replace("nan,nan","",inplace = True)
train["Final Date"] = train["Final Date"].apply(lambda x: x.replace("nan,",""))
