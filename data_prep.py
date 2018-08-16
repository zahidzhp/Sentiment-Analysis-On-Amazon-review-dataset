import pandas as pd
from bs4 import BeautifulSoup
import string

csv_file=pd.read_csv('amazon_review_updated.csv')

def lebelling(file):
    lebel=[]
    rows=len(file['Score'])
    for i in range(rows):
        val = int(file.at[i, 'Score'])
        if(val>3):
            value = 'Positive'
        elif (val==3):
            value = 'Neutral'
        else:
            value = 'Negative'
        lebel.insert(i,value)
    return lebel

lebels=lebelling(csv_file)
csv_file['Sentiment']=lebels
csv_file=csv_file[['Text','Sentiment']]
print(csv_file.head())

csv_file['Text'] = csv_file['Text'].apply(lambda x: BeautifulSoup(x,'lxml').get_text())
print(csv_file.head())
csv_file['Text'] = csv_file['Text'].apply(lambda x: x.translate(string.punctuation))
print(csv_file.head())
csv_file['Text'] = csv_file['Text'].apply(lambda x: x.translate(string.digits))
print(csv_file.head())
csv_file['Text'] = csv_file['Text'].apply(lambda x: x.lower())
import re
for i in range(len(csv_file['Text'])):
    csv_file.at[i,'Text'] = re.sub(r'\d+', '', csv_file.at[i,'Text'])
print(csv_file.head())


csv_file.to_csv('amazon_review_updated.csv')

