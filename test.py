import pandas as pd
import nltk
import numpy as np
import ast

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

df = pd.read_csv('Airline_Short_Test.csv')

''''
df = pd.read_csv('P1-D1-Normalized.csv')

df = pd.read_csv('AirlineAmended.csv', dtype = 'float64', converters = {'text': str})  
'''

# REMOVE NaN VALUES AND EMPTY STRINGS:
df.dropna(inplace=True)

blanks = []  # start with an empty list

a = 'This was a good movie.'
print(sid.polarity_scores(a))

a = '@VirginAmerica seriously would pay $30 a flight for seats that didnt have this playing. its really the only bad thing about flying VA'
print(sid.polarity_scores(a))

for i,lb,rv in df.itertuples():  
    if type(rv)==str:            
        if rv.isspace():        
            blanks.append(i)     

df.drop(blanks, inplace=True)

df['temp'] = df.apply(lambda text : sid.polarity_scores(text))

tweets = df['text']
for f in tweets[:14641]:
    # print(f)
  print(sid.polarity_scores(f)['compound'])
  
'''
df['text'] = df['text'].apply(ast.literal_eval).str.decode("utf-8")
df.head()

# create compound as a separate column where all values greater than zeroes will be considered a positive review and all values less than zero would be considered as a negative review
df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])

df.head()

df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

df.head()

a = 'This was a good movie.'
print(sid.polarity_scores(a))

a = 'This was the best, most awesome movie EVER MADE!!!'
print(sid.polarity_scores(a))
'''
