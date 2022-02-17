import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download(["stopwords", "names","punkt"])
# nltk.download('vader_lexicon')
stopwords = nltk.corpus.stopwords.words("english")
more_stopwords = ['#','@','.','!','i',':','?','&','-','\'','http','n\'t',';','(',')','\'s','\'m',',','...','amp','flight','you','us','fly','get','one',]
for a in more_stopwords:
  stopwords.append(a)

df = pd.read_csv('AirlineAmendedPlusName.csv')

airline_gps = df.groupby(by=['airline'])

va = airline_gps.get_group('Virgin America')
un = airline_gps.get_group('United')
sw = airline_gps.get_group('Southwest')
dt = airline_gps.get_group('Delta')
ua = airline_gps.get_group('US Airways')
am = airline_gps.get_group('American')

airlines = [va, un, sw, dt, ua, am]

def get_distribution(airline):
  a_text = ' '.join(airline['text'].tolist())
  a_words = nltk.tokenize.word_tokenize(a_text)
  a_dist = nltk.FreqDist(w.lower() for w in a_words if w not in stopwords)
  return a_dist

def display_keywords(top_n):
  for n in top_n:
    word, count = n
    print(word, count)

# va_text = ' '.join(va['text'].tolist())
# va_words = nltk.tokenize.word_tokenize(va_text)
# va_dist = nltk.FreqDist(w.lower() for w in va_words if w not in stopwords)
for a in airlines:
  dist = get_distribution(a)
  print('----------')
  display_keywords(dist.most_common(22))
