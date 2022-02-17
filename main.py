import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download(["stopwords", "names"])
nltk.download('vader_lexicon')
stopwords = nltk.corpus.stopwords.words("english")
qwords = [
    'who', 'what', 'when', 'where', 'why', 'how', 'is', 'can', 'does', 'do'
]

df = pd.read_csv('AirlineAmended2.csv')

sia = SentimentIntensityAnalyzer()

unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])


def remove_unwanted(text):
	text_words = text.split(' ')
	text_words = [t.lower() for t in text_words]
	for s in stopwords:
		# print(s, text_words)
		if (s in text_words):
			text_words.remove(s)
	return ' '.join(text_words)


tweets = df['text']

score_col = []
sentiment_col = []
type_col = []
airline_col = []

for f in tweets:
	f = remove_unwanted(f)
	# print(f)
	output = sia.polarity_scores(f)['compound']
	# print(output)
	score_col.append(output)

	if (output > 0.5):
		sentiment = 'positive'
	elif (output < -0.5):
		sentiment = 'negative'
	else:
		sentiment = 'neutral'

	sentiment_col.append(sentiment)

	if ('?' in f) and (any(qwords in f for qwords in qwords)):
		senttype = 'question'
	else:
		senttype = ''
	type_col.append(senttype)

	airline = f.split()[0]
	airline_col.append(airline)

df['Compound'] = score_col
df['Sentiment'] = sentiment_col
df['Airplane'] = airline_col
df['Question Tag'] = type_col

df.to_csv('AirlineResult.csv')
