import numpy as np
import pandas as pd

result = pd.read_csv('AirlineResult.csv')
provided = pd.read_csv('Airline_No_Text_Real.csv')

total_correct = 0
total_tweets = 0
confusion_matrix = np.zeros((3, 3))

for i in range(0,len(result)):
  r = result.iloc[i][4]
  p = provided.iloc[i][5]
  if(r == p):
    total_correct += 1
    if(r == 'positive'):
      confusion_matrix[0][0] += 1
    elif(r == 'negative'):
      confusion_matrix[2][2] += 1
    elif(r == 'neutral'):
      confusion_matrix[1][1] += 1
  elif(r == 'positive' and p == 'neutral'):
    confusion_matrix[0][1] += 1
  elif(r == 'positive' and p == 'negative'):
    confusion_matrix[0][2] += 1
  elif(r == 'neutral' and p == 'positive'):
    confusion_matrix[1][0] += 1
  elif(r == 'neutral' and p == 'negative'):
    confusion_matrix[1][2] += 1
  elif(r == 'negative' and p == 'positive'):
    confusion_matrix[2][0] += 1
  elif(r == 'negative' and p == 'neutral'):
    confusion_matrix[2][1] += 1
  else:
    print(r, p)
    # print(r=='neutral', p=='positive', r=='neutral' and p=='positive')
  total_tweets += 1

print('accuracy =', total_correct/total_tweets)
print('total =', total_tweets)
print(confusion_matrix)