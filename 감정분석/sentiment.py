from main import model, predict_sentence
import csv
_f = open('./before.csv', 'r')
wr = csv.writer(open('./before_sentiment.csv', 'w', newline=''))

def returnLabelByScore(score):
    if score >= 0.5:
        return '{0}% 긍정'.format(round(score, 2))
    else:
        return '{0}% 부정'.format(round(score, 2))
        
for line in csv.reader(_f):
    print(line)
    wr.writerow([line[0], returnLabelByScore(predict_sentence(line[0], model))])

    

    