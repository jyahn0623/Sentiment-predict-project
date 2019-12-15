from konlpy.tag import Okt
import os
import nltk # 자연어 처리 패키지
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model


def read_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = [ l.split('\t') for l in f.read().splitlines() ]
        data = data[1:]
    return data

def normalization(sentence):
    return [ '/'.join(t) for t in okt.pos(sentence, norm=True, stem=True) ]

def preNormalization(train_datas, test_datas):
    import json
    if os.path.isfile('train_docs.json'):
        with open('train_docs.json', encoding='utf-8') as f:
            train_docs = json.load(f)
        with open('test_docs.json', encoding='utf-8') as f:
            test_docs = json.load(f)
    else:
        train_docs = [ (normalization(r[1]), r[2]) for r in train_datas ]
        test_docs = [ (normalization(r[1]), r[2]) for r in test_datas ]

        json.dump(train_docs, open('train_docs.json', 'w', encoding='utf-8'), ensure_ascii=False, indent='\t')
        json.dump(test_docs, open('test_docs.json', 'w', encoding='utf-8'), ensure_ascii=False, indent='\t')

    return train_docs, test_docs

def term_frequency(doc):
    return [ doc.count(w) for w in selected_words]

def predict_sentence(sentence, model):
    token = normalization(sentence)
    tf = term_frequency(token)
    
    # 데이터의 형태를 맞춰주기 위해서 축 확대
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    print(score)
    return score

def getModel():
    try:
        return load_model('./sentiment_model.h5')
    except:
        # 모델 구성
        train_x = [ term_frequency(d) for d, _ in train_docs ]
        test_x = [ term_frequency(d) for d, _ in test_docs ]
        train_y = [c for _, c in train_docs]
        test_y = [c for _, c in test_docs]

        x_train = np.asarray(train_x).astype('float32')
        x_test = np.asarray(test_x).astype('float32')

        y_train = np.asarray(train_y).astype('float32')
        y_test = np.asarray(test_y).astype('float32')

        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(500,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))


        # # 모델 학습 과정
        model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                    loss=losses.binary_crossentropy,
                    metrics=[metrics.binary_accuracy])

        # # 모델 학습
        model.fit(x_train, y_train, epochs=15, batch_size=512)
        results = model.evaluate(x_test, y_test)

        # 모델 저장
        model.save('sentiment_model.h5')
        return model

okt = Okt()
train_datas = read_data('../nsmc/ratings_train.txt')
test_datas = read_data('../nsmc/ratings_test.txt')


train_docs, test_docs = preNormalization(train_datas, test_datas)
tokens = [ t for d in train_docs for t in d[0] ]

# 전처리를 위해 nltk 사용, Text 클래스는 문서를 편리하게 탐색할 수 있는 기능 제공해 줌.
text = nltk.Text(tokens, name='NMSC')

# 한글 위해서
font_fname = 'C:/Windows/Fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

# 토큰 벡터화 
# CountVectorization 사용 ==> 문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW 인코딩한 벡터를 만드는 역할
selected_words = [ f[0] for f in text.vocab().most_common(500) ]

model = getModel()

