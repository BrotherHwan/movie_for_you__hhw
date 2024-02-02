#단어를 벡터로 바꾸겠다
#tfidf는 계산값을 그냥 좌표로 보겠다고 한것. 여기서는 의미공간상의 좌표를 정해준다.
#비슷한 값을 가진다는게 사전적으로 유의어를 말하는건 아니다. 유사단어, 연상단어를 얘기하는것.
#의미적으로 유사한걸 찾기위해 AI사용

import pandas as pd
from gensim.models import Word2Vec

df_review = pd.read_csv('./cleaned_one_review.csv')
df_review.info()

reviews = list(df_review['reviews']) #모델에 리스트형태로 줘야함
print(reviews[0])

tokens = []
for sentence in reviews:
    token = sentence.split()
    tokens.append(token)
print(tokens[0])

embedding_model = Word2Vec(tokens, vector_size=100, window=4, min_count=20, workers=4, epochs=100, sg=1)
#임베딩레이어를 하나씩 따로 쓸 수 있는게 Word2Vec
#tokens 자체는 칠천몇백 차원인데
#vector_size로 차원을 100차원까지 줄이겠다.
#window는 kernel_size처럼 4개까지 윈도우로 씌워가면서 움직이겠다는것
#min_count=20은 20번이상 출연하는 단어만 의미 학습하겠다는 뜻.
#workers는 학습할 때 cpu 몇개쓰겠다
#sg=1은 학습할 때 어떤 알고리즘 쓰겠다는것
embedding_model.save('./models/word2vec_movie_review.model')
print(list(embedding_model.wv.index_to_key)) #Word2Vec에 준
print(len(embedding_model.wv.index_to_key)) #학습한 단어의 개수 7440개에 대한 단어의 의미만 알고있다.