#영화하나의 리뷰를 하나의 텍스트라 하자. 전체 영화의 리뷰를 document라고 하자.
#TF-IDF(Term Frequency - Inverse Document Frequency)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

df_reviews = pd.read_csv('./cleaned_one_review.csv')
df_reviews.info()

Tfidf = TfidfVectorizer(sublinear_tf=True) #좌표화.
Tfidf_matrix = Tfidf.fit_transform(df_reviews['reviews']) #모든 단어의 TF-IDF값이 각각의 벡터위치라 볼 수 있다.
print(Tfidf_matrix.shape)
#두 벡터가 이루는 각의 cos값을 보고 방향이 같으면(유사하면)(0도)1, 반대방향이면(상반되면)(180도) -1, 수직방향이면(관련없으면)(90도) 0

with open('./models/tfidf.pickle', 'wb') as f:
    pickle.dump(Tfidf, f)

mmwrite('./models/Tfidf_movie_review.mtx', Tfidf_matrix) #행렬 저장
