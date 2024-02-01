import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec

def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1])) #인덱스 같이 가져오기 위해. 안하면 밑에줄에서 소팅할 때 인덱스 깨지니까.
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True) #cos값이 큰값부터 정렬
    simScore = simScore[:11] #앞에서부터 11개. 10개추천할건데 첫번째는 자기 자신이기 때문에 그거 빼고 추천하려고 11개 함.
    movieIdx = [i[0] for i in simScore]
    recmovieList = df_reviews.iloc[movieIdx, 0] #0번컬럼은 영화제목
    return recmovieList[1:11]

df_reviews = pd.read_csv('./cleaned_one_review.csv')
Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

ref_idx = 21     #확인해보고 싶은 영화 인덱스

print(df_reviews.iloc[ref_idx, 0])
cosine_sim = linear_kernel(Tfidf_matrix[ref_idx], Tfidf_matrix) #11번 인덱스영화와 전체영화와의 cos값을 구해줌.
print(cosine_sim[0])
print(len(cosine_sim))
reccommendation = getRecommendation(cosine_sim)
print(reccommendation)