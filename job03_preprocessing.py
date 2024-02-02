import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('./reviews_kinolights.csv')
df.info()

df_stopwords = pd.read_csv('./stopwords.csv')
stopwords = list(df_stopwords['stopword'])
stopwords = stopwords + ['영화', '감독', '연출', '배우', '연기', '작품', '관객', '장면', '각본', '개봉', '모르다']
okt = Okt()
cleaned_sentences = []
for review in df.reviews[:2]:
    review = re.sub('[^가-힣]', ' ', review)
    tokened_review = okt.pos(review, stem=True) #형태소로 나누는데 pos는 명사, 형용사 이런거까지 같이 리턴해준다. 리스트안에 튜플쌍으로 묶어서 리턴.
    print(tokened_review)
    df_token = pd.DataFrame(tokened_review, columns=['word', 'class']) #단어와 품사 컬럼으로 나눠주기
    df_token = df_token[(df_token['class']=='Noun') |           #품사가 명사, 형용사, 동사인 것만 남기겠다.
                        (df_token['class']=='Adjective') |
                        (df_token['class']=='Verb')]
    words = []
    for word in df_token.word:
        if 1 < len(word):
            if word not in stopwords:
                words.append(word)
    cleaned_sentence = ' '.join(words) #댓글자체가 1글자 이거나, 명사,형용사,동사가 아닌 품사만으로 이루어진경우(?). 아무것도 없으면 ' ' (띄어쓰기 하나)라도 들어온다.
    cleaned_sentences.append(cleaned_sentence)
df['reviews'] = cleaned_sentences
df.dropna(inplace=True) #한글자나 stopword로만 되있는 댓글은 아무것도 안남을 수 있는경우 있으니 드롭 사용.
df.to_csv('./cleaned_reviews.csv', index=False)

print(df.head())
df.info()

df = pd.read_csv('./cleaned_reviews.csv') #저장했다 다시 읽어들이면 공백만으로 되있던거는 nan값으로 들어온다.
df.dropna(inplace=True)
df.info()
df.to_csv('./cleaned_reviews2.csv', index=False)