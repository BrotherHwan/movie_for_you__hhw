import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
import matplotlib as mpl

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus'] = False
rc('font', family=font_name)

embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
key_word = '스파이더맨' #모델을 불러다가 키워드 주기

sim_word = embedding_model.wv.most_similar(key_word, topn=10) #키워드 근처에 가장 가까운 단어 30개보기
print(sim_word)

vectors = []
labels = []

for label, _ in sim_word: #뒤에있는 유사도 값은 쓰지 않을거라 -로 받은것
    labels.append(label)
    vectors.append(embedding_model.wv[label]) #100차원인 label을 준다
print(vectors[0]) #하나만 찍어보기
print(len(vectors[0]))

#100차원 그림을 그릴 수 없으니 2차원으로 차원 축소
df_vectors = pd.DataFrame(vectors)
print(df_vectors.head())

tsne_model = TSNE(perplexity=9, n_components=2, init='pca', n_iter=2500)
#TSNE는차원축소 모델. 근데 쓴거는 pca사용. n_components=2차원으로 축소하겠다
new_value = tsne_model.fit_transform(df_vectors) #2차원 좌표를 만들어 준다.
df_xy = pd.DataFrame({'words':labels, 'x':new_value[:, 0], 'y':new_value[:,1]})
df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0) #keyword를 좌표 0,0에 그리겠다

print(df_xy)
print(df_xy.shape)

plt.figure(figsize=(8, 8))
plt.scatter(0, 0, s=1500, marker='*') #원점에 점하나 그리는 스캐터

for i in range(len(df_xy)):
    a = df_xy.loc[[i, 10]]
    plt.plot(a.x, a.y, '-D', linewidth=1) #'-D'는 선과 다이아몬드로 그리겠다는 뜻.
    plt.annotate(df_xy.words[i], xytext=(1, 1), xy=(df_xy.x[i], df_xy.y[i]), textcoords='offset points', ha='right', va='bottom')

plt.show()