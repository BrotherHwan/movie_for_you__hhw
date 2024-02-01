import pandas as pd
from wordcloud import WordCloud
import collections
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = './malgun.ttf' #여러가지 폰트가 들어있는 파일. #구글크롬에 datasets에서 다운
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family='NanumBarunGothic') #이런이름의 폰트 쓰겠다.

df = pd.read_csv('./cleaned_one_review.csv') #슬랙에서 교수님이 보내주신거 다운. 다음영화 크롤링한것.
words = df.iloc[1829, 1].split() #1829번영화의 reviews를 띄어쓰기 기준으로 잘라서 리스트 만들어 주기.
print(words)

worddict =collections.Counter(words) #words안의 유니크값값의 출연빈도 알려줌
worddict = dict(worddict) #그걸 딕셔너리 형태로 변형
print(worddict)

wordcloud_img = WordCloud(background_color='white', max_words=2000, font_path=font_path).generate_from_frequencies(worddict)
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')
plt.show()