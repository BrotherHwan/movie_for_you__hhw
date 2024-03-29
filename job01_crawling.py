from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
import datetime

options = ChromeOptions()
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
options.add_argument('user_agent=' + user_agent)
options.add_argument('lang=ko_KR')
#options.add_argument('headless') #메모리에만 띄우고 유저에게 보이는 창은 안띄운다.
#options.add_argument('window-size=1920X1080') #사이즈를 정해줄 수 있다. 방은형 페이지의 경우 xpath가 바뀔수 있으니 창크기 고정해준다.

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

start_url = 'https://m.kinolights.com/discover/explore'
button_movie_tv_xpath = '//*[@id="contents"]/section/div[3]/div/div/div[3]/button'
button_movie_xpath = '//*[@id="contents"]/section/div[4]/div[2]/div[1]/div[3]/div[2]/div[2]/div/button[1]'
button_ok_xpath = '//*[@id="applyFilterButton"]'

driver.get(start_url)
time.sleep(0.5)

button_movie_tv = driver.find_element(By.XPATH, button_movie_tv_xpath)
driver.execute_script('arguments[0].click();', button_movie_tv) #자바스크립트로 되있어서 이런식으로 해줘야 한다...
time.sleep(1)
button_movie = driver.find_element(By.XPATH, button_movie_xpath)
driver.execute_script('arguments[0].click();', button_movie)
time.sleep(1.5)
button_ok = driver.find_element(By.XPATH, button_ok_xpath)
driver.execute_script('arguments[0].click();', button_ok)
time.sleep(1)

for i in range(25):
    driver.execute_script("window.scrollTo(0,document.documentElement.scrollHeight);")#스크롤 하기
    time.sleep(1)

list_review_url = []
movie_titles = []

for i in range(1, 1000):
    print(i)
    base = driver.find_element(By.XPATH, f'//*[@id="contents"]/div/div/div[3]/div[2]/div[{i}]/a').get_attribute("href")
    #속성값 가져올 때 get_attribute. class, atag등등
    #a 태그로 그 포스터를 클릭했을 경우의 주소 받기
    list_review_url.append(f'{base}/reviews') #받은 주소뒤에 /reviews붙여주면 댓글전체보기 창으로 이동
    title = driver.find_element(By.XPATH, f'//*[@id="contents"]/div/div/div[3]/div[2]/div[{i}]/div/div[1]').text
    movie_titles.append(title)

print(list_review_url[:5])
print(len(list_review_url))
print(movie_titles[:5])
print(len(movie_titles))


reviews = []
for idx, url in enumerate(list_review_url[450:500]):
    driver.get(url)
    time.sleep(0.5)
    review = ''
    for i in range(1, 31):
        review_title_xpath = '//*[@id="contents"]/div[2]/div[2]/div[{}]/div/div[3]/a[1]/div'.format(i)
        review_more_xpath = '//*[@id="contents"]/div[2]/div[2]/div[{}]/div/div[3]/div/button'.format(i)
        try:
            review_more = driver.find_element(By.XPATH, review_more_xpath)
            driver.execute_script('arguments[0].click();', review_more)
            time.sleep(1)
            review_xpath = '//*[@id="contents"]/div[2]/div[1]/div/section[2]/div/div' #리뷰눌러서 들어갔을때 타이틀은 이주소 뒤에 h3, 내용은 div/p 붙어있는데
                                                                                      #어차피 이거 아래 있으니까 이걸로 다 긁어오기
            review = review + ' ' + driver.find_element(By.XPATH, review_xpath).text

            driver.back()
            time.sleep(1)
        except NoSuchElementException as e:  #엘레멘츠가 없을 때. 즉, 더보기가 없을 때.
            print('더보기', e)
            try: #댓글이 10개인데 만약 11번째거 가져오려하면 에러
                review = review + ' ' + driver.find_element(By.XPATH, review_title_xpath).text
            except:
                print('review title error')
        except StaleElementReferenceException as e: #페이지가 안열렸을 때
            print('stale', e)
        except :
            print('error')

    print(review)
    reviews.append(review)
print(reviews[:5])
print(len(reviews))

df = pd.DataFrame({'titles':movie_titles[450:500], 'reviews':reviews})
today = datetime.datetime.now().strftime('%Y%m%d')
df.to_csv('./crawling_data/reviews_.csv', index=False)

