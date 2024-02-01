import pandas as pd
import glob

data_path = glob.glob('./crawling_data/*')
print(data_path)

df = pd.DataFrame()

for path in data_path:
    df_temp = pd.read_csv(path)
    df_temp.columns = ['titles', 'reviews'] #title이라 되있는 컬럼명이 있어서 전부 titles로 통일.
    df_temp.dropna(inplace=True) #어떤 컬럼이던 nan값있으면 그 행을 드랍
    df = pd.concat([df, df_temp], ignore_index=True)

df.drop_duplicates(inplace=True)
df.info()
df.to_csv('./reviews_kinolights.csv', index=False)
