import pandas as pd
import os

path = '/home/loki/AmazonML/data/metadata/train.csv'

df = pd.read_csv(path)
df['image_path'] = df['image_path'].apply(lambda x: '/'.join(x.split('/')[-2:]))
# print(df.head())

df.to_csv(path, index=False)