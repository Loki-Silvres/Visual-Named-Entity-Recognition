import os.path as osp
import pandas as pd
from tqdm import tqdm

suffix = 'test'
df = pd.read_csv(f"/home/loki/AmazonML/data/student_resource 3/dataset/{suffix}.csv")
save_path = f"/home/loki/AmazonML/data/metadata/{suffix}.csv"
image_folder = f'/home/loki/AmazonML/data/images/{suffix}'

print(f"Initial size of {suffix} dataframe: {len(df)}")

df['image_link'] = df['image_link'].apply(lambda x: osp.join(image_folder, x.split('/')[-1]))
df = df.rename(columns={'image_link': 'image_path'})

for i in tqdm(range(len(df))):
    if not osp.exists(df['image_path'][i]):
        df = df.drop(i)  # Drop the row if the image does not exist

print(f"Final size of {suffix} dataframe: {len(df)}")
df.to_csv(save_path, index=False)