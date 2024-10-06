import os
import os.path as osp
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

suffix = 'train'
df = pd.read_csv(f"/home/loki/AmazonML/data/student_resource 3/dataset/{suffix}.csv")
save_folder = f"/home/loki/AmazonML/data/images/{suffix}"

links = df['image_link'].tolist()

def download_image(image_url : str) -> bool:
    save_name = image_url.split('/')[-1]
    save_path = osp.join(save_folder, save_name)
    success_flag = True
    if not osp.exists(save_path):
        try:
            response = requests.get(image_url)
            response.raise_for_status()  

            img = Image.open(BytesIO(response.content))
            img.save(save_path)
            print(f"Image successfully downloaded and saved at {save_path}")
        except Exception as e:
            print(f"Error downloading the image {image_url.split('/')[-1]}: {e}")
            success_flag = False
    return success_flag

def main() -> None:
    with mp.Pool(os.cpu_count()//2) as P:
        results = P.map(download_image, links)
        print(f"{len(results)} images downloaded out of {len(links)}")

if __name__ == "__main__":
    main()