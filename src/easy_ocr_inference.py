import easyocr
import cv2
import os
from os import path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np
import albumentations as A
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from torch.utils.data import DataLoader

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def preprocess(image):

    # image = cv2.resize(image, dsize=(640, 640))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    sharpened_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    image = sharpened_bgr
    return image

# def visualize(path):
    image = cv2.imread(path)
    image = preprocess(image)
    results = reader.readtext(image)

    for (bbox, text, prob) in results:
        top_left = tuple([int(val) for val in bbox[0]])
        bottom_right = tuple([int(val) for val in bbox[2]])

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 6)

    output = image
    output = cv2.resize(output, (1280, 960))
    cv2.imshow('Image with Bounding Boxes', output)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

def run_inference(path, reader : easyocr.Reader, preprocess = None):
    image = cv2.imread(path)
    if preprocess is not None:
        image = preprocess(image)
    angles = [0, 90, 180, 270]
    # angles = [0]
    results = []
    for angle in angles:
        image = rotate_image(image, angle)
        # result = reader.readtext(image)
        result = reader.readtext_batched(np.stack((image, image)))
        results.append(result)
    return results


        
reader = easyocr.Reader(['en'], gpu = True, detector= True)
# reader.recognize()
suffix = "test"
data_path = f'/home/loki/AmazonML/data/metadata/{suffix}.csv'
df = pd.read_csv(data_path)
out_df = pd.DataFrame(columns=['image_name', 'text_0', "text_90", "text_180", "text_270", "group_id", "entity_name", "entity_value"])

if suffix == "test":
    out_df = pd.DataFrame(columns=["index", 'image_name', 'text_0', "text_90", "text_180", "text_270", "group_id", "entity_name"])

start = 0
end = 60000
step = 500


for packet in tqdm(range(start, end, step)):
    for idx, data in df.iloc[start:start+step,:].iterrows():
        path = data['image_path']
        results = run_inference(path, reader, preprocess = preprocess)
        texts = []
        dict = {}
        if suffix == "test":
            dict["index"] = data['index']
        dict["image_name"] = path.split('/')[-1]
        dict['entity_name'] = data['entity_name']
        if suffix == "train":
            dict['entity_value'] = data['entity_value']
        dict['group_id'] = data['group_id']
        

        for result in results:
            print(len(result))
            text = ''
            for bb, word, prob in result:
                text += word + " "
            texts.append(text)
        for i, text in enumerate(texts):
            dict[f"text_{str(i*90)}"] = text
        temp_df = pd.DataFrame(dict, index = [idx])
        out_df = out_df._append(temp_df, ignore_index = True)
    out_df.to_csv(f'/home/loki/AmazonML/data/metadata/{suffix}_inference_{packet}_{packet+step}.csv', index = False)

     