from os import path 
import pandas as pd 

suffix = 'test'
path = f"/home/loki/AmazonML/data/metadata/{suffix}_inference.csv"

start = 0
end = 60000
step = 500
if suffix == 'train':
    out_df = pd.DataFrame(columns=['image_name', 'text_0', "text_90", "text_180", "text_270", "group_id", "entity_name", "entity_value"])
elif suffix == 'test':
    out_df = pd.DataFrame(columns=["index", 'image_name', 'text_0', "text_90", "text_180", "text_270", "group_id", "entity_name"])
else:
    raise NotImplementedError

for packet in range(start, end, step):
    packet_path = path.replace('.csv', f'_{packet}_{packet+step}.csv')
    df = pd.read_csv(packet_path)
    out_df = pd.concat([out_df, df])

print(out_df.head())
final_path = path.replace('.csv', f'_{start}_{end}.csv')
out_df.to_csv(final_path, index=False)
