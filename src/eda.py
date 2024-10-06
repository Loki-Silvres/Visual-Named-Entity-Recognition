import pandas as pd

suffix = 'test'

df = pd.read_csv(f"/home/loki/AmazonML/data/metadata/{suffix}.csv")

def perform_eda(df: pd.DataFrame) -> None:
    print(df.head())
    print("Columns: ", df.columns.to_list())
    print("No. of unique entities: ", len(df['entity_name'].unique()))
    print("Unique values of key: ", df['entity_name'].unique())
    print("No. of unique groups: ", len(df['group_id'].unique()))
    print("No. of images: ", len(df['image_path']))
    print("No. of unique images: ", len(df['image_path'].unique()))
    

def main() -> None:
    perform_eda(df)

if __name__ == "__main__":
    main()