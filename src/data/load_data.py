import pandas as pd

def load_raw_excel(path):
    df = pd.read_excel(path)
    return df

def save_csv(df, csv_path):
    df.to_csv(csv_path, index=False)