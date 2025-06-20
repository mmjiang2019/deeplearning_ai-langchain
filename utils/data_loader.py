import os

import pandas as pd

from pandas import DataFrame

from lang_chain_project.config import data

def load_csv_data(file_name: str = data.CSV_DATA_PATH) -> DataFrame:
    csv_path = os.path.join(os.getcwd(), data.CSV_DATA_PATH)
    print(f"csv data path: {csv_path}")
    df = pd.read_csv(csv_path)
    return df