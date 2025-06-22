import os

import pandas as pd

from pandas import DataFrame
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader

from lang_chain_project.config import data

def load_csv_data_as_pd(file_name: str = data.CSV_DATA_PATH) -> DataFrame:
    csv_path = os.path.join(os.getcwd(), file_name)
    if os.name == 'nt':
        csv_path = csv_path.replace('/', '\\')
    print(f"csv data path: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def load_csv_data_as_doc_loader(file_name: str = data.CSV_DOC_PATH) -> list[Document]:
    csv_path = os.path.join(os.getcwd(), file_name)
    if os.name == 'nt':
        csv_path = csv_path.replace('/', '\\')
    print(f"csv doc path: {csv_path}")
    loader = CSVLoader(file_path=csv_path)
    return loader