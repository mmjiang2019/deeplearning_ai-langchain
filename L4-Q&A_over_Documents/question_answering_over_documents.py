from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings

from IPython.display import display, Markdown

from lang_chain_project.utils import data_loader

loader = data_loader.load_csv_data_as_doc_loader()

embeddings = OpenAIEmbeddings(
    base_url="http://127.0.0.1:11434/v1/",
    api_key="ollama",
    model="qwen3:1.7b",
)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    # embedding=embeddings,
).from_loaders([loader])

query = "Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

llm_replacement_model = OpenAI(
    model="qwen3:1.7b",
    temperature=0.0,
)

response = index.query(
    question=query,
    llm=llm_replacement_model,
)

display(Markdown(response))