from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import DocArrayInMemorySearch

from IPython.display import display, Markdown

from lang_chain_project.utils import data_loader
from lang_chain_project.utils import open_ai

loader = data_loader.load_csv_data_as_doc_loader()
docs = loader.load()
print(f"docs loaded: {len(docs)}")
print(f"first doc:\n{docs[0]}")
print(f"=================================================================================================")

embeddings = open_ai.NewEmbeddings(model="deepseek-r1:1.5b")

index_creator = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
)
index = index_creator.from_loaders([loader])

query = "Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

chat = open_ai.NewLangChainChatOpenAI(model="deepseek-r1:1.5b")
# Doesn't work with the OpenAI model, comment
# llm_replacement_model = open_ai.NewOpenAI(model="deepseek-r1:1.5b")

response = index.query(
    question=query,
    llm=chat,
)
print(f"response:\n{response}")

display(Markdown(response))
print(f"=================================================================================================")

embed = embeddings.embed_query("Hi my name is Harrison")
print(f"embed len: {len(embed)}")
print(f"embed[:5]: {embed[:5]}")

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
print(f"docs len: {len(docs)}")
print(f"first doc:\n{docs[0]}")

retriever = db.as_retriever()

qdocs = "".join([docs[i].page_content for i in range(len(docs))])
response = chat.invoke(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")
display(Markdown(response))

qa_stuff = RetrievalQA.from_chain_type(
    llm=chat, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.invoke(query)
display(Markdown(response))
response = index.query(query, llm=chat)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])