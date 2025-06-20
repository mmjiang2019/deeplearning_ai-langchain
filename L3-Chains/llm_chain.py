from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from lang_chain_project.utils import data_loader as dl
from lang_chain_project.utils import open_ai

df = dl.load_csv_data()
print(f"data frame head: \n{df.head()}")
print(f"=================================================================================================")

chat = open_ai.NewLangChainChatOpenAI(model="openchat:7b", temperature=0.0)
prompt = ChatPromptTemplate.from_template("What is the best name to describe \
    a company that makes {product}?")

print(f"prompt: \n{prompt}")

chain = LLMChain(llm=chat, prompt=prompt)
product = "Queen Size Sheet Set"
response = chain.run(product)
print(f"response: \n{response}")
print(f"=================================================================================================")
