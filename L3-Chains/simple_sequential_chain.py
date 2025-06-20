from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from lang_chain_project.utils import data_loader as dl
from lang_chain_project.utils import open_ai

df = dl.load_csv_data()
print(f"data frame head: \n{df.head()}")
print(f"=================================================================================================")

chat = open_ai.NewLangChainChatOpenAI(model="openchat:7b", temperature=0.0)

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=chat, prompt=first_prompt)

# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=chat, prompt=second_prompt)

# overall chain: chain_one -> chain_two
overall_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

product = "Queen Size Sheet Set"
response = overall_chain.run(product)
print(f"response: \n{response}")
print(f"=================================================================================================")
