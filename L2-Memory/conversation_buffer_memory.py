import os
import warnings

from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from lang_chain_project.utils import open_ai

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv()) # read local .env file

chat = open_ai.NewLangChainChatOpenAI(model="deepseek-r1:1.5b", temperature=0.0)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1 + 1 ?")
conversation.predict(input="What is my name?")

# print(f"memory buffer: {memory.buffer}")

print(f"memory buffer: {memory.load_memory_variables({})}")
print(f"=================================================================================================")
memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})
# print(f"memory buffer: {memory.buffer}")
print(f"memory buffer: {memory.load_memory_variables({})}")
print(f"=================================================================================================")
memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})
print(f"memory buffer: {memory.load_memory_variables({})}")
print(f"=================================================================================================")
