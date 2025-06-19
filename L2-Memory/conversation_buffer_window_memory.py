from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

from lang_chain_project.utils import open_ai

memory = ConversationBufferWindowMemory(k = 1)
memory.save_context({"input": "Hi!"},{"output": "What's up?"})
memory.save_context({"input": "How are you?"}, {"output": "I am fine, thank you."})
print(f"memory buffer: {memory.load_memory_variables({})}")
print(f"=================================================================================================")

memory = ConversationBufferWindowMemory(k = 1)
chat = open_ai.NewLangChainChatOpenAI(model="deepseek-r1:1.5b")
conversation = ConversationChain(llm=chat, memory=memory, verbose=False)

conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1 + 1 ?")
conversation.predict(input="What is my name?")
print(f"memory buffer: {memory.load_memory_variables({})}")
print(f"=================================================================================================")
