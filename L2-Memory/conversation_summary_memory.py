from langchain.memory import ConversationSummaryMemory

from langchain.chains import ConversationChain

from lang_chain_project.utils import open_ai

chat = open_ai.NewLangChainChatOpenAI(model="deepseek-r1:1.5b")
# conversation summary memory do not need max token limit parameter
memory = ConversationSummaryMemory(llm=chat)

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

print(f"memory buffer: {memory.load_memory_variables({})}")
print(f"=================================================================================================")

conversation = ConversationChain(llm=chat, memory=memory, verbose=True)

conversation.predict(input="What would be a good demo to show?")

print(f"memory buffer: {memory.load_memory_variables({})}")
print(f"=================================================================================================")
