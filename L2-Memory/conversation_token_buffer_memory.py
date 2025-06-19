from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain

from lang_chain_project.utils import open_ai

chat = open_ai.NewLangChainChatOpenAI(model="qwen3:1.7b")
memory = ConversationTokenBufferMemory(
    llm=chat,
    max_token_limit=50,
    verbose=False
    )

memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

print(f"memory buffer: {memory.load_memory_variables({})}")
print(f"=================================================================================================")
