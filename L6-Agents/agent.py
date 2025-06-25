import langchain

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits.load_tools import load_tools
# from langchain.python import PythonREPL

from langchain.agents import tool
from datetime import date

from lang_chain_project.utils import open_ai

# Note: this is a simple example, and is not meant to be used in production.
# some built-in tools maybe not work as expected, due to langchain library has been updated.
# TODO: fix in the future

# Built-in LangChain tools
llm_model = "deepseek-r1:1.5b"
# llm_model = "openchat:7b"
llm = open_ai.NewLangChainChatOpenAI(model=llm_model)
tools = load_tools(["llm-math","wikipedia"], llm=llm)
agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

result = agent.invoke("What is the 25% of 300?")
print(f"result: {result}")
print(f"=================================================================================================")

# Wikipedia example
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent.invoke(question)
print(f"result: {result}")
print(f"=================================================================================================")


# Python agent
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)

customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]

result = agent.invoke(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
print(f"result: {result}")
print(f"=================================================================================================")

langchain.debug=True
print(f"set langchain debug to true")
result = agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
print(f"result: {result}")
langchain.debug=False
print(f"set langchain debug to false")
print(f"=================================================================================================")

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())
agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
try:
    result = agent.invoke("whats the date today?") 
    print(f"result: {result}")
    print(f"=================================================================================================")
except: 
    print("exception on external access")