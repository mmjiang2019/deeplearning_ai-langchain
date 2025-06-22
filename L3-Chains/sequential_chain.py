from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from lang_chain_project.utils import data_loader as dl
from lang_chain_project.utils import open_ai

df = dl.load_csv_data_as_pd()
print(f"data frame head: \n{df.head()}")
print(f"=================================================================================================")

chat = open_ai.NewLangChainChatOpenAI(model="openchat:7b", temperature=0.0)

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=chat, prompt=first_prompt, output_key="English_Review")

# prmpt template 2: make a summary for the english review output of the first prompt
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=chat, prompt=second_prompt, output_key="Summary")

# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template("What language is the following review:\n\n{Review}")
chain_three = LLMChain(llm=chat, prompt=third_prompt, output_key="Language")

# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {Summary}\n\nLanguage: {Language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=chat, prompt=fourth_prompt,
                      output_key="FollowupMessage"
                     )

# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "Summary", "FollowupMessage"],
    verbose=True
)

review = df.Review[5]
print(f"review: \n{review}")
response = overall_chain(review)
print(f"response: \n{response}")
print(f"=================================================================================================")
