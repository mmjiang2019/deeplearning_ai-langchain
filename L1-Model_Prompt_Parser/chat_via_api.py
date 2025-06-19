import openai

from lang_chain_project.utils import open_ai

client = open_ai.NewOpenAIClient()
models = open_ai.list_models(client=client)
print(f"models: {models}")
model = open_ai.get_llm_model_name()

prompt = "What is the capital of France?"

print(prompt)
response = open_ai.get_completion(client=client, prompt=prompt, model=model)
print(f"response: {response}")

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
style = """American English \
in a calm and respectful tone
"""
prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""
print(prompt)
response = open_ai.get_completion(client=client, prompt=prompt, model=model)
print(response)