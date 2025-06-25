import os
import datetime

import openai
import tiktoken

# from typing import List, Tuple
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI

from langchain_deepseek import ChatDeepSeek

from lang_chain_project.config import open_ai as open_ai_config

def NewOpenAIClient():
    _ = load_dotenv(find_dotenv()) # read local .env file.
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(
        base_url=open_ai_config.BASE_URL,
        api_key=open_ai_config.API_KEY,  # this is also the default, it can be omitted
        )
    return client

def NewLangChainChatOpenAI(model: str, temperature: float = 0.0):
     return ChatOpenAI(
         base_url=get_base_url(),
         api_key=get_api_key(),
         temperature=temperature, model=model)

# useless
def NewLangChainDeepSeekChat(model: str, temperature: float = 0.0):
     return ChatDeepSeek(
         base_url=get_base_url(),
         api_key=get_api_key(),
         temperature=temperature, model=model)

def NewEmbeddings(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=get_base_url(),
        api_key=get_api_key(),
        model=model,
        check_embedding_ctx_length=False # check_embedding_ctx_length must be set to False for local testing, otherwise it will fail with a 400 error.
    )

def NewOpenAI(model: str, temperature: float = 0.0) -> OpenAI:
    return OpenAI(
        base_url=get_base_url(),
        api_key=get_api_key(),
        model=model,
        temperature=temperature,
    )
def list_models(client: openai.OpenAI):
    return client.models.list()

def get_base_url() -> str:
    return open_ai_config.BASE_URL

def get_api_key() -> str:
    return open_ai_config.API_KEY

def get_completion(client: openai.OpenAI, prompt: str, model: str):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content