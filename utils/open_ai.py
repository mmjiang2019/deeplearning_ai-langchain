import os
import datetime

import openai

from typing import List
from dotenv import load_dotenv, find_dotenv
from lang_chain_project.config import open_ai as open_ai_config

def NewOpenAIClient():
    _ = load_dotenv(find_dotenv()) # read local .env file.
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(
        base_url=open_ai_config.BASE_URL,
        api_key=open_ai_config.API_KEY,  # this is also the default, it can be omitted
        )
    return client

def get_llm_model_name() -> str:
    # Get current date
    current_date = datetime.datetime.now().date()

    # Define the date after which the model should be set to "gpt-3.5-turbo"
    target_date = datetime.date(2024, 6, 12)

    # Set the model variable based on the current date
    model = "deepseek-r1:1.5b"
    if current_date <= target_date:
        model = "qwen3:1.7b"
    
    return model

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