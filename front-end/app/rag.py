import os
from openai import OpenAI

client = OpenAI()

def ask_ai(question: str):

    context = ""

    for file in os.listdir("knowledge"):
        with open(f"knowledge/{file}", "r", encoding="utf-8") as f:
            context += f.read() + "\n"

    prompt = f"""
You are a cardiology AI assistant.

Use the following medical knowledge to answer.

Knowledge:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output[0].content[0].text