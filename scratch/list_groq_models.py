import os
import asyncio
from openai import AsyncOpenAI

async def list_groq_models():
    from dotenv import load_dotenv
    load_dotenv("../.env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found in environment.")
        return
    
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    try:
        models = await client.models.list()
        print("\nAvailable Groq Models:")
        print("-" * 30)
        for m in models.data:
            print(f"- {m.id}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(list_groq_models())
