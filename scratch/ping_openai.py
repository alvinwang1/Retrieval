import asyncio
import os
from openai import AsyncOpenAI

async def ping_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        return

    client = AsyncOpenAI(api_key=api_key)
    print("Connecting to OpenAI...")
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        print(f"✅ Success! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ API Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(ping_openai())
