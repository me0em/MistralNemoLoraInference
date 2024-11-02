from openai import OpenAI

prompt = "The king is dead. Long live to who?"

client = OpenAI(
    base_url="http://127.0.0.1:8005",
    api_key="token-abc123",
)


completion = client.chat.completions.create(
  model="",
  messages=[
    {"role": "user", "content": prompt}
  ],
  temperature=0
)

print("==============")
print(f"Assistant response: {completion.to_dict()}")
