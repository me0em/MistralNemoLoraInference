from openai import OpenAI

prompt = "the king is dead long live the king"

client = OpenAI(
    base_url="http://127.0.0.1:8001",
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
