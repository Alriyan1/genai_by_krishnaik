#from openai import OpenAI

from langchain_nvidia_ai_endpoints import ChatNVIDIA

client = ChatNVIDIA(
  model="meta/llama-3.3-70b-instruct",
  api_key="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", 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

for chunk in client.stream([{"role":"user","content":"which algorithm mediapipe use"}]): 
  print(chunk.content, end="")

  
# client = OpenAI(
#   base_url = "https://integrate.api.nvidia.com/v1",
#   api_key = "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"
# )

# completion = client.chat.completions.create(
#   model="meta/llama-3.3-70b-instruct",
#   messages=[{"role":"user","content":"provide me an article on machine learning"}],
#   temperature=0.2,
#   top_p=0.7,
#   max_tokens=1024,
#   stream=True
# )

# for chunk in completion:
#   if chunk.choices[0].delta.content is not None:
#     print(chunk.choices[0].delta.content, end="")

