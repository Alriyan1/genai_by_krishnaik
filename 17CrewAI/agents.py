from crewai import Agent,LLM
from tools import yt_tool

llm = LLM(
    model="groq/llama-3.2-90b-text-preview",
    temperature=0.7,
    api_key="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"
)

blog_researcher=Agent(
    role='blog researcher from youtube videos',
    goal='get the relevent video content for the topic{topic} from youtube channel',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI data science, machine learning and genai and provide suggestion"
    ),
    llm=llm,
    tools=[yt_tool],
    allow_delegation=True
)


blog_writer=Agent(
    role='blog writer',
    goal='Narrate compelling tech stories about the video {topic} from youtube channel',
    verbose=True,
    memory=True,
    backstory=(
        'with a flair for simplifying complex topics, you craft'
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    llm=llm,
    tools=[yt_tool],
    allow_delegation=False
)