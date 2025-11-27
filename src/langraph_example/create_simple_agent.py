from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

checkpointer = InMemorySaver()

llm = ChatOllama(model="qwen3:14b", temperature=0)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model=llm, tools=[get_weather], prompt="You are a helpful assistant", checkpointer=checkpointer
)

# # Run the agent
# ai_msg = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
# print(ai_msg)

# Run the agent
config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]}, config)
ny_response = agent.invoke({"messages": [{"role": "user", "content": "what about new york?"}]}, config)

print(sf_response)
print(ny_response)
