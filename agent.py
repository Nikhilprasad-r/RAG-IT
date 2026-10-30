from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

def lookup_order(order_id: str) -> str:
    """Look up an order by id."""
    return f"Order {order_id} is shipped."

agent = create_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[lookup_order],
    system_prompt="You are a support agent. Use tools when you need facts.",
    checkpointer=InMemorySaver(),
)