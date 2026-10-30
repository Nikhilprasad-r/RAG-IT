from fastapi import FastAPI
from pydantic import BaseModel

from agent import agent

app = FastAPI()

class ChatRequest(BaseModel):
    message:str
    thread_id: str = "demo"

@app.post("/chat")
async def chat(body:ChatRequest):
    result = await agent.ainvoke(
        {"messages":[{"role":"user","content":body.message}]},
        config={"configurable":{"thread_id":body.thread_id}},
    )
    return {"reply":result["messages"][-1].content}