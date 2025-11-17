from agents import Agent,Runner,AsyncOpenAI,RunConfig,OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

python_agent = Agent(
    name="Python Agent",
    instructions=""""
    You are a Python agent to help with Python programming tasks.
    just simplify the answer.
    """
)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from Saad!"}

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def main(req: ChatMessage):
    result = await Runner.run(
        python_agent,
        req.message,
        run_config=config
    )
    return {"response": result.final_output}
