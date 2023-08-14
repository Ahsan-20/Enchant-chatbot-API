import os
from fastapi import FastAPI
import uvicorn
import openai
from dotenv import load_dotenv
from routers import eleven


# Load the environment variables from the .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

app.include_router(eleven.router)



if __name__== '__main__':
    uvicorn.run(app)
