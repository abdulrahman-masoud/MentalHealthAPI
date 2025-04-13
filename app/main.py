from fastapi import FastAPI, Depends
from app.routers import chat
from app.models.chatbot import MentalHealthChatbot

# Create FastAPI app
app = FastAPI(
    title="Mental Health Chatbot API",
    description="API for interacting with a mental health chatbot",
    version="1.0.0"
)

# Include routers
app.include_router(chat.router)

# Create a global chatbot instance
chatbot = MentalHealthChatbot()

# Root endpoint
@app.get("/", tags=["status"])
async def root():
    return {
        "message": "Mental Health Chatbot API is running!",
        "usage": "POST to /chat with a JSON body containing a message field"
    }

# Model info endpoint
@app.get("/model-info", tags=["status"])
async def model_info():
    return chatbot.get_model_info()