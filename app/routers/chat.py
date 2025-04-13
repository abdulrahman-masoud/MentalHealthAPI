from fastapi import APIRouter, HTTPException, Depends
from app.schemas.chat import ChatRequest, ChatResponse
from app.models.chatbot import MentalHealthChatbot

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

# We'll use this function to get our chatbot instance
# In a more complex app, you might use a dependency injection system
def get_chatbot():
    # This would ideally come from a singleton or dependency injection system
    # but for simplicity, we'll create it here
    return MentalHealthChatbot()

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, chatbot: MentalHealthChatbot = Depends(get_chatbot)):
    try:
        response = chatbot.generate_response(
            message=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")