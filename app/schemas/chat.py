from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message to the chatbot")
    max_tokens: int = Field(200, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_k: int = Field(50, description="Number of highest probability tokens to keep")
    top_p: float = Field(0.9, description="Cumulative probability for nucleus sampling")
    repetition_penalty: float = Field(1.2, description="Penalty for repetition")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's response")