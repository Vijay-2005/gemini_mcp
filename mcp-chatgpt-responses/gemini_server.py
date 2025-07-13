#!/usr/bin/env python3
"""
Gemini API Web Server

This converts the MCP server into a regular web API that can be deployed on Render.
"""

import os
import json
import logging
from typing import Optional, Dict, List

from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("gemini_server")

# Check for API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure Gemini
genai.configure(api_key=api_key)

# Default settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-pro")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))

# Model list for resource
AVAILABLE_MODELS = [
    "gemini-pro",
    "gemini-pro-vision",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

# Store conversation history
conversation_history: Dict[str, List[Dict[str, str]]] = {}


class GeminiRequest(BaseModel):
    """Model for Gemini API request parameters"""
    prompt: str = Field(description="The message to send to Gemini")
    model: str = Field(default=DEFAULT_MODEL, description="Gemini model name")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, description="Temperature (0-2)", ge=0, le=2)
    max_output_tokens: int = Field(default=MAX_OUTPUT_TOKENS, description="Maximum tokens in response", ge=1)
    conversation_id: Optional[str] = Field(default=None, description="Optional conversation ID for chat history")


class GeminiResponse(BaseModel):
    """Model for Gemini API response"""
    response: str
    conversation_id: Optional[str] = None


def get_conversation_history(conversation_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a given conversation ID"""
    return conversation_history.get(conversation_id, [])


def add_to_conversation_history(conversation_id: str, role: str, content: str):
    """Add a message to conversation history"""
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    conversation_history[conversation_id].append({"role": role, "content": content})


def format_conversation_for_gemini(history: List[Dict[str, str]], new_prompt: str) -> str:
    """Format conversation history for Gemini API"""
    formatted_conversation = ""
    for message in history:
        if message["role"] == "user":
            formatted_conversation += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            formatted_conversation += f"Assistant: {message['content']}\n"
    
    formatted_conversation += f"User: {new_prompt}\n"
    return formatted_conversation


# Initialize FastAPI - MOVED AFTER FUNCTION DEFINITIONS
app = FastAPI(title="Gemini API Server", version="1.0.0")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Gemini API Server", "models": AVAILABLE_MODELS}


@app.get("/models")
async def get_models():
    """Get available models"""
    return {"models": AVAILABLE_MODELS}


@app.post("/ask", response_model=GeminiResponse)
async def ask_gemini(request: GeminiRequest):
    """
    Send a prompt to Gemini and get a response
    """
    logger.info(f"Calling Gemini with model: {request.model}")
    
    try:
        # Initialize the model
        gemini_model = genai.GenerativeModel(request.model)
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )
        
        # Handle conversation history if conversation_id is provided
        if request.conversation_id:
            history = conversation_history.get(request.conversation_id, [])
            if history:
                # Format the conversation with history
                formatted_prompt = format_conversation_for_gemini(history, request.prompt)
            else:
                formatted_prompt = request.prompt
            
            # Add current message to history
            add_to_conversation_history(request.conversation_id, "user", request.prompt)
        else:
            formatted_prompt = request.prompt
        
        # Generate response
        response = gemini_model.generate_content(
            formatted_prompt,
            generation_config=generation_config
        )
        
        # Extract response text
        response_text = response.text
        
        # Store assistant response in history if conversation_id is provided
        if request.conversation_id:
            add_to_conversation_history(request.conversation_id, "assistant", response_text)
        
        return GeminiResponse(
            response=response_text,
            conversation_id=request.conversation_id
        )
    
    except Exception as e:
        error_message = f"Error calling Gemini API: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)


if __name__ == "__main__":
    import uvicorn
    # Get port from environment (Render sets $PORT)
    port = int(os.environ.get("PORT", 8080))
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)