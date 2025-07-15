#!/usr/bin/env python3
"""
Gemini MCP Server

This MCP server provides tools to interact with Google's Gemini API from Claude Desktop.
Uses the Google Gemini API for AI conversations and responses.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("gemini_server")

# Default settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash-8b") 
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
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE, description="Temperature (0-2)", ge=0, le=2
    )
    max_output_tokens: int = Field(
        default=MAX_OUTPUT_TOKENS, description="Maximum tokens in response", ge=1
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Optional conversation ID for chat history"
    )

# User config model for API key
class Config:
    api_key: str = None
    
    def __init__(self):
        # First check environment variables
        self.api_key = os.getenv("GEMINI_API_KEY")

config = Config()

# Initialize FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up application resources"""
    logger.info("Gemini MCP Server starting up")
    yield
    logger.info("Gemini MCP Server shutting down")

app = FastAPI(title="Gemini MCP Server", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_conversation_history(conversation_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a given conversation ID"""
    return conversation_history.get(conversation_id, [])

def add_to_conversation_history(conversation_id: str, role: str, content: str):
    """Add a message to conversation history"""
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    conversation_history[conversation_id].append({"role": role, "content": content})

# Helper function to format conversation for Gemini
def format_conversation_for_gemini(
    history: List[Dict[str, str]], new_prompt: str
) -> str:
    """Format conversation history for Gemini API"""
    formatted_conversation = ""
    for message in history:
        if message["role"] == "user":
            formatted_conversation += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            formatted_conversation += f"Assistant: {message['content']}\n"
    
    formatted_conversation += f"User: {new_prompt}\n"
    return formatted_conversation

# Parse nested configuration from query params
async def parse_config(request: Request):
    """Parse configuration from query parameters using dot notation"""
    query_params = dict(request.query_params)
    parsed_config = {}
    
    for key, value in query_params.items():
        if "." in key:
            parts = key.split(".")
            current = parsed_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            parsed_config[key] = value
    
    # Update global config if API key is provided
    if "apiKey" in parsed_config:
        config.api_key = parsed_config["apiKey"]
        # Configure Gemini with the provided API key
        genai.configure(api_key=config.api_key)
        
    return parsed_config

# MCP endpoint for tool manifest (no auth required for discovery)
@app.get("/mcp/manifest")
@app.post("/mcp/manifest")
async def get_manifest(config_data: dict = Depends(parse_config)):
    """Return the MCP manifest with available tools"""
    # Define tools for manifest - no heavy initialization required
    tools = [
        {
            "name": "ask_gemini",
            "description": "Send a prompt to Gemini and get a response",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The message to send to Gemini"
                    },
                    "model": {
                        "type": "string",
                        "default": DEFAULT_MODEL,
                        "description": "Gemini model name (gemini-pro, gemini-1.5-pro, etc.)"
                    },
                    "temperature": {
                        "type": "number",
                        "default": DEFAULT_TEMPERATURE,
                        "description": "Temperature value for generation (0-2)"
                    },
                    "max_output_tokens": {
                        "type": "integer",
                        "default": MAX_OUTPUT_TOKENS,
                        "description": "Maximum tokens in response"
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Optional conversation ID for maintaining chat history"
                    }
                },
                "required": ["prompt"]
            }
        }
    ]
    
    return tools

# Tool endpoint for Gemini API
@app.post("/mcp/run/ask_gemini")
async def run_ask_gemini(request: Request, config_data: dict = Depends(parse_config)):
    """Run the ask_gemini tool"""
    # Verify API key is provided
    if not config.api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Configure Gemini with the API key
    genai.configure(api_key=config.api_key)
    
    try:
        # Parse request body
        body = await request.json()
        
        # Extract parameters
        prompt = body.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
            
        model = body.get("model", DEFAULT_MODEL)
        temperature = body.get("temperature", DEFAULT_TEMPERATURE)
        max_output_tokens = body.get("max_output_tokens", MAX_OUTPUT_TOKENS)
        conversation_id = body.get("conversation_id")
        
        logger.info(f"Calling Gemini with model: {model}")
        
        try:
            # Initialize the model
            gemini_model = genai.GenerativeModel(model)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            
            # Handle conversation history if conversation_id is provided
            if conversation_id:
                history = get_conversation_history(conversation_id)
                if history:
                    # Format the conversation with history
                    formatted_prompt = format_conversation_for_gemini(history, prompt)
                else:
                    formatted_prompt = prompt
                
                # Add current message to history
                add_to_conversation_history(conversation_id, "user", prompt)
            else:
                formatted_prompt = prompt
            
            # Generate response
            response = gemini_model.generate_content(
                formatted_prompt, generation_config=generation_config
            )
            
            # Extract response text
            response_text = response.text
            
            # Store assistant response in history if conversation_id is provided
            if conversation_id:
                add_to_conversation_history(conversation_id, "assistant", response_text)
            
            # Return response
            return {"response": response_text, "conversation_id": conversation_id}
            
        except Exception as e:
            error_message = f"Error calling Gemini API: {str(e)}"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Root endpoint for health checks
@app.get("/")
@app.post("/")
def root():
    return {"status": "ok", "message": "Gemini API MCP Server", "models": AVAILABLE_MODELS}

# MCP endpoint for health check
@app.get("/mcp")
@app.post("/mcp")
@app.delete("/mcp")
async def mcp_root(config_data: dict = Depends(parse_config)):
    """Root MCP endpoint"""
    return {"status": "ok", "message": "Gemini MCP API ready"}

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Smithery sets $PORT)
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting Gemini MCP Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
