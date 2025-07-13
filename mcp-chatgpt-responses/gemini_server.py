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
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Context

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

# Initialize FastMCP server
mcp = FastMCP(
    "Gemini API",
    dependencies=["google-generativeai", "python-dotenv", "httpx", "pydantic"],
)


class GeminiRequest(BaseModel):
    """Model for Gemini API request parameters"""
    model: str = Field(default=DEFAULT_MODEL, description="Gemini model name")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, description="Temperature (0-2)", ge=0, le=2)
    max_output_tokens: int = Field(default=MAX_OUTPUT_TOKENS, description="Maximum tokens in response", ge=1)
    conversation_id: Optional[str] = Field(default=None, description="Optional conversation ID for chat history")


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Initialize and clean up application resources"""
    logger.info("Gemini MCP Server starting up")
    try:
        yield {}
    finally:
        logger.info("Gemini MCP Server shutting down")


# Resources

@mcp.resource("gemini://models")
def available_models() -> str:
    """List available Gemini models"""
    return json.dumps(AVAILABLE_MODELS, indent=2)


# Store conversation history
conversation_history: Dict[str, List[Dict[str, str]]] = {}


def get_conversation_history(conversation_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a given conversation ID"""
    return conversation_history.get(conversation_id, [])


def add_to_conversation_history(conversation_id: str, role: str, content: str):
    """Add a message to conversation history"""
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    conversation_history[conversation_id].append({"role": role, "content": content})


# Helper function to format conversation for Gemini
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


# Tools

@mcp.tool()
async def ask_gemini(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    conversation_id: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Send a prompt to Gemini and get a response
    
    Args:
        prompt: The message to send to Gemini
        model: The Gemini model to use (default: gemini-pro)
        temperature: Sampling temperature (0-2, default: 0.7)
        max_output_tokens: Maximum tokens in response (default: 1000)
        conversation_id: Optional conversation ID for maintaining chat history
    
    Returns:
        Gemini's response
    """
    ctx.info(f"Calling Gemini with model: {model}")
    
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
            formatted_prompt,
            generation_config=generation_config
        )
        
        # Extract response text
        response_text = response.text
        
        # Store assistant response in history if conversation_id is provided
        if conversation_id:
            add_to_conversation_history(conversation_id, "assistant", response_text)
        
        # Return response with conversation ID for reference
        if conversation_id:
            return f"{response_text}\n\n(Conversation ID: {conversation_id})"
        else:
            return response_text
    
    except Exception as e:
        error_message = f"Error calling Gemini API: {str(e)}"
        logger.error(error_message)
        return error_message


@mcp.tool()
async def ask_gemini_with_search(
    prompt: str,
    model: str = "gemini-pro",  # Use gemini-pro for search capability
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    conversation_id: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Send a prompt to Gemini with search capability enabled (uses Gemini Pro)
    
    Args:
        prompt: The message to send to Gemini
        model: The Gemini model to use (default: gemini-pro)
        temperature: Sampling temperature (0-2, default: 0.7)
        max_output_tokens: Maximum tokens in response (default: 1000)
        conversation_id: Optional conversation ID for maintaining chat history
    
    Returns:
        Gemini's response with search information
    """
    ctx.info(f"Calling Gemini with search capability using model: {model}")
    
    try:
        # Initialize the model
        gemini_model = genai.GenerativeModel(model)
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        
        # Add search instruction to the prompt
        search_enhanced_prompt = f"""Please search for current information and provide a comprehensive answer to this question: {prompt}

Use your knowledge and any available search capabilities to provide accurate, up-to-date information."""
        
        # Handle conversation history if conversation_id is provided
        if conversation_id:
            history = get_conversation_history(conversation_id)
            if history:
                # Format the conversation with history
                formatted_prompt = format_conversation_for_gemini(history, search_enhanced_prompt)
            else:
                formatted_prompt = search_enhanced_prompt
            
            # Add current message to history
            add_to_conversation_history(conversation_id, "user", prompt)
        else:
            formatted_prompt = search_enhanced_prompt
        
        # Generate response
        response = gemini_model.generate_content(
            formatted_prompt,
            generation_config=generation_config
        )
        
        # Extract response text
        response_text = response.text
        
        # Store assistant response in history if conversation_id is provided
        if conversation_id:
            add_to_conversation_history(conversation_id, "assistant", response_text)
        
        # Return response with conversation ID for reference
        if conversation_id:
            return f"{response_text}\n\n(Conversation ID: {conversation_id})"
        else:
            return response_text
    
    except Exception as e:
        error_message = f"Error calling Gemini with search: {str(e)}"
        logger.error(error_message)
        return error_message


if __name__ == "__main__":
    # Get port from environment (Render sets $PORT)
    port = int(os.environ.get("PORT", 8080))
    # Run the server on the specified port
    mcp.run(transport='streamable-http', port=port)