[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/billster45-mcp-chatgpt-responses-badge.png)](https://mseep.ai/app/billster45-mcp-chatgpt-responses)

# MCP Gemini Server

This MCP server allows you to access Google's Gemini API directly from Claude Desktop.

## Features

- Call the Gemini API with customizable parameters
- Ask Claude and Gemini to talk to each other in a long running discussion!
- Configure model versions, temperature, and other parameters
- Access to various Gemini models (gemini-pro, gemini-1.5-pro, gemini-1.5-flash, etc.)
- Conversation history management for context-aware responses
- Use your own Google AI Studio API key

## Setup Instructions

### Installing via Smithery

To install Gemini Server for Claude Desktop automatically via [Smithery](https://smithery.ai):

```bash
# Manual installation required - see below
```

### Prerequisites

- Python 3.10 or higher
- [Claude Desktop](https://claude.ai/download) application
- [Google AI Studio API key](https://aistudio.google.com/app/apikey)
- [uv](https://github.com/astral-sh/uv) for Python package management

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/mcp-gemini-responses.git
   cd mcp-gemini-responses
   ```

2. Set up a virtual environment and install dependencies using uv:
   ```bash
   uv venv
   ```

   ```bash
   .venv\\Scripts\\activate
   ```
   
   ```bash
   uv pip install -r requirements.txt
   ```

### Using with Claude Desktop

1. Configure Claude Desktop to use this MCP server by following the instructions at:
   [MCP Quickstart Guide](https://modelcontextprotocol.io/quickstart/user#2-add-the-filesystem-mcp-server)

2. Add the following configuration to your Claude Desktop config file (adjust paths as needed):
   ```json
   {
     "mcpServers": {
       "gemini": {
         "command": "uv",
         "args": [
           "--directory",
           "\\path\\to\\mcp-gemini-responses",
           "run",
           "gemini_server.py"
         ],
         "env": {
           "GEMINI_API_KEY": "your-api-key-here",
           "DEFAULT_MODEL": "gemini-pro",
           "DEFAULT_TEMPERATURE": "0.7",
           "MAX_OUTPUT_TOKENS": "1000"
         }
       }
     }
   }
   ```

3. Restart Claude Desktop.

4. You can now use the Gemini API through Claude by asking questions that mention Gemini or that Claude might not be able to answer.

## Available Tools

The MCP server provides the following tools:

1. `ask_gemini(prompt, model, temperature, max_output_tokens, conversation_id)` - Send a prompt to Gemini and get a response

2. `ask_gemini_with_search(prompt, model, temperature, max_output_tokens, conversation_id)` - Send a prompt to Gemini with enhanced search instructions

## Example Usage

### Basic Gemini usage:

Tell Claude to ask Gemini a question!
```
Use the ask_gemini tool to answer: What is the best way to learn Python?
```

Tell Claude to have a conversation with Gemini:
```
Use the ask_gemini tool to have a two way conversation between you and Gemini about the topic that is most important to you.
```
Note how the conversation_id allows maintaining conversation history for context-aware responses across multiple interactions.

### With search capability:

For questions that may benefit from comprehensive information:
```
Use the ask_gemini_with_search tool to answer: What are the latest developments in quantum computing?
```

Try search-enhanced responses for planning:
```
Use the ask_gemini_with_search tool to find information about weather patterns and based on that, keep using the tool to build up a great day out for someone who loves food and parks
```

## How It Works

This tool utilizes Google's Gemini API with conversation state management. This approach:

1. Maintains conversation history locally for context tracking
2. Provides access to various Gemini models (pro, flash, vision, etc.)
3. Improves the user experience by maintaining context across messages
4. Allows enhanced responses with search-oriented prompting

## License

MIT License
