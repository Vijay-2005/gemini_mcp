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

To install Gemini Server for Claude Desktop via [Smithery](https://smithery.ai):

```bash
# Install dependencies
npm install

# Deploy to Smithery
npm run deploy
```

After deployment, you should be able to access your MCP server at:
```
https://smithery.ai/server/@YOUR_USERNAME/gemini_mcp/mcp/manifest
```

### Prerequisites

- Node.js 18+ (for Smithery deployment)
- [Google AI Studio API key](https://aistudio.google.com/app/apikey)
- [Claude Desktop](https://claude.ai/download) application

### Installation for Local Development

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/gemini_mcp.git
   cd gemini_mcp
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set your Gemini API key in the Smithery dashboard or through the config file.

4. Deploy to Smithery:
   ```bash
   npm run deploy
   ```

### Using with Claude Desktop

1. Configure Claude Desktop to use this MCP server by following the instructions at:
   [MCP Quickstart Guide](https://modelcontextprotocol.io/quickstart/user#2-add-the-filesystem-mcp-server)

2. Add the following configuration to your Claude Desktop settings:
   - Name: Gemini
   - URL: https://smithery.ai/server/@YOUR_USERNAME/gemini_mcp
   - API Key: your-gemini-api-key-here

3. Restart Claude Desktop.

4. You can now use the Gemini API through Claude by asking questions that mention Gemini.

## Available Tools

The MCP server provides the following tools:

1. `ask_gemini(prompt, model, temperature, max_output_tokens, conversation_id)` - Send a prompt to Gemini and get a response

2. `ask_gemini_with_web_search(prompt, model, temperature, max_output_tokens, conversation_id)` - Send a prompt to Gemini with enhanced search instructions

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
Use the ask_gemini_with_web_search tool to answer: What are the latest developments in quantum computing?
```

Try search-enhanced responses for planning:
```
Use the ask_gemini_with_web_search tool to find information about weather patterns and based on that, keep using the tool to build up a great day out for someone who loves food and parks
```

## How It Works

This tool utilizes Google's Gemini API with conversation state management. This approach:

1. Maintains conversation history locally for context tracking
2. Provides access to various Gemini models (pro, flash, vision, etc.)
3. Improves the user experience by maintaining context across messages
4. Allows enhanced responses with search-oriented prompting

## License

MIT License
