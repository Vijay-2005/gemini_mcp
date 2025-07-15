// Tool to send prompts to Gemini API with web search enhancement
const { GoogleGenerativeAI } = require("@google/generative-ai");

// Store conversation history
const conversationHistory = {};

// Get conversation history for a given ID
function getConversationHistory(conversationId) {
  return conversationHistory[conversationId] || [];
}

// Add to conversation history
function addToConversationHistory(conversationId, role, content) {
  if (!conversationHistory[conversationId]) {
    conversationHistory[conversationId] = [];
  }
  conversationHistory[conversationId].push({ role, content });
}

// Format conversation for Gemini with web search enhancement
function formatConversationForGemini(history, newPrompt) {
  let formatted = "";
  for (const message of history) {
    if (message.role === "user") {
      formatted += `User: ${message.content}\n`;
    } else if (message.role === "assistant") {
      formatted += `Assistant: ${message.content}\n`;
    }
  }
  formatted += `User: ${newPrompt}\n`;
  return formatted;
}

// Add web search instructions to the prompt
function enhancePromptWithSearchInstructions(prompt) {
  return `Please search the web for the most up-to-date and comprehensive information to answer this question: ${prompt}

Remember to:
1. Use current information from the web
2. Cite any specific sources you use
3. Be thorough in your research
4. Provide a complete and detailed answer

Question: ${prompt}`;
}

// The tool implementation
module.exports = {
  async run({ prompt, model = "gemini-pro", temperature = 0.7, max_output_tokens = 1000, conversation_id = null }) {
    try {
      // Get API key from environment
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error("GEMINI_API_KEY environment variable is not set");
      }

      // Initialize the API client
      const genAI = new GoogleGenerativeAI(apiKey);
      const genModel = genAI.getGenerativeModel({ model });

      // Set up generation config
      const generationConfig = {
        temperature,
        maxOutputTokens: max_output_tokens,
      };

      // Handle conversation history and enhance prompt with search instructions
      let formattedPrompt = enhancePromptWithSearchInstructions(prompt);
      if (conversation_id) {
        const history = getConversationHistory(conversation_id);
        if (history.length > 0) {
          formattedPrompt = formatConversationForGemini(history, enhancePromptWithSearchInstructions(prompt));
        }
        
        // Add current message to history
        addToConversationHistory(conversation_id, "user", prompt);
      }

      // Generate content
      const result = await genModel.generateContent({
        contents: [{ role: "user", parts: [{ text: formattedPrompt }] }],
        generationConfig,
      });

      const response = result.response;
      const responseText = response.text();

      // Store assistant response in history if conversation_id is provided
      if (conversation_id) {
        addToConversationHistory(conversation_id, "assistant", responseText);
      }

      return {
        response: responseText,
        conversation_id
      };
    } catch (error) {
      console.error("Error calling Gemini API:", error);
      throw new Error(`Error calling Gemini API: ${error.message}`);
    }
  }
}; 