// Smithery configuration
module.exports = {
  // MCP configuration
  mcp: {
    entry: "./mcp/manifest.json",
  },
  // Build and deployment settings
  deploy: {
    root: "./",
    exclude: ["node_modules", "dist", ".git"]
  }
}; 