# Model Context Protocol

1. link
   * [documentation](https://modelcontextprotocol.io/)
2. concept
   * client
   * resources: file-like data that can be read by clients (like API responses or file contents)
   * tools: functions that can be called by the LLM (with user approval)
   * prompts: pre-written templates that help users accomplish specific tasks
3. install
   * `pip install "mcp[cli]"`
4. config file
   * macos: `~/Library/Application Support/Claude/claude_desktop_config.json`
   * windows: `$env:AppData\Claude\claude_desktop_config.json`

```bash
micromamba install -c conda-forge httpx
```

```json
{
  "mcpServers": {
    "weather": {
      "command": "/Users/zhangc/.local/bin/micromamba",
      "args": [
        "run",
        "-n",
        "metal",
        "--cwd",
        "/Users/zhangc/Documents/learning/python/openai/ws_mcp",
        "python",
        "weather.py"
      ]
    }
  }
}
```
