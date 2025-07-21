# openai-compatible-TeXmacs-plugin
TeXmacs plugin for openAI compatible LLM servers

## INSTALL
1. Copy test_py folder into ~/.TeXmacs/plugins  location is hardcoded for alpha
2. Python requirements:
   needs packages: aiohttp (and optionally pypandoc)
3. Optional for full formula support in plugin: Command line tool pandoc

## Start
Starting TeXmacs with plugin installed should give new session called "test py" for now.

## Basic Usage:
Just type your message to chat with the AI

Special Commands:
-  /help               - Show this help
-  /config             - Show current configuration
-  /config key=value   - Change configuration (e.g., /config temperature=0.5)
-  /clear              - Clear conversation history
-  /history            - Show conversation summary
-  /model [name]       - Show/change model
-  /models             - List available models
-  /test               - Test API connection
-  /oneshot <msg>      - Single request without history
-  /system [prompt]    - Show/change system prompt
-  /hidethinking [0|1] - Hide <think> blocks (0=off, 1=on)
-  /latex [0|1]        - Convert Markdown to LaTeX via pandoc (0=off, 1=on)

Configuration Keys:
  api_base, api_key, model, temperature, max_tokens, hidethinking, latex_output

Examples:
-  /config model=gpt-4-turbo
-  /hidethinking 1
-  /latex 1
