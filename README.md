# openai-compatible-TeXmacs-plugin
TeXmacs plugin for openAI compatible LLM servers

<img width="2050" height="1170" alt="grafik" src="https://github.com/user-attachments/assets/f3f8c340-0bfe-4cea-ac48-89669b532345" />



## INSTALL
1. Copy test_py folder into ~/.TeXmacs/plugins  location is hardcoded for alpha
2. Python requirements:
   needs packages: aiohttp (and optionally markdown-it-py mdit-plugins)
3. You can change the server, default prompt, api-key and such by editing openai_config.json

## Start
Starting TeXmacs with plugin installed should give new session called "LLM plugin" for now.

## Basic Usage:
Just type your message to chat with the AI

Special Commands:
-  /help               - Show this help
-  /config             - Show current configuration
-  /clear              - Clear conversation history
-  /history            - Show conversation summary
-  /model [name]       - Show/change model
-  /hidethinking [0|1] - Hide <think> blocks (0=off, 1=on)
-  /markdown [0|1]     - Parse Markdown output (0=off, 1=on)

Examples:
-  /model llama3.2
-  /hidethinking 1
-  /markdown 1
