#!/usr/usr/bin/env python3
###############################################################################
##
## MODULE      : tm_openai.py
## DESCRIPTION : OpenAI Session for TeXmacs via tmpy
## COPYRIGHT   : (C) 2025 TeXmacs Community
##
## This software falls under the GNU general public license version 3 or later.
## It comes WITHOUT ANY WARRANTY WHATSOEVER. For details, see the file LICENSE
## in the root directory or <http://www.gnu.org/licenses/gpl-3.0.html>.

import os
import sys
import asyncio
import json
import traceback
import platform
import re
from os.path import exists
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Standard tmpy setup
tmpy_home_path = os.environ.get("TEXMACS_HOME_PATH") + "/plugins/tmpy"
if (exists(tmpy_home_path)):
    sys.path.append(os.environ.get("TEXMACS_HOME_PATH") + "/plugins/")
else:
    sys.path.append(os.environ.get("TEXMACS_PATH") + "/plugins/")

# tmpy imports
from tmpy.protocol import *
from tmpy.compat import *

# Additional imports
import aiohttp
try:
    from markdown_it import MarkdownIt
    from mdit_py_plugins.dollarmath import dollarmath_plugin
except ImportError:
    MarkdownIt = None
    dollarmath_plugin = None

###############################################################################
## Configuration Management
###############################################################################

@dataclass
class OpenAIConfig:
    """Configuration for the OpenAI Session."""
    api_base: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 2000
    hidethinking: int = 0 # 0 = off, 1 = on
    markdown_parse: int = 1 # 0 = off, 1 = on
    system_prompt: str = (
        "You are an AI assistant integrated into TeXmacs, a scientific text editor. "
        "Provide helpful, accurate responses. Use Markdown formatting when appropriate, "
        "including for mathematical notation using $...$ for inline and $$...$$ for display math. "
        "Keep responses concise but informative."
    )
    
    @classmethod
    def load(cls):
        """Loads configuration from a file."""
        config_file = os.path.expanduser(os.environ.get("TEXMACS_HOME_PATH")+"/plugins/test_py/openai_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                fields = {f.name for f in cls.__dataclass_fields__.values()}
                filtered_data = {k: v for k, v in config_data.items() if k in fields}
                return cls(**filtered_data)
            except Exception as e:
                flush_verbatim(f"Config load error: {e}")
                flush_newline()
        return cls()

    def save(self):
        """Saves configuration to a file."""
        config_file = os.path.expanduser(os.environ.get("TEXMACS_HOME_PATH")+"/plugins/test_py/openai_config.json")
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        try:
            with open(config_file, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            flush_verbatim(f"Config save error: {e}")
            flush_newline()

###############################################################################
## OpenAI API Client
###############################################################################

class OpenAIClient:
    """OpenAI-compatible API Client."""
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.session = None
        self.conversation_history = []
    
    async def initialize(self):
        """Initializes the HTTP Session."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    
    async def chat_completion(self, message: str, use_history: bool = True) -> str:
        """Sends a chat completion request."""
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        if use_history:
            messages.extend(self.conversation_history[-20:])
        messages.append({"role": "user", "content": message})

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = { "model": self.config.model, "messages": messages, "temperature": self.config.temperature, "max_tokens": self.config.max_tokens, "stream": False }
        
        try:
            async with self.session.post( f"{self.config.api_base}/chat/completions", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    reply = data["choices"][0]["message"]["content"]
                    if use_history:
                        self.conversation_history.append({"role": "user", "content": message})
                        self.conversation_history.append({"role": "assistant", "content": reply})
                    return reply
                else:
                    return f"API Error {response.status}: {await response.text()}"
        except asyncio.TimeoutError:
            return "Error: Request timeout."
        except Exception as e:
            return f"Connection Error: {str(e)}"
    
    def clear_history(self):
        """Clears the conversation history."""
        self.conversation_history = []
    
    def get_history_summary(self) -> str:
        """Returns a summary of the history."""
        if not self.conversation_history: return "No conversation history"
        return f"Conversation history: {len(self.conversation_history) // 2} message pairs"
    
    async def cleanup(self):
        """Cleans up resources."""
        if self.session: await self.session.close()

###############################################################################
## Command Processing
###############################################################################

class CommandProcessor:
    """Processes TeXmacs commands."""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.openai_client = OpenAIClient(config)
        if MarkdownIt and dollarmath_plugin:
            self.md_parser = MarkdownIt().use(dollarmath_plugin)
        else:
            self.md_parser = None

    async def initialize(self):
        """Initializes all components."""
        await self.openai_client.initialize()

    def _filter_thinking_blocks(self, text: str) -> str:
        """Removes <think>...</think> blocks from the text."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _flush_inline_tokens(self, tokens: List[Dict]):
        """Processes a list of inline tokens and flushes them."""
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == 'text':
                flush_verbatim(token.content)
            elif token.type == 'strong_open':
                if i + 2 < len(tokens) and tokens[i+1].type == 'text' and tokens[i+2].type == 'strong_close':
                    flush_latex(r'\textbf{' + tokens[i+1].content + '}')
                    i += 2
                else:
                    flush_verbatim('**')
            elif token.type == 'em_open':
                if i + 2 < len(tokens) and tokens[i+1].type == 'text' and tokens[i+2].type == 'em_close':
                    flush_latex(r'\textit{' + tokens[i+1].content + '}')
                    i += 2
                else:
                    flush_verbatim('*')
            elif token.type == 'code_inline':
                flush_latex(r'\texttt{' + token.content + '}')
            elif token.type == 'math_inline':
                flush_latex(f'${token.content}$')
            i += 1

    def _flush_markdown(self, markdown_text: str):
        """Parses Markdown text and flushes it as formatted output in TeXmacs."""
        if not self.md_parser:
            flush_verbatim("ERROR: markdown-it-py and/or mdit-plugins not installed.")
            return
        
        tokens = self.md_parser.parse(markdown_text)

        for i, token in enumerate(tokens):
            if token.type == 'paragraph_open':
                inline_content_token = tokens[i+1]
                if inline_content_token.type == 'inline' and inline_content_token.children:
                    self._flush_inline_tokens(inline_content_token.children)
            elif token.type == 'paragraph_close':
                flush_newline(2)
            elif token.type == 'fence':
                flush_latex(f'\\begin{{verbatim}}\n{token.content}\\end{{verbatim}}')
                flush_newline(2)
            elif token.type == 'math_block':
                flush_latex(r'\[' + token.content + r'\]')
                flush_newline(2)

    async def process_and_flush_command(self, command: str):
        """Processes a command and flushes the result directly."""
        command = command.strip()
        if command.startswith("/"):
            response_str = await self._handle_special_command(command)
            flush_verbatim(response_str)
            flush_newline(2)
            return

        raw_response = await self.openai_client.chat_completion(command)
        if self.config.hidethinking == 1:
            raw_response = self._filter_thinking_blocks(raw_response)
        
        if self.config.markdown_parse == 1:
            self._flush_markdown(raw_response)
        else:
            flush_verbatim(raw_response)
            flush_newline(2)

    async def _handle_special_command(self, command: str) -> str:
        """Handles special commands. Always returns a string."""
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "help": return self._show_help()
        elif cmd == "clear":
            self.openai_client.clear_history()
            return "✓ Conversation history cleared"
        elif cmd == "hidethinking":
            if args in ("0", "1"):
                self.config.hidethinking = int(args)
                self.config.save()
                status = "enabled" if self.config.hidethinking == 1 else "disabled"
                return f"✓ Hiding of thinking blocks is now {status}."
            status = "enabled (1)" if self.config.hidethinking == 1 else "disabled (0)"
            return f"Current status: {status}. Use /hidethinking [0|1] to change."
        elif cmd == "markdown":
            if args in ("0", "1"):
                self.config.markdown_parse = int(args)
                self.config.save()
                status = "enabled" if self.config.markdown_parse == 1 else "disabled"
                return f"✓ Markdown parsing is now {status}."
            status = "enabled (1)" if self.config.markdown_parse == 1 else "disabled (0)"
            return f"Current status: {status}. Use /markdown [0|1] to change."
        elif cmd == "config":
             return (f"Current configuration:\n"
                   f"  API Base: {self.config.api_base}\n"
                   f"  API Key: {'***' if self.config.api_key else 'not set'}\n"
                   f"  Model: {self.config.model}\n"
                   f"  Temperature: {self.config.temperature}\n"
                   f"  Max Tokens: {self.config.max_tokens}\n"
                   f"  Hide Thinking: {self.config.hidethinking}\n"
                   f"  Markdown Parse: {self.config.markdown_parse}")
        elif cmd == "model":
            if args:
                self.config.model = args
                self.config.save()
                return f"✓ Model changed to: {args}"
            else:
                return f"Current model: {self.config.model}"
        else:
            return f"Unknown command: {cmd}. Type /help for available commands."

    def _show_help(self) -> str:
        """Displays help information."""
        return """OpenAI Session Commands:

Basic Usage:
  Just type your message to chat with the AI

Special Commands:
  /help               - Show this help
  /config             - Show current configuration
  /clear              - Clear conversation history
  /model [name]       - Show/change model
  /hidethinking [0|1] - Hide <think> blocks (0=off, 1=on)
  /markdown [0|1]     - Parse Markdown output (0=off, 1=on)

Note: Markdown parsing requires 'pip install markdown-it-py mdit-plugins'
"""

###############################################################################
## Session Main Loop
###############################################################################

async def main():
    """Main loop for the OpenAI Session."""
    flush_verbatim("OpenAI Session for TeXmacs")
    flush_newline()
    flush_verbatim(f"Python {platform.python_version()} [{sys.executable}]")
    flush_newline()
    if MarkdownIt is None or dollarmath_plugin is None:
        flush_verbatim("WARNING: 'markdown-it-py' or 'mdit-plugins' not found.")
        flush_verbatim("Markdown parsing will be disabled. Please run:")
        flush_verbatim("pip install markdown-it-py mdit-plugins")
        flush_newline()
    flush_verbatim("Type /help for available commands")
    flush_newline(2)

    config = OpenAIConfig.load()
    if not (MarkdownIt and dollarmath_plugin):
        config.markdown_parse = 0 

    flush_verbatim(f"Using API: {config.api_base}")
    flush_verbatim(f"Model: {config.model}")
    flush_newline(2)

    processor = CommandProcessor(config)
    await processor.initialize()

    try:
        while True:
            line = tm_input()
            if not line or not line.strip():
                continue
            try:
                await processor.process_and_flush_command(line)
            except Exception as e:
                flush_verbatim(f"Error: {str(e)}")
                flush_newline()
                if "API" not in str(e) and "Connection" not in str(e):
                    flush_verbatim("Debug traceback:")
                    flush_verbatim(traceback.format_exc())
                flush_newline(2)
    except (KeyboardInterrupt, EOFError):
        flush_verbatim("Session ended.")
        flush_newline()
    finally:
        await processor.cleanup()

###############################################################################
## Session Startup
###############################################################################

if __name__ == "__main__":
    if py_ver >= 3:
        asyncio.run(main())
    else:
        flush_err("Python 3 or higher is required for this session.")
        exit(-1)
