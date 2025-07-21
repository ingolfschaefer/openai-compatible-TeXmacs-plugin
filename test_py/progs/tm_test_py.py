#!/usr/bin/env python3
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
    import pypandoc
except ImportError:
    pypandoc = None

###############################################################################
## Configuration Management
###############################################################################

@dataclass
class OpenAIConfig:
    """default config for openai session with ollama"""
    api_base: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "llama-3.2"
    temperature: float = 0.7
    max_tokens: int = 2000
    hidethinking: int = 0 # 0 = off, 1 = on
    latex_output: int = 0 # 0 = off, 1 = on
    system_prompt: str = (
        "You are an AI assistant integrated into TeXmacs, a scientific text editor. "
        "Provide helpful, accurate responses. When appropriate, use Markdown formatting, "
        "including for mathematical notation (e.g., $E=mc^2$). "
        "Keep responses concise but informative."
    )

    @classmethod
    def load(cls):
        """load config from file. FIXME: hardcoded path"""
        config_file = os.path.expanduser("~/.TeXmacs/plugins/test_py/openai_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                # Ensure all fields are present, using defaults for missing ones
                fields = {f.name for f in cls.__dataclass_fields__.values()}
                filtered_data = {k: v for k, v in config_data.items() if k in fields}
                return cls(**filtered_data)
            except Exception as e:
                flush_verbatim(f"Config load error: {e}")
                flush_newline()
        return cls()

    def save(self):
        """save config to file- FIXME: hardcoded path"""
        config_file = os.path.expanduser("~/.TeXmacs/plugins/test_py/openai_config.json")
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
    """OpenAI-compatible API client"""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.session = None
        self.conversation_history = []

    async def initialize(self):
        """init HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )

    async def chat_completion(self, message: str, use_history: bool = True) -> str:
        """Sendet Chat-Completion Anfrage"""
        messages = []

        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        if use_history:
            messages.extend(self.conversation_history[-20:])

        messages.append({"role": "user", "content": message})

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }

        try:
            async with self.session.post(
                f"{self.config.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    reply = data["choices"][0]["message"]["content"]

                    if use_history:
                        self.conversation_history.append({"role": "user", "content": message})
                        self.conversation_history.append({"role": "assistant", "content": reply})

                    return reply
                else:
                    error_text = await response.text()
                    return f"API Error {response.status}: {error_text}"

        except asyncio.TimeoutError:
            return "Error: Request timeout. Check your connection and API endpoint."
        except Exception as e:
            return f"Connection Error: {str(e)}"

    def clear_history(self):
        """Löscht Conversation History"""
        self.conversation_history = []

    def get_history_summary(self) -> str:
        """Gibt eine Zusammenfassung der History zurück"""
        if not self.conversation_history:
            return "No conversation history"
        return f"Conversation history: {len(self.conversation_history) // 2} message pairs"

    async def cleanup(self):
        """Cleanup Ressourcen"""
        if self.session:
            await self.session.close()

###############################################################################
## Command Processing
###############################################################################

class CommandProcessor:
    """handle input from TeXmacs"""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.openai_client = OpenAIClient(config)

    async def initialize(self):
        """init all components"""
        await self.openai_client.initialize()

    async def process_command(self, command: str) -> str:
        """process the commands given"""
        command = command.strip()
        if command.startswith("/"):
            return await self._handle_special_command(command)
        else:
            return await self._handle_chat_message(command)

    def _filter_thinking_blocks(self, text: str) -> str:
        """renmmove <think>...</think> blocks from the text."""
        pattern = r'<think>.*?</think>'
        filtered_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return filtered_text.strip()

    def _convert_markdown_to_latex(self, text: str) -> str:
        """convert markdown to latex using pandoc."""
        if pypandoc is None:
            return "ERROR: pypandoc library is not installed. Please run 'pip install pypandoc'."
        try:
            return pypandoc.convert_text(text, 'latex', format='md')
        except OSError:
            return "ERROR: pandoc executable not found. Please install pandoc and ensure it is in your system's PATH."
        except Exception as e:
            return f"ERROR during LaTeX conversion: {e}"

    async def _get_filtered_chat_completion(self, message: str, use_history: bool = True) -> str:
        """Ruft die Chat-Completion auf und wendet Filter/Konvertierungen an."""
        response = await self.openai_client.chat_completion(message, use_history=use_history)
        if self.config.hidethinking == 1:
            response = self._filter_thinking_blocks(response)
        if self.config.latex_output == 1:
            response = self._convert_markdown_to_latex(response)
        return response

    async def _handle_special_command(self, command: str) -> str:
        """handle special commands for plugin (beginning with /)"""
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            return self._show_help()

        elif cmd == "config":
            return await self._handle_config_command(args)

        elif cmd == "clear":
            self.openai_client.clear_history()
            return "✓ Conversation history cleared"

        elif cmd == "history":
            return self.openai_client.get_history_summary()

        elif cmd == "model":
            if args:
                self.config.model = args
                self.config.save()
                return f"✓ Model changed to: {args}"
            return f"Current model: {self.config.model}"

        elif cmd == "hidethinking":
            if args in ("0", "1"):
                self.config.hidethinking = int(args)
                self.config.save()
                status = "enabled" if self.config.hidethinking == 1 else "disabled"
                return f"✓ Hiding of thinking blocks is now {status}."
            status = "enabled (1)" if self.config.hidethinking == 1 else "disabled (0)"
            return f"Current status: {status}. Use /hidethinking [0|1] to change."
            
        elif cmd == "latex":
            if args in ("0", "1"):
                self.config.latex_output = int(args)
                self.config.save()
                status = "enabled" if self.config.latex_output == 1 else "disabled"
                return f"✓ Markdown-to-LaTeX conversion is now {status}."
            status = "enabled (1)" if self.config.latex_output == 1 else "disabled (0)"
            return f"Current status: {status}. Use /latex [0|1] to change."

        elif cmd == "models":
            return await self._list_available_models()

        elif cmd == "test":
            return await self._test_connection()

        elif cmd == "oneshot":
            if args:
                return await self._get_filtered_chat_completion(args, use_history=False)
            return "Usage: /oneshot <message>"

        elif cmd == "system":
            if args:
                self.config.system_prompt = args
                self.config.save()
                return "✓ System prompt updated"
            return f"Current system prompt:\n{self.config.system_prompt}"

        else:
            return f"Unknown command: {cmd}. Type /help for available commands."

    async def _handle_chat_message(self, message: str) -> str:
        """Behandelt normale Chat-Nachrichten"""
        return await self._get_filtered_chat_completion(message)

    async def _handle_config_command(self, args: str) -> str:
        """handle configuration switches"""
        if not args:
            return (f"Current configuration:\n"
                   f"  API Base: {self.config.api_base}\n"
                   f"  API Key: {'***' if self.config.api_key else 'not set'}\n"
                   f"  Model: {self.config.model}\n"
                   f"  Temperature: {self.config.temperature}\n"
                   f"  Max Tokens: {self.config.max_tokens}\n"
                   f"  Hide Thinking: {self.config.hidethinking}\n"
                   f"  LaTeX Output: {self.config.latex_output}")

        try:
            key, value = args.split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            if hasattr(self.config, key):
                field_type = type(getattr(self.config, key))
                try:
                    setattr(self.config, key, field_type(value))
                    self.config.save()
                    return f"✓ {key} updated"
                except (ValueError, TypeError):
                    return f"Error: Invalid value '{value}' for {key} (expected {field_type.__name__})"
            else:
                return f"Unknown config key: {key}"

        except ValueError:
            return "Usage: /config key=value"

    async def _list_available_models(self) -> str:
        """list all available models"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with self.openai_client.session.get(
                f"{self.config.api_base}/models",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["id"] for model in data.get("data", [])]
                    return "Available models:\n" + "\n".join(f"  {model}" for model in models)
                else:
                    return f"Could not fetch models (status {response.status})"
        except Exception as e:
            return f"Error fetching models: {e}"

    async def _test_connection(self) -> str:
        """test the connection to  api server"""
        test_message = "Hello, please respond with 'Connection test successful!'"
        try:
            response = await self.openai_client.chat_completion(test_message, use_history=False)
            if "successful" in response.lower() or "test" in response.lower():
                return f"✓ Connection test successful!\nResponse: {response}"
            else:
                return f"⚠ Connection works but unexpected response:\n{response}"
        except Exception as e:
            return f"✗ Connection test failed: {e}"

    def _show_help(self) -> str:
        """Show help page"""
        return """OpenAI Session Commands:

Basic Usage:
  Just type your message to chat with the AI

Special Commands:
  /help               - Show this help
  /config             - Show current configuration
  /config key=value   - Change configuration (e.g., /config temperature=0.5)
  /clear              - Clear conversation history
  /history            - Show conversation summary
  /model [name]       - Show/change model
  /models             - List available models
  /test               - Test API connection
  /oneshot <msg>      - Single request without history
  /system [prompt]    - Show/change system prompt
  /hidethinking [0|1] - Hide <think> blocks (0=off, 1=on)
  /latex [0|1]        - Convert Markdown to LaTeX via pandoc (0=off, 1=on)

Configuration Keys:
  api_base, api_key, model, temperature, max_tokens, hidethinking, latex_output

Examples:
  /config model=gpt-4-turbo
  /hidethinking 1
  /latex 1
"""

    async def cleanup(self):
        """Cleanup alle Ressourcen"""
        await self.openai_client.cleanup()

###############################################################################
## Session Main Loop
###############################################################################

async def main():
    """main loop for OpenAI Session"""
    flush_verbatim("OpenAI Session for TeXmacs")
    flush_newline()
    flush_verbatim(f"Python {platform.python_version()} [{sys.executable}]")
    flush_newline()
    if pypandoc is None:
        flush_verbatim("WARNING: pypandoc not found. /latex command will not work.")
        flush_newline()
    flush_verbatim("Type /help for available commands")
    flush_newline(2)

    config = OpenAIConfig.load()
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
                response = await processor.process_command(line)
                if processor.config.latex_output == 1:
                    flush_latex(response)
                else:
                    flush_verbatim(response)
                flush_newline(2)

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
