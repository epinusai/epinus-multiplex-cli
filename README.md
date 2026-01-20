# DeepSeek CLI

A Claude Code-style terminal agent for DeepSeek AI. Run the DeepSeek 671B model directly from your terminal with agentic capabilities.

## Features

- **Agentic Execution** - Automatically runs commands and creates files
- **Session Memory** - Resume previous conversations with `--resume`
- **Smart Polling** - Detects server/build commands and polls logs for success
- **Rich UI** - Syntax highlighted code, colored output
- **Auto-fix** - Automatically attempts to fix failed commands

## Installation

**Recommended (using pipx):**
```bash
# Install pipx if you don't have it
brew install pipx
pipx ensurepath

# Install deepseek-cli
pipx install git+https://github.com/epinusai/epinus-deepseek-v3.1671bil.git
```

**Or install from source:**
```bash
git clone https://github.com/epinusai/epinus-deepseek-v3.1671bil
cd epinus-deepseek-v3.1671bil
pip install -e .
```

## Prerequisites

1. **Install Ollama** - https://ollama.com/download
2. **Sign in to Ollama** for cloud model access:
   ```bash
   ollama signin
   ```
3. **Pull the DeepSeek model**:
   ```bash
   ollama run deepseek-v3.1:671b-cloud "hello"
   ```

## Usage

```bash
# Interactive mode
deepseek

# Quick prompt
deepseek "create a hello world python script"

# Resume last session
deepseek --resume

# Auto-execute commands without confirmation
deepseek --auto

# Specify working directory
deepseek --dir /path/to/project
```

## Commands

Inside the CLI, use these slash commands:

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/status` | Check last task status |
| `/kill` | Kill background task |
| `/auto` | Toggle auto-execute mode |
| `/cd <path>` | Change directory |
| `/ls` | List files |
| `/run <cmd>` | Run command directly |
| `/sessions` | List saved sessions |
| `/clear` | Clear session history |
| `/model <name>` | Change model |
| `/exit` | Quit |

## How It Works

1. **You type a request** - "create a Next.js app with a login page"
2. **DeepSeek generates code and commands** - Shows you what it wants to do
3. **You approve or skip** - Press Enter to run, or choose to skip
4. **Auto-continues** - If there are more steps, it continues automatically
5. **Polls long-running tasks** - For `npm run dev`, it polls logs and detects when server is ready

## Configuration

Config stored in `~/.dsk/config.json`:

```json
{
  "model": "deepseek-v3.1:671b-cloud",
  "max_tokens": 4096,
  "temperature": 0.7,
  "auto_execute": false
}
```

## License

MIT License - see [LICENSE](LICENSE)
