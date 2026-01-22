#!/usr/bin/env python3
"""
DeepSeek CLI - Terminal Agent for DeepSeek AI
Claude Code style terminal interface with inline execution + rich UI.

https://github.com/epinusai/epinus-deepseek-v3.1671bil
"""

import sys
import os
import json
import argparse
import subprocess
import time
import sqlite3
import re
import threading
from pathlib import Path
from datetime import datetime

# Force UTF-8 and ANSI colors for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except (OSError, AttributeError, ValueError):
        pass

# Rich for terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Ollama SDK
try:
    import ollama
    OLLAMA_SDK = True
except ImportError:
    OLLAMA_SDK = False

# Groq SDK
try:
    from groq import Groq
    GROQ_SDK = True
except ImportError:
    GROQ_SDK = False

# Anthropic SDK
try:
    import anthropic
    ANTHROPIC_SDK = True
except ImportError:
    ANTHROPIC_SDK = False

# Prompt toolkit for input & menus
try:
    from prompt_toolkit import prompt as pt_prompt, PromptSession
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.styles import Style as PTStyle
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout, ScrollablePane
    from prompt_toolkit.layout.containers import Window, HSplit, VSplit, WindowAlign
    from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.formatted_text import HTML, ANSI
    from prompt_toolkit.widgets import TextArea, Frame
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    PROMPT_TOOLKIT = True
except ImportError:
    PROMPT_TOOLKIT = False

console = Console() if RICH_AVAILABLE else None

# Unicode symbols with ASCII fallbacks for Windows compatibility
def _supports_unicode():
    """Check if terminal supports Unicode"""
    if sys.platform != "win32":
        return True
    try:
        # Try to encode a test character
        "\u2713".encode(sys.stdout.encoding or 'utf-8')
        return True
    except (UnicodeEncodeError, LookupError):
        return False

_UNICODE = _supports_unicode()

# Symbol mappings: (unicode, ascii_fallback)
SYM = {
    'arrow': ('>', '>'),           # prompt arrow (was ▸)
    'check': ('+', '+'),           # success (was ✓)
    'cross': ('x', 'x'),           # failure (was ✗)
    'vline': ('|', '|'),           # vertical line (was │)
    'warn': ('!', '!'),            # warning (was ⚠)
    'diamond': ('*', '*'),         # action marker (was ◆)
    'loop': ('~', '~'),            # auto-continue (was ↻)
    'corner': ('->', '->'),        # skip indicator (was ↳)
    'updown': ('Up/Down', 'Up/Down'),  # navigation hint (was ↑↓)
}

def sym(name):
    """Get symbol by name, using ASCII fallback if needed"""
    return SYM.get(name, ('?', '?'))[0 if _UNICODE else 1]

CONFIG_DIR = Path.home() / ".dsk"
CONFIG_FILE = CONFIG_DIR / "config.json"
SESSIONS_DIR = CONFIG_DIR / "sessions"
LEARNED_DIR = CONFIG_DIR / "learned"  # Continuous learning patterns
HOOKS_FILE = CONFIG_DIR / "hooks.json"  # Custom hooks config
STATE_FILE = CONFIG_DIR / "last_state.json"  # Memory persistence

DEFAULT_CONFIG = {
    "provider": "ollama",  # ollama, groq, or anthropic
    "model": "deepseek-v3.1:671b-cloud",
    "groq_model": "compound-beta",  # Groq compound model
    "anthropic_model": "claude-sonnet-4-20250514",  # Anthropic model
    "max_tokens": 4096,
    "temperature": 0.7,
    "auto_execute": False,
    "mode": "normal",  # normal, concise, explain, review, tdd, plan, security
    # Hook settings
    "hooks_enabled": True,
    "auto_format": True,  # Auto-format JS/TS files after edit
    "warn_console_log": True,  # Warn about console.log in code
    "strategic_compact": True,  # Suggest compaction at logical points
    # Learning settings
    "continuous_learning": True,  # Extract patterns from sessions
}

# Agent mode prompts (specialized system prompts for different tasks)
AGENT_MODES = {
    "normal": "",  # Default - no extra instructions
    "review": """You are in CODE REVIEW mode. Focus on:
- Code quality and readability
- Security vulnerabilities (OWASP top 10)
- Performance issues (O(n²), N+1 queries, memory leaks)
- Best practices violations
- Missing error handling
- console.log statements that should be removed

Provide feedback as: CRITICAL (must fix), WARNING (should fix), SUGGESTION (consider).
Include specific line numbers and fix examples.""",

    "tdd": """You are in TDD (Test-Driven Development) mode. Follow the RED-GREEN-REFACTOR cycle:
1. SCAFFOLD - Define interfaces/types first
2. RED - Write failing tests BEFORE implementation
3. GREEN - Write minimal code to pass tests
4. REFACTOR - Improve code while keeping tests green

NEVER write implementation before tests. Always run tests after each step.
Target 80%+ code coverage. 100% for critical business logic.""",

    "plan": """You are in PLANNING mode. Before implementing:
1. Understand the full scope of the task
2. Identify affected files and dependencies
3. Consider edge cases and error scenarios
4. Break down into small, testable steps
5. Identify potential risks or blockers

Output a clear implementation plan with numbered steps.
Do NOT write code until the plan is approved.""",

    "security": """You are in SECURITY REVIEW mode. Check for:
- Hardcoded credentials (API keys, passwords, tokens)
- SQL injection (string concatenation in queries)
- XSS vulnerabilities (unescaped user input)
- Path traversal (user-controlled file paths)
- CSRF vulnerabilities
- Authentication/authorization bypasses
- Insecure dependencies
- Missing input validation
- Sensitive data exposure in logs/errors

Flag all issues with severity: CRITICAL, HIGH, MEDIUM, LOW.""",

    "refactor": """You are in REFACTOR mode. Focus on:
- Extract repeated code into functions
- Simplify complex conditionals
- Reduce function/file size (target <50 lines per function, <400 lines per file)
- Improve naming (clear, descriptive, consistent)
- Remove dead code
- Add missing type annotations

Keep existing tests passing. Do NOT change behavior.""",
}

# Groq models
GROQ_MODELS = [
    "compound-beta",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# Anthropic models
ANTHROPIC_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]


class Spinner:
    """Animated spinner for visual feedback"""
    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    FRAMES_ASCII = ['|', '/', '-', '\\']

    def __init__(self, message="Thinking"):
        self.message = message
        self.running = False
        self.thread = None
        self.frames = self.FRAMES if _UNICODE else self.FRAMES_ASCII

    def _spin(self):
        i = 0
        while self.running:
            frame = self.frames[i % len(self.frames)]
            print(f"\033[2m{frame} {self.message}...\033[0m", end="\r", flush=True)
            time.sleep(0.1)
            i += 1

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        print(" " * 40, end="\r", flush=True)  # Clear line


def estimate_tokens(text):
    """Estimate token count (rough: ~4 chars per token for English)"""
    if not text:
        return 0
    return len(text) // 4 + 1


def format_tokens(count):
    """Format token count with K suffix"""
    if count >= 1000:
        return f"{count/1000:.1f}k"
    return str(count)


def get_model_context_limit(model_name):
    """Get context window size for a model"""
    # Try to get from ollama
    try:
        result = subprocess.run(
            ['ollama', 'show', model_name],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'context length' in line.lower():
                    # Extract number from line like "    context length      163840"
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            return int(part)
    except:
        pass

    # Fallback defaults by model name
    model_lower = model_name.lower()

    # Anthropic models
    if 'claude' in model_lower:
        return 200000  # 200k context

    # Groq models
    if 'compound' in model_lower:
        return 131072  # 128k
    elif 'llama-3.3' in model_lower or 'llama-3.1' in model_lower:
        return 131072  # 128k
    elif 'mixtral' in model_lower:
        return 32768  # 32k
    elif 'gemma' in model_lower:
        return 8192  # 8k

    # Ollama models
    if 'deepseek' in model_lower:
        return 163840  # 160k
    elif 'gpt-oss' in model_lower:
        return 131072  # 128k
    elif 'qwen' in model_lower:
        return 131072  # 128k
    else:
        return 128000  # Safe default


class DSK:
    def __init__(self, working_dir=None, session_id=None):
        CONFIG_DIR.mkdir(exist_ok=True)
        SESSIONS_DIR.mkdir(exist_ok=True)
        LEARNED_DIR.mkdir(exist_ok=True)

        self.config = self._load_config()
        self.hooks = self._load_hooks()
        self.working_dir = working_dir or os.getcwd()
        os.chdir(self.working_dir)

        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_db = SESSIONS_DIR / f"session_{self.session_id}.db"
        self.session_messages = []
        self._init_db()

        # Token tracking
        self.tokens_in = 0
        self.tokens_out = 0

        # Compaction state (Claude Code style)
        self._compacted_summary = None  # State summary after compaction
        self._recent_turns = 10  # Keep last N turns after compaction
        self._compaction_threshold = 0.70  # Trigger at 70% of context

        # Action output buffering (for collapsed view)
        self._action_buffer = []  # Stores action output lines
        self._buffering_actions = False  # Flag to buffer instead of print
        self._last_actions_output = []  # Last completed action set

        # API clients (initialized on demand)
        self._groq_client = None
        self._anthropic_client = None

        # Strategic compaction tracking
        self._tool_calls_count = 0  # Track tool calls for strategic compaction
        self._strategic_compact_threshold = 50  # Suggest after N tool calls
        self._last_compact_suggestion = 0  # Last tool count when suggested

        # Learned patterns (continuous learning)
        self._learned_patterns = self._load_learned_patterns()

        # Load previous session state if available
        if session_id:
            self._load_session()
        else:
            self._load_last_state()  # Memory persistence

    def _get_groq_client(self):
        """Get or create Groq client"""
        if self._groq_client is None and GROQ_SDK:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                self._print("[red]GROQ_API_KEY not set. Export it: export GROQ_API_KEY=your_key[/red]")
                return None
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    def _get_anthropic_client(self):
        """Get or create Anthropic client"""
        if self._anthropic_client is None and ANTHROPIC_SDK:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                self._print("[red]ANTHROPIC_API_KEY not set. Export it: export ANTHROPIC_API_KEY=your_key[/red]")
                return None
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def _load_config(self):
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        return DEFAULT_CONFIG.copy()

    def _save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)

    def _load_hooks(self):
        """Load custom hooks from hooks.json"""
        default_hooks = {
            "pre_command": [],  # Run before shell commands
            "post_command": [],  # Run after shell commands
            "pre_write": [],  # Run before file writes
            "post_write": [  # Run after file writes
                {"pattern": r"\.(js|jsx|ts|tsx)$", "action": "format_check"},
                {"pattern": r"\.(js|jsx|ts|tsx)$", "action": "console_log_warn"},
            ],
            "pre_compact": [],  # Run before compaction
            "post_compact": [],  # Run after compaction
        }
        if HOOKS_FILE.exists():
            try:
                with open(HOOKS_FILE) as f:
                    return {**default_hooks, **json.load(f)}
            except:
                pass
        return default_hooks

    def _load_learned_patterns(self):
        """Load learned patterns from continuous learning"""
        patterns = []
        if LEARNED_DIR.exists():
            for f in LEARNED_DIR.glob("*.json"):
                try:
                    with open(f) as fp:
                        patterns.append(json.load(fp))
                except:
                    pass
        return patterns

    def _save_learned_pattern(self, pattern):
        """Save a learned pattern for future sessions"""
        if not self.config.get("continuous_learning", True):
            return
        pattern_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        pattern_file = LEARNED_DIR / f"pattern_{pattern_id}.json"
        with open(pattern_file, "w") as f:
            json.dump(pattern, f, indent=2)
        self._learned_patterns.append(pattern)

    def _load_last_state(self):
        """Load previous session state (memory persistence)"""
        if not STATE_FILE.exists():
            return
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            # Check if state is recent (within 24 hours)
            state_time = datetime.fromisoformat(state.get("timestamp", "2000-01-01"))
            if (datetime.now() - state_time).total_seconds() > 86400:
                return  # State too old
            # Offer to restore
            self._print(f"\n[dim]Found previous session state from {state_time.strftime('%H:%M')}[/dim]")
            if state.get("summary"):
                self._print(f"[dim]Context: {state['summary'][:100]}...[/dim]")
            self._compacted_summary = state.get("summary")
        except Exception as e:
            pass

    def _save_last_state(self):
        """Save session state before exit (memory persistence)"""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "working_dir": self.working_dir,
                "summary": self._compacted_summary,
                "mode": self.config.get("mode", "normal"),
                "recent_files": self._get_recent_files(),
                "task_context": self._extract_task_context(),
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except:
            pass

    def _get_recent_files(self):
        """Get list of recently modified files in session"""
        files = []
        for msg in self.session_messages[-20:]:
            content = msg.get("content", "")
            # Extract file paths from write/edit actions
            for match in re.findall(r'(?:Wrote|Edited|Created|Modified)\s+([^\s\n]+)', content):
                if match not in files:
                    files.append(match)
        return files[-10:]  # Last 10 files

    def _extract_task_context(self):
        """Extract current task context from conversation"""
        if len(self.session_messages) < 2:
            return ""
        # Get first user message as likely task description
        for msg in self.session_messages[:5]:
            if msg.get("role") == "user":
                return msg.get("content", "")[:500]
        return ""

    def _run_hook(self, hook_type, context=None):
        """Run hooks of specified type"""
        if not self.config.get("hooks_enabled", True):
            return True  # Hooks disabled, allow action
        hooks = self.hooks.get(hook_type, [])
        for hook in hooks:
            pattern = hook.get("pattern", ".*")
            action = hook.get("action", "")
            target = context.get("target", "") if context else ""
            if re.search(pattern, target):
                result = self._execute_hook_action(action, context)
                if result is False:
                    return False  # Hook blocked the action
        return True

    def _execute_hook_action(self, action, context):
        """Execute a specific hook action"""
        if action == "format_check":
            return self._hook_format_check(context)
        elif action == "console_log_warn":
            return self._hook_console_log_warn(context)
        elif action == "block":
            self._print(f"[yellow]{sym('warn')} Hook blocked action[/yellow]")
            return False
        return True

    def _hook_format_check(self, context):
        """Check if file needs formatting (hook action)"""
        if not self.config.get("auto_format", True):
            return True
        filepath = context.get("target", "")
        if not filepath or not os.path.exists(filepath):
            return True
        # Check if prettier is available
        try:
            result = subprocess.run(
                ["prettier", "--check", filepath],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                self._print(f"[dim]  {sym('arrow')} Formatting with prettier...[/dim]")
                subprocess.run(["prettier", "--write", filepath], capture_output=True, timeout=10)
        except FileNotFoundError:
            pass  # Prettier not installed
        except:
            pass
        return True

    def _hook_console_log_warn(self, context):
        """Warn about console.log statements (hook action)"""
        if not self.config.get("warn_console_log", True):
            return True
        filepath = context.get("target", "")
        if not filepath or not os.path.exists(filepath):
            return True
        try:
            with open(filepath, "r") as f:
                content = f.read()
            matches = list(re.finditer(r'console\.log\s*\(', content))
            if matches:
                lines = content[:matches[0].start()].count('\n') + 1
                self._print(f"[yellow]  {sym('warn')} console.log found at line {lines} - remove before commit[/yellow]")
        except:
            pass
        return True

    def _suggest_strategic_compact(self):
        """Suggest compaction at strategic points (not arbitrary threshold)"""
        if not self.config.get("strategic_compact", True):
            return
        self._tool_calls_count += 1
        # Suggest at threshold and every 25 calls after
        if self._tool_calls_count >= self._strategic_compact_threshold:
            if self._tool_calls_count - self._last_compact_suggestion >= 25:
                self._last_compact_suggestion = self._tool_calls_count
                self._print(f"\n[dim]{sym('warn')} Consider /compact - {self._tool_calls_count} tool calls in session[/dim]")

    def _extract_session_patterns(self):
        """Extract reusable patterns from session (continuous learning)"""
        if not self.config.get("continuous_learning", True):
            return
        if len(self.session_messages) < 10:
            return  # Session too short

        # Look for error->fix patterns
        patterns_found = []
        for i, msg in enumerate(self.session_messages[:-1]):
            content = msg.get("content", "")
            next_content = self.session_messages[i + 1].get("content", "") if i + 1 < len(self.session_messages) else ""

            # Pattern: Error followed by fix
            if "error" in content.lower() or "failed" in content.lower():
                if "fixed" in next_content.lower() or "resolved" in next_content.lower():
                    patterns_found.append({
                        "type": "error_resolution",
                        "error_context": content[:200],
                        "resolution": next_content[:200],
                        "timestamp": datetime.now().isoformat(),
                    })

        # Save unique patterns
        for pattern in patterns_found[:3]:  # Max 3 patterns per session
            self._save_learned_pattern(pattern)

        if patterns_found:
            self._print(f"[dim]{sym('check')} Learned {len(patterns_found)} pattern(s) for future sessions[/dim]")

    def _init_db(self):
        conn = sqlite3.connect(self.session_db)
        conn.execute('''CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY, role TEXT, content TEXT,
            timestamp TEXT, seq INTEGER)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY, value TEXT)''')
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('active', '1')")
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('working_dir', ?)", (self.working_dir,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('model', ?)", (self.config['model'],))
        conn.commit()
        conn.close()

    def _save_message(self, role, content):
        conn = sqlite3.connect(self.session_db)
        cur = conn.cursor()
        cur.execute("SELECT MAX(seq) FROM messages")
        seq = (cur.fetchone()[0] or 0) + 1
        cur.execute("INSERT INTO messages VALUES (NULL,?,?,?,?)",
                   (role, content, datetime.now().isoformat(), seq))
        conn.commit()
        conn.close()
        self.session_messages.append({"role": role, "content": content})

    def _load_session(self):
        if not self.session_db.exists():
            return
        conn = sqlite3.connect(self.session_db)
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM messages ORDER BY seq")
        self.session_messages = [{"role": r, "content": c} for r, c in cur.fetchall()]
        cur.execute("SELECT value FROM meta WHERE key='working_dir'")
        row = cur.fetchone()
        if row and os.path.exists(row[0]):
            self.working_dir = row[0]
            os.chdir(row[0])
        conn.close()
        self._print(f"[green]{sym('check')} Resumed session with {len(self.session_messages)} messages[/green]")

    def _close_session(self):
        """Close session with memory persistence and learning"""
        # Save session state for next time (memory persistence)
        self._save_last_state()

        # Extract patterns from session (continuous learning)
        self._extract_session_patterns()

        # Close database
        try:
            conn = sqlite3.connect(self.session_db)
            conn.execute("UPDATE meta SET value='0' WHERE key='active'")
            conn.commit()
            conn.close()
        except sqlite3.Error:
            pass

    def _needs_compaction(self):
        """Check if context needs compaction (Claude Code style)"""
        if self._compacted_summary:
            # Already compacted - check if we need to re-compact
            recent_text = json.dumps(self.session_messages[-self._recent_turns * 2:])
            recent_tokens = estimate_tokens(recent_text) + estimate_tokens(self._compacted_summary)
        else:
            # Not compacted yet - check full context
            full_text = json.dumps(self.session_messages)
            recent_tokens = estimate_tokens(full_text)

        # Get context limit based on provider
        provider = self.config.get("provider", "ollama")
        if provider == "groq":
            current_model = self.config.get("groq_model", "compound-beta")
        elif provider == "anthropic":
            current_model = self.config.get("anthropic_model", "claude-sonnet-4-20250514")
        else:
            current_model = self.config["model"]
        context_limit = get_model_context_limit(current_model)
        return recent_tokens > (context_limit * self._compaction_threshold)

    def _generate_state_summary(self):
        """Generate a state summary using AI (not chat summary - task/state summary)"""
        self._print(f"\n[yellow]{sym('warn')} Context at threshold - compacting...[/yellow]")

        # Build prompt for state summary
        summary_prompt = """Summarize the current state of this conversation for context continuity.
Include:
- Current goal/task
- Key decisions made
- Important constraints or requirements
- Unresolved questions
- Critical context needed to continue

Be concise but complete. This is a STATE summary, not a chat summary.
Format as bullet points."""

        # Include previous summary if already compacted (so context compounds, not lost)
        if self._compacted_summary:
            conversation_context = f"PREVIOUS STATE SUMMARY:\n{self._compacted_summary}\n\nRECENT CONVERSATION:\n{json.dumps(self.session_messages, indent=2)}"
        else:
            conversation_context = json.dumps(self.session_messages, indent=2)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversation state."},
            {"role": "user", "content": f"Here is the conversation to summarize:\n\n{conversation_context}\n\n{summary_prompt}"}
        ]

        try:
            provider = self.config.get("provider", "ollama")

            if provider == "anthropic" and ANTHROPIC_SDK:
                client = self._get_anthropic_client()
                if client:
                    # Anthropic uses system as separate param
                    response = client.messages.create(
                        model=self.config.get("anthropic_model", "claude-sonnet-4-20250514"),
                        max_tokens=2048,
                        system=messages[0]["content"],
                        messages=[messages[1]],
                    )
                    return response.content[0].text

            elif provider == "groq" and GROQ_SDK:
                client = self._get_groq_client()
                if client:
                    response = client.chat.completions.create(
                        model=self.config.get("groq_model", "compound-beta"),
                        messages=messages,
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    return response.choices[0].message.content

            elif OLLAMA_SDK:
                response = ollama.chat(
                    model=self.config["model"],
                    messages=messages,
                    stream=False,
                    options={"temperature": 0.3}
                )
                return response.message.content

        except Exception as e:
            self._print(f"[red]{sym('cross')} Compaction failed: {e}[/red]")
            return None

        return None

    def _compact_context(self):
        """Perform context compaction (Claude Code style)"""
        old_count = len(self.session_messages)

        # Step 1: Try to generate state summary
        summary = self._generate_state_summary()

        if summary:
            # Step 2: Store the summary
            self._compacted_summary = summary
            self._print(f"[green]{sym('check')} Compacted: {old_count} msgs → summary + recent[/green]")
        else:
            # Summary failed (rate limit?) - force truncate without summary
            self._print(f"[yellow]{sym('warn')} Summary failed - force truncating to recent turns[/yellow]")
            if self._compacted_summary:
                # Keep existing summary
                self._print(f"[dim]  Keeping previous summary[/dim]")
            else:
                # No summary - just note context was truncated
                self._compacted_summary = "[Context truncated due to API limits - some earlier context may be missing]"

        # Step 3: Keep only recent turns in active memory
        # (Full history still in SQLite!)
        if len(self.session_messages) > self._recent_turns * 2:
            self.session_messages = self.session_messages[-self._recent_turns * 2:]

        # Step 4: Reset token counters for new context window
        self.tokens_in = estimate_tokens(self._compacted_summary) if self._compacted_summary else 0
        self.tokens_out = 0

        self._print(f"[dim]  Now: {len(self.session_messages)} msgs | Full history in SQLite[/dim]")
        return True

    def _get_messages_for_context(self):
        """Get messages to send to model (summary + recent turns if compacted)"""
        if self._compacted_summary:
            # After compaction: summary + recent turns
            summary_msg = {"role": "system", "content": f"[CONTEXT SUMMARY]\n{self._compacted_summary}\n[END SUMMARY]"}
            return [summary_msg] + self.session_messages[-self._recent_turns * 2:]
        else:
            # Before compaction: full history
            return self.session_messages

    @staticmethod
    def get_active_session():
        """Find crashed/active session"""
        for f in sorted(SESSIONS_DIR.glob("*.db"), reverse=True):
            try:
                conn = sqlite3.connect(f)
                cur = conn.cursor()
                cur.execute("SELECT value FROM meta WHERE key='active'")
                row = cur.fetchone()
                conn.close()
                if row and row[0] == '1':
                    return f.stem.replace("session_", "")
            except sqlite3.Error:
                pass
        return None

    @staticmethod
    def list_sessions():
        """List all saved sessions"""
        sessions = []
        for f in sorted(SESSIONS_DIR.glob("*.db"), reverse=True)[:10]:
            try:
                conn = sqlite3.connect(f)
                cur = conn.cursor()
                cur.execute("SELECT value FROM meta WHERE key='working_dir'")
                wd = cur.fetchone()
                cur.execute("SELECT COUNT(*) FROM messages")
                count = cur.fetchone()[0]
                cur.execute("SELECT content FROM messages WHERE role='user' ORDER BY seq LIMIT 1")
                preview = cur.fetchone()
                conn.close()
                sessions.append({
                    "id": f.stem.replace("session_", ""),
                    "working_dir": wd[0] if wd else "?",
                    "messages": count,
                    "preview": (preview[0][:50] + "...") if preview and len(preview[0]) > 50 else (preview[0] if preview else "")
                })
            except sqlite3.Error:
                pass
        return sessions

    def _print(self, text, style=None, force=False):
        """Print text, or buffer if in action buffering mode"""
        # Buffer action output if in buffering mode (unless forced)
        if self._buffering_actions and not force:
            self._action_buffer.append(text)
            return

        if RICH_AVAILABLE:
            console.print(text, style=style)
        else:
            print(text)

    def _select_menu(self, options, title=None):
        """Interactive arrow-key menu like Claude Code"""
        if not PROMPT_TOOLKIT:
            # Fallback to number selection
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
            try:
                choice = input("  Choice [1]: ").strip()
                return int(choice) - 1 if choice else 0
            except:
                return 0

        selected = [0]

        def get_text():
            lines = []
            for i, opt in enumerate(options):
                if i == selected[0]:
                    lines.append(f"<style bg='#333333' fg='#00ff00'> {sym('arrow')} {opt} </style>")
                else:
                    lines.append(f"<style fg='#666666'>   {opt}</style>")
            lines.append("")
            lines.append(f"<style fg='#444444'>{sym('updown')} select - enter confirm</style>")
            return HTML("\n".join(lines))

        kb = KeyBindings()

        @kb.add(Keys.Up)
        def up(e): selected[0] = (selected[0] - 1) % len(options)
        @kb.add(Keys.Down)
        def down(e): selected[0] = (selected[0] + 1) % len(options)
        @kb.add(Keys.Enter)
        def enter(e): e.app.exit(result=selected[0])
        @kb.add(Keys.Escape)
        def esc(e): e.app.exit(result=-1)

        try:
            app = Application(
                layout=Layout(Window(FormattedTextControl(get_text))),
                key_bindings=kb,
                full_screen=False
            )
            result = app.run()
            # Clear menu
            print(f"\033[{len(options)+2}A\033[J", end="")
            return result if result >= 0 else 0
        except (KeyboardInterrupt, EOFError, OSError):
            return 0

    def _ask_permission(self, action_desc):
        """Ask Y/N/A for actions with clear preview"""
        if self.config.get("auto_execute"):
            self._print(f"  [dim]{sym('arrow')} Auto:[/dim] [cyan]{action_desc}[/cyan]")
            return True

        self._print(f"\n  [bold yellow]Proposed:[/bold yellow] [cyan]{action_desc}[/cyan]")

        options = ["Run", "Run all (auto)", "Skip"]
        choice = self._select_menu(options)

        if choice == 0:
            return True
        elif choice == 1:
            self.config["auto_execute"] = True
            self._print(f"  [dim]Auto-execute enabled for this session[/dim]")
            return True
        return False

    # Patterns for long-running/polling commands (servers + builds + scaffolding)
    LONG_RUNNING_CMDS = ['npm run dev', 'npm start', 'yarn dev', 'yarn start',
                         'pnpm dev', 'pnpm start', 'next dev', 'vite',
                         'python -m http.server', 'flask run', 'uvicorn',
                         'node server', 'nodemon', 'ts-node', 'webpack serve',
                         # Build commands (can take a while)
                         'npm run build', 'yarn build', 'pnpm build', 'next build',
                         'webpack', 'tsc', 'npm run lint', 'yarn lint',
                         'npm test', 'yarn test', 'pytest', 'cargo build', 'go build',
                         # Scaffolding commands (slow to start)
                         'npx create-', 'npm create', 'yarn create', 'pnpm create',
                         'npm install', 'yarn install', 'pnpm install', 'pip install']

    # Success patterns indicating task is done/ready
    SUCCESS_PATTERNS = ['ready on', 'listening on', 'started at', 'server running',
                        'localhost:', '127.0.0.1:', 'compiled successfully',
                        'ready in', 'started server', 'accepting connections',
                        'webpack compiled', 'ready -', 'Local:',
                        # Build success patterns
                        '✓ Compiled', 'Build completed', 'Successfully compiled',
                        'Done in', 'Finished', 'built in', 'passed']

    # Error patterns
    ERROR_PATTERNS = ['error:', 'Error:', 'ERROR', 'failed to compile',
                      'FAILED', 'exception', 'Exception', 'EADDRINUSE', 'EACCES',
                      'Module not found', 'Cannot find', 'SyntaxError', 'TypeError']

    # Last task tracking
    _last_task = None  # {'cmd': str, 'process': Popen, 'output': [], 'status': str, 'start_time': float}

    def _is_long_running(self, cmd):
        """Check if command is a long-running server process"""
        cmd_lower = cmd.lower()

        # Split by && to check each command in a chain
        commands = [c.strip() for c in cmd_lower.split('&&')]

        for single_cmd in commands:
            # Skip cd commands at the start
            if single_cmd.startswith('cd '):
                continue

            # Check if the command starts with or matches a long-running pattern
            for pattern in self.LONG_RUNNING_CMDS:
                # For patterns like 'npm run dev', 'npx create-', check if command starts with it
                if single_cmd.startswith(pattern):
                    return True
                # For single-word patterns like 'vite', 'nodemon', check if it's the command itself
                if ' ' not in pattern:
                    # Match "vite" or "vite ..." but not "my-vite-test"
                    cmd_parts = single_cmd.split()
                    if cmd_parts and cmd_parts[0] == pattern:
                        return True

        return False

    def _run_command(self, cmd):
        """Execute command with live streaming output and stats"""
        # Always show which command is running (force=True)
        self._print(f"  [dim]$ {cmd[:70]}{'...' if len(cmd) > 70 else ''}[/dim]", force=True)

        is_server = self._is_long_running(cmd)
        if is_server:
            self._print(f"  [dim]{sym('vline')} Long-running command - polling mode[/dim]", force=True)
            return self._run_server_command(cmd)

        start_time = time.time()
        spinner = Spinner("Running")
        spinner.start()

        try:
            process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=self.working_dir, bufsize=1
            )

            output_lines = []
            first_output = True
            for line in iter(process.stdout.readline, ''):
                if first_output:
                    spinner.stop()
                    first_output = False
                line = line.rstrip()
                output_lines.append(line)
                elapsed = time.time() - start_time
                # Show live output with stats
                if len(output_lines) <= 12:
                    self._print(f"  [dim]{sym('vline')}[/dim] {line}")
                elif len(output_lines) == 13:
                    self._print(f"  [dim]{sym('vline')} ... streaming ({elapsed:.1f}s)[/dim]")

            process.wait()
            spinner.stop()
            elapsed = time.time() - start_time

            # Summary stats
            if len(output_lines) > 12:
                self._print(f"  [dim]{sym('vline')} {len(output_lines)} lines in {elapsed:.1f}s[/dim]")

            # Always show success/fail status (force=True)
            if process.returncode == 0:
                self._print(f"  [green]{sym('check')} Done ({elapsed:.1f}s)[/green]", force=True)
            else:
                self._print(f"  [red]{sym('cross')} Failed (exit {process.returncode}, {elapsed:.1f}s)[/red]", force=True)

            return '\n'.join(output_lines), process.returncode

        except KeyboardInterrupt:
            spinner.stop()
            process.kill()
            self._print(f"  [yellow]{sym('warn')} Interrupted by user[/yellow]", force=True)
            return "INTERRUPTED", -999  # Special code for user interruption
        except Exception as e:
            spinner.stop()
            self._print(f"  [red]{sym('cross')} Error: {e}[/red]", force=True)
            return str(e), -1

    def _run_server_command(self, cmd):
        """Run a long-running server command with polling"""
        import fcntl
        import os as _os

        start_time = time.time()
        max_wait = 30  # Max seconds to wait for server ready
        poll_interval = 3  # Poll every 3 seconds

        try:
            # Start process in background
            process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=self.working_dir, bufsize=1
            )

            output_lines = []
            success_detected = False
            error_detected = False

            # Store task info for status tracking
            DSK._last_task = {
                'cmd': cmd,
                'process': process,
                'output': output_lines,
                'status': 'running',
                'start_time': start_time,
                'success_detected': False,
                'error_detected': False
            }

            self._print(f"  [dim]{sym('vline')} Polling logs every {poll_interval}s (max {max_wait}s)...[/dim]")

            while time.time() - start_time < max_wait:
                # Set non-blocking
                fd = process.stdout.fileno()
                fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, fl | _os.O_NONBLOCK)

                # Read available output
                try:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        line = line.rstrip()
                        output_lines.append(line)

                        # Show output
                        if len(output_lines) <= 15:
                            self._print(f"  [dim]{sym('vline')}[/dim] {line}")

                        # Check for success patterns
                        line_lower = line.lower()
                        if any(p.lower() in line_lower for p in self.SUCCESS_PATTERNS):
                            success_detected = True
                            DSK._last_task['success_detected'] = True

                        # Check for error patterns
                        if any(p.lower() in line_lower for p in self.ERROR_PATTERNS):
                            error_detected = True
                            DSK._last_task['error_detected'] = True
                except (IOError, BlockingIOError):
                    pass  # No more data available

                # Update task status
                DSK._last_task['output'] = output_lines

                # Check if process exited
                if process.poll() is not None:
                    DSK._last_task['status'] = 'exited'
                    if process.returncode != 0:
                        error_detected = True
                    elif success_detected:
                        DSK._last_task['status'] = 'completed'
                    break

                # If success detected, we're done
                if success_detected and not error_detected:
                    elapsed = time.time() - start_time
                    DSK._last_task['status'] = 'running_ok'
                    self._print(f"  [green]{sym('check')} Task running ({elapsed:.1f}s) - continuing in background[/green]")
                    return '\n'.join(output_lines), 0

                # If error detected, report failure
                if error_detected:
                    elapsed = time.time() - start_time
                    DSK._last_task['status'] = 'error'
                    self._print(f"  [red]{sym('cross')} Error detected ({elapsed:.1f}s)[/red]")
                    process.kill()
                    return '\n'.join(output_lines), 1

                # Wait before next poll
                time.sleep(poll_interval)
                self._print(f"  [dim]{sym('vline')} ... polling ({time.time() - start_time:.0f}s)[/dim]")

            # Timeout reached - assume it's running if no errors
            elapsed = time.time() - start_time
            if not error_detected:
                DSK._last_task['status'] = 'running_ok'
                self._print(f"  [yellow]{sym('warn')} Timeout ({elapsed:.1f}s) - assuming task is running[/yellow]")
                return '\n'.join(output_lines), 0
            else:
                DSK._last_task['status'] = 'error'
                self._print(f"  [red]{sym('cross')} Failed after {elapsed:.1f}s[/red]")
                process.kill()
                return '\n'.join(output_lines), 1

        except KeyboardInterrupt:
            # Don't kill - keep running, just stop polling
            DSK._last_task['status'] = 'running_background'
            self._print(f"  [yellow]{sym('warn')} Stopped polling - task continues in background[/yellow]")
            self._print(f"  [dim]{sym('vline')} Use /status to check task status[/dim]")
            return '\n'.join(output_lines), -999  # Special code for user interruption
        except Exception as e:
            if DSK._last_task:
                DSK._last_task['status'] = 'error'
            self._print(f"  [red]{sym('cross')} Error: {e}[/red]")
            return str(e), -1

    def _get_task_status(self):
        """Get status of the last/background task"""
        if not DSK._last_task:
            return None, "No recent task"

        task = DSK._last_task
        process = task.get('process')
        output = task.get('output', [])
        elapsed = time.time() - task.get('start_time', time.time())

        # Check if process is still running
        if process and process.poll() is None:
            # Still running - try to read more output
            import fcntl
            import os as _os
            try:
                fd = process.stdout.fileno()
                fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, fl | _os.O_NONBLOCK)
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    line = line.rstrip()
                    output.append(line)
                    # Check patterns
                    line_lower = line.lower()
                    if any(p.lower() in line_lower for p in self.SUCCESS_PATTERNS):
                        task['success_detected'] = True
                    if any(p.lower() in line_lower for p in self.ERROR_PATTERNS):
                        task['error_detected'] = True
            except (IOError, BlockingIOError):
                pass

            if task.get('success_detected') and not task.get('error_detected'):
                return 'running_ok', f"Task running OK ({elapsed:.0f}s)\nCmd: {task['cmd']}\nLast output: {output[-1] if output else 'none'}"
            elif task.get('error_detected'):
                return 'error', f"Task has errors ({elapsed:.0f}s)\nCmd: {task['cmd']}\nLast output: {output[-1] if output else 'none'}"
            else:
                return 'running', f"Task running ({elapsed:.0f}s)\nCmd: {task['cmd']}\nLast output: {output[-1] if output else 'none'}"
        else:
            # Process exited
            returncode = process.returncode if process else -1
            if returncode == 0 or task.get('success_detected'):
                return 'completed', f"Task completed successfully\nCmd: {task['cmd']}\nLast output: {output[-1] if output else 'none'}"
            else:
                return 'failed', f"Task failed (exit {returncode})\nCmd: {task['cmd']}\nLast output: {output[-1] if output else 'none'}"

    def _write_file(self, filepath, content):
        """Write file with diff-style preview"""
        full_path = os.path.join(self.working_dir, filepath)
        is_new = not os.path.exists(full_path)

        # Always show file header (force=True)
        self._print(f"\n  [bold {'green' if is_new else 'yellow'}]{'+ New' if is_new else '~ Edit'}:[/bold {'green' if is_new else 'yellow'}] {filepath}", force=True)

        # File content preview (can be collapsed)
        lines = content.split('\n')
        for i, line in enumerate(lines[:10], 1):
            self._print(f"  [dim]{i:3}[/dim] [{'green' if is_new else 'yellow'}]{sym('vline')} {line}[/{'green' if is_new else 'yellow'}]")
        if len(lines) > 10:
            self._print(f"  [dim]    {sym('vline')} ... ({len(lines) - 10} more lines)[/dim]")

        if not self._ask_permission(f"Write {len(lines)} lines to {filepath}"):
            self._print(f"  [dim]{sym('corner')} Skipped[/dim]", force=True)
            return False

        os.makedirs(os.path.dirname(full_path) or '.', exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Always show result (force=True)
        self._print(f"  [green]{sym('check')} Wrote {filepath}[/green]", force=True)

        # Run post-write hooks (format, lint warnings)
        self._run_hook("post_write", {"target": full_path, "content": content})
        self._suggest_strategic_compact()

        return True

    def _get_system_prompt(self, user_query=""):
        cwd = os.getcwd()
        try:
            files_list = ', '.join(sorted(os.listdir(cwd))[:20])
        except OSError:
            files_list = "(unable to list)"

        # Check if user is asking about task status
        status_keywords = ['status', 'what happened', 'how is it', 'is it running',
                          'did it work', 'is it done', 'check', 'last task', 'background']
        asking_status = any(kw in user_query.lower() for kw in status_keywords)

        # Get last task status if relevant
        task_context = ""
        if DSK._last_task or asking_status:
            status, msg = self._get_task_status()
            if status:
                task_context = f"""
LAST TASK STATUS:
- Status: {status}
- {msg}
"""

        # Mode-specific instructions
        mode = self.config.get("mode", "normal")

        # Check for agent modes (review, tdd, plan, security, refactor)
        if mode in AGENT_MODES and AGENT_MODES[mode]:
            mode_rules = f"AGENT MODE: {mode.upper()}\n{AGENT_MODES[mode]}"
        elif mode == "concise":
            mode_rules = """STYLE: CONCISE MODE
- Minimal explanation, maximum action
- Just show code/commands, skip the commentary
- One-line responses when possible"""
        elif mode == "explain":
            mode_rules = """STYLE: EXPLAIN MODE
- Teaching mode: explain WHY, not just WHAT
- Break down complex concepts
- Help user learn from each step"""
        else:
            mode_rules = """STYLE: NORMAL MODE
- Brief explanations, then action
- Balance speed with clarity"""

        # Add learned patterns context if any
        learned_context = ""
        if self._learned_patterns:
            recent_patterns = self._learned_patterns[-3:]  # Last 3 patterns
            learned_context = "\n\nLEARNED PATTERNS (from previous sessions):\n"
            for p in recent_patterns:
                if p.get("type") == "error_resolution":
                    learned_context += f"- Error pattern: {p.get('error_context', '')[:50]}...\n"

        return f"""You are DeepSeek CLI - an AI agent that bootstraps, refactors, and debugs real projects inside your repo.

CONTEXT:
- Directory: {cwd}
- Platform: macOS
- Files: {files_list}
{task_context}
{mode_rules}

CAPABILITIES:
- Execute shell commands (bash blocks)
- Create/edit files (# filename: header)
- Auto-continue on multi-step tasks
- Check status of background tasks

RULES:
1. NO emojis
2. For multi-step tasks, say "Next:" to continue automatically
3. Say "Done." when task is complete
4. If user asks about status/progress, report the LAST TASK STATUS above
5. When you find relevant results, CONTINUE exploring - don't stop to ask questions
6. Be proactive: if user asks to "check" something, explore it fully before stopping

CRITICAL - READ CAREFULLY:
- NEVER claim success before seeing actual command output
- ALWAYS check [COMMAND RESULTS] messages to see what actually happened
- `cd` DOES NOT persist between commands! Use full paths or combine: `cd dir && command`
- If a command fails, READ the error message before trying to fix it
- Do NOT hallucinate about directories - run `ls` to verify what exists

FILE FORMAT:
```python
# filename: myfile.py
code here
```

COMMAND FORMAT:
```bash
python myfile.py
```"""

    def _extract_actions(self, response):
        """Extract commands and file writes"""
        actions = []
        pattern = r'```(\w+)?\n(.*?)```'

        for lang, content in re.findall(pattern, response, re.DOTALL):
            content = content.strip()
            lang = (lang or '').lower()

            # File creation - support multiple comment styles
            # # filename: (Python, Shell, YAML, etc.)
            # // filename: (JavaScript, C, Go, etc.)
            # <!-- filename: --> (HTML, XML)
            first_line = content.split('\n')[0].strip()
            if first_line.startswith('# filename:') or first_line.startswith('// filename:'):
                filename = first_line.split(':', 1)[1].strip()
                file_content = '\n'.join(content.split('\n')[1:])
                actions.append(('file', filename, file_content))
            elif first_line.startswith('<!-- filename:') and first_line.endswith('-->'):
                # HTML comment style: <!-- filename: path/to/file.html -->
                filename = first_line.replace('<!-- filename:', '').replace('-->', '').strip()
                file_content = '\n'.join(content.split('\n')[1:])
                actions.append(('file', filename, file_content))
            # Shell command
            elif lang in ('bash', 'sh', 'shell', 'cmd', 'powershell', 'bat'):
                lines = content.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if not line or line.startswith('#'):
                        i += 1
                        continue

                    # Check for heredoc pattern: << 'DELIM', << "DELIM", << DELIM, <<DELIM
                    heredoc_match = re.search(r'<<-?\s*[\'"]?(\w+)[\'"]?\s*$', line)
                    if heredoc_match:
                        delimiter = heredoc_match.group(1)
                        heredoc_lines = [line]
                        i += 1
                        # Collect until we hit the delimiter
                        while i < len(lines):
                            heredoc_lines.append(lines[i])
                            if lines[i].strip() == delimiter:
                                break
                            i += 1
                        # Join as single command
                        actions.append(('cmd', '\n'.join(heredoc_lines)))
                    else:
                        actions.append(('cmd', line))
                    i += 1

        return actions

    def _stream_response(self, user_message):
        """Stream AI response with live output (with auto-compaction)"""
        # Check if compaction is needed BEFORE this turn
        if self._needs_compaction():
            self._compact_context()

        # Build messages: system + context (summary + recent if compacted) + new user message
        messages = [{"role": "system", "content": self._get_system_prompt(user_message)}]
        messages.extend(self._get_messages_for_context())
        messages.append({"role": "user", "content": user_message})

        # Groq has stricter payload limits - truncate if needed
        provider = self.config.get("provider", "ollama")
        if provider == "groq":
            MAX_GROQ_CHARS = 100000  # ~25k tokens, safe limit
            total_chars = sum(len(m.get("content", "")) for m in messages)
            if total_chars > MAX_GROQ_CHARS:
                self._print(f"[yellow]{sym('warn')} Context too large for Groq - truncating...[/yellow]")
                # Keep system prompt + last few messages
                system_msg = messages[0]
                user_msg = messages[-1]
                # Truncate middle messages
                middle = messages[1:-1]
                while sum(len(m.get("content", "")) for m in [system_msg] + middle + [user_msg]) > MAX_GROQ_CHARS and len(middle) > 2:
                    middle = middle[1:]  # Remove oldest messages
                messages = [system_msg] + middle + [user_msg]
                self._print(f"[dim]  Reduced to {len(messages)} messages[/dim]")

        full_response = ""
        in_code_block = False
        backtick_count = 0
        first_token = True

        # ANSI color codes
        BLUE = "\033[94m"
        WHITE = "\033[97m"
        RESET = "\033[0m"

        # Calculate input tokens (all messages being sent)
        input_text = json.dumps(messages)
        input_tokens = estimate_tokens(input_text)
        self.tokens_in += input_tokens

        # Animated spinner while waiting
        spinner = Spinner("Thinking")
        spinner.start()

        try:
            if provider == "groq" and GROQ_SDK:
                # Use Groq API
                client = self._get_groq_client()
                if not client:
                    spinner.stop()
                    return None

                stream = client.chat.completions.create(
                    model=self.config.get("groq_model", "compound-beta"),
                    messages=messages,
                    stream=True,
                    temperature=self.config["temperature"],
                    max_tokens=self.config.get("max_tokens", 4096),
                )

                for chunk in stream:
                    if first_token:
                        spinner.stop()
                        print(WHITE, end="", flush=True)
                        first_token = False
                    if chunk.choices and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        full_response += text

                        for char in text:
                            if char == '`':
                                backtick_count += 1
                            else:
                                if backtick_count >= 3:
                                    in_code_block = not in_code_block
                                    print('`' * backtick_count, end="", flush=True)
                                    print(BLUE if in_code_block else WHITE, end="", flush=True)
                                elif backtick_count > 0:
                                    print('`' * backtick_count, end="", flush=True)
                                backtick_count = 0
                                print(char, end="", flush=True)

                if backtick_count >= 3:
                    in_code_block = not in_code_block
                    print('`' * backtick_count, end="", flush=True)
                elif backtick_count > 0:
                    print('`' * backtick_count, end="", flush=True)
                print(RESET)

            elif provider == "anthropic" and ANTHROPIC_SDK:
                # Use Anthropic API
                client = self._get_anthropic_client()
                if not client:
                    spinner.stop()
                    return None

                # Anthropic uses system as separate param, not in messages
                system_content = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
                api_messages = [m for m in messages if m["role"] != "system"]

                with client.messages.stream(
                    model=self.config.get("anthropic_model", "claude-sonnet-4-20250514"),
                    max_tokens=self.config.get("max_tokens", 4096),
                    system=system_content,
                    messages=api_messages,
                ) as stream:
                    for text in stream.text_stream:
                        if first_token:
                            spinner.stop()
                            print(WHITE, end="", flush=True)
                            first_token = False
                        full_response += text

                        for char in text:
                            if char == '`':
                                backtick_count += 1
                            else:
                                if backtick_count >= 3:
                                    in_code_block = not in_code_block
                                    print('`' * backtick_count, end="", flush=True)
                                    print(BLUE if in_code_block else WHITE, end="", flush=True)
                                elif backtick_count > 0:
                                    print('`' * backtick_count, end="", flush=True)
                                backtick_count = 0
                                print(char, end="", flush=True)

                if backtick_count >= 3:
                    in_code_block = not in_code_block
                    print('`' * backtick_count, end="", flush=True)
                elif backtick_count > 0:
                    print('`' * backtick_count, end="", flush=True)
                print(RESET)

            elif OLLAMA_SDK:
                # Use Ollama
                stream = ollama.chat(
                    model=self.config["model"],
                    messages=messages,
                    stream=True,
                    options={"temperature": self.config["temperature"]}
                )

                for chunk in stream:
                    # Stop spinner on first token
                    if first_token:
                        spinner.stop()
                        print(WHITE, end="", flush=True)
                        first_token = False
                    if chunk.message.content:
                        text = chunk.message.content
                        full_response += text

                        for char in text:
                            if char == '`':
                                backtick_count += 1
                            else:
                                # If we had backticks, check if it was a code fence
                                if backtick_count >= 3:
                                    # Toggle code block state
                                    in_code_block = not in_code_block
                                    # Print the backticks and switch color
                                    print('`' * backtick_count, end="", flush=True)
                                    if in_code_block:
                                        print(BLUE, end="", flush=True)
                                    else:
                                        print(WHITE, end="", flush=True)
                                elif backtick_count > 0:
                                    # Print any accumulated backticks (inline code)
                                    print('`' * backtick_count, end="", flush=True)
                                backtick_count = 0
                                print(char, end="", flush=True)

                # Handle any remaining backticks at the end
                if backtick_count >= 3:
                    in_code_block = not in_code_block
                    print('`' * backtick_count, end="", flush=True)
                elif backtick_count > 0:
                    print('`' * backtick_count, end="", flush=True)

                print(RESET)  # Reset color at end

        except KeyboardInterrupt:
            spinner.stop()
            self._print(f"\n[yellow]{sym('warn')} Interrupted[/yellow]")
            return None
        except Exception as e:
            spinner.stop()
            self._print(f"\n[red]{sym('cross')} Error: {e}[/red]")
            return None

        self._save_message("user", user_message)
        self._save_message("assistant", full_response)

        # Track output tokens and show stats
        output_tokens = estimate_tokens(full_response)
        self.tokens_out += output_tokens
        total_context = self.tokens_in + self.tokens_out

        # Get context limit based on provider
        provider = self.config.get("provider", "ollama")
        if provider == "groq":
            current_model = self.config.get("groq_model", "compound-beta")
        elif provider == "anthropic":
            current_model = self.config.get("anthropic_model", "claude-sonnet-4-20250514")
        else:
            current_model = self.config["model"]
        context_limit = get_model_context_limit(current_model)
        context_pct = min(100, (total_context / context_limit) * 100)

        self._print(f"\n[dim]  tokens: in={format_tokens(input_tokens)} out={format_tokens(output_tokens)} | context: {format_tokens(total_context)} ({context_pct:.0f}%)[/dim]")

        return full_response

    def chat(self, message, auto_continue=True):
        """Agentic chat - auto-continues after tasks until done"""
        # Show user input (only for initial message)
        msg_id = getattr(self, 'msg_id', 0)
        if not message.startswith("[AUTO]"):
            self._print(f"\n[dim]#{msg_id}[/dim] [bold white]You[/bold white] [dim]›[/dim] {message}")

        # Show provider name in response header
        provider = self.config.get("provider", "ollama")
        if provider == "anthropic":
            ai_name = "Claude"
        elif provider == "groq":
            ai_name = "Groq"
        else:
            ai_name = "DeepSeek"

        self._print(f"\n[dim]#{msg_id}[/dim] [bold cyan]{ai_name}[/bold cyan] [dim]›[/dim]")

        response = self._stream_response(message.replace("[AUTO] ", ""))
        if not response:
            return

        # Process actions
        actions = self._extract_actions(response)

        if not actions:
            # No actions - AI is done or just explaining
            return

        # Enable buffering in auto mode for collapsed view
        is_auto = self.config.get("auto_execute", False)
        if is_auto:
            self._buffering_actions = True
            self._action_buffer = []

        self._print(f"\n[bold white]  {sym('diamond')} Actions ({len(actions)})[/bold white]")

        results = []  # Collect results for auto-continue
        completed = 0
        failed = 0

        for action in actions:
            if action[0] == 'cmd':
                cmd = action[1]

                # Handle `cd` commands specially - actually change working dir
                if cmd.strip().startswith('cd ') and '&&' not in cmd:
                    target = cmd.strip()[3:].strip()
                    try:
                        new_dir = os.path.abspath(os.path.join(self.working_dir, os.path.expanduser(target)))
                        if os.path.isdir(new_dir):
                            self.working_dir = new_dir
                            os.chdir(new_dir)
                            self._print(f"  [green]{sym('check')} Changed to {new_dir}[/green]", force=True)
                            results.append(f"CD: Changed to {new_dir}")
                            completed += 1
                            continue
                        else:
                            self._print(f"  [red]{sym('cross')} Directory not found: {target}[/red]", force=True)
                            results.append(f"ERROR: Directory not found: {target}")
                            failed += 1
                            continue
                    except Exception as e:
                        self._print(f"  [red]{sym('cross')} cd failed: {e}[/red]", force=True)
                        results.append(f"ERROR: cd failed: {e}")
                        failed += 1
                        continue

                if self._ask_permission(cmd[:60]):
                    output, code = self._run_command(cmd)

                    # Check for user interruption - don't treat as error
                    if code == -999:
                        results.append(f"INTERRUPTED: {cmd[:50]}")
                        self._print(f"\n  [dim]{sym('corner')} Stopping - user interrupted[/dim]")
                        break  # Stop processing further actions

                    results.append(f"Command: {cmd[:50]}\nExit: {code}\nOutput: {output[:300]}")
                    if code == 0:
                        completed += 1
                    else:
                        failed += 1

                    if code != 0:
                        # Auto-fix on failure
                        self._print(f"\n  [yellow]{sym('warn')} Failed - AI will attempt fix...[/yellow]")
                        results.append(f"ERROR: Command failed with exit code {code}")
                else:
                    results.append(f"SKIPPED: {cmd[:50]}")

            elif action[0] == 'file':
                if self._write_file(action[1], action[2]):
                    results.append(f"CREATED: {action[1]}")
                    completed += 1
                else:
                    results.append(f"SKIPPED: {action[1]}")

        # End buffering and show collapsed summary
        if is_auto and self._buffering_actions:
            self._buffering_actions = False
            self._last_actions_output = self._action_buffer.copy()

            # Print collapsed summary (force=True to bypass buffering)
            status_color = "green" if failed == 0 else "yellow"
            status_icon = sym('check') if failed == 0 else sym('warn')
            self._print(f"\n  [{status_color}]{status_icon} Actions: {completed} done, {failed} failed[/{status_color}] [dim](Ctrl+A to expand)[/dim]", force=True)

        # Save command results to session so AI can see them next turn
        if results:
            results_summary = "\n".join(results)
            self._save_message("system", f"[COMMAND RESULTS]\n{results_summary}")

        # Auto-continue: feed results back to AI
        if auto_continue and results:
            # Check if user interrupted - don't auto-continue
            was_interrupted = any("INTERRUPTED:" in r for r in results)
            if was_interrupted:
                return

            # Check if any command failed (but not interrupted)
            has_error = any("ERROR:" in r for r in results)

            if has_error:
                # Auto-fix mode
                self._print(f"\n[dim]  {sym('loop')} Auto-continuing to fix...[/dim]")
                context = "\n".join(results[-3:])  # Last 3 results
                self.chat(f"[AUTO] Results:\n{context}\n\nThere was an error. Please fix it and continue.", auto_continue=True)
            else:
                # Check if AI wants to continue (look for indicators)
                continue_words = ['next', 'then', 'now', 'after', 'continue', 'step', 'let me', "i'll"]
                stop_words = ['done.', 'complete', 'finished', 'that\'s all', 'let me know']

                should_continue = any(word in response.lower() for word in continue_words)
                should_stop = any(word in response.lower() for word in stop_words)

                # Also continue if AI asked questions but there are more steps to take
                asked_questions = '?' in response and len(actions) > 0

                if (should_continue or asked_questions) and not should_stop and len(actions) > 0:
                    self._print(f"\n[dim]  {sym('loop')} Auto-continuing...[/dim]")
                    context = "\n".join(results[-2:])
                    self.chat(f"[AUTO] Done. Results:\n{context}\n\nAnalyze the results and continue. If you found what was requested, explore it further.", auto_continue=True)

    def interactive(self):
        """Interactive chat loop"""
        # Use simple mode - fullscreen has flicker issues
        self._interactive_simple()

    def _interactive_simple(self):
        """Simple interactive mode with message IDs"""
        # Message ID counter
        self.msg_id = len(self.session_messages) // 2

        # Mode indicator
        mode = self.config.get("mode", "normal")
        mode_str = f" [{mode}]" if mode != "normal" else ""

        # Provider/model info
        provider = self.config.get("provider", "ollama")
        if provider == "groq":
            model_info = f"Groq: {self.config.get('groq_model', 'compound-beta')}"
        elif provider == "anthropic":
            model_info = f"Anthropic: {self.config.get('anthropic_model', 'claude-sonnet-4-20250514')}"
        else:
            model_info = f"Ollama: {self.config['model']}"

        self._print(f"\n[bold cyan]DeepSeek CLI[/bold cyan] v0.1.0{mode_str}")
        self._print(f"[dim]I bootstrap, refactor, and debug projects in your repo.[/dim]")
        self._print(f"[dim]{model_info}[/dim]")
        self._print(f"[dim]/help · /auto · /provider · /exit[/dim]\n")

        # Show resumed messages
        if self.session_messages:
            self._print(f"[green]{sym('check')} Resumed session ({len(self.session_messages)} msgs)[/green]")

        # Setup prompt session for history/completion
        session = None
        if PROMPT_TOOLKIT:
            history_file = CONFIG_DIR / "history.txt"

            # Key bindings for Ctrl+A to expand actions
            kb = KeyBindings()

            @kb.add('c-a')
            def expand_actions(event):
                """Expand last action output"""
                if self._last_actions_output:
                    print()  # Newline
                    for line in self._last_actions_output:
                        if RICH_AVAILABLE:
                            console.print(line)
                        else:
                            print(line)
                    print()  # Newline after

            try:
                session = PromptSession(history=FileHistory(str(history_file)), key_bindings=kb)
            except:
                session = PromptSession(key_bindings=kb)

        while True:
            try:
                dir_name = Path(self.working_dir).name
                mode = self.config.get("mode", "normal")
                mode_hint = f":{mode[0]}" if mode != "normal" else ""  # :c or :e

                # Status indicator before prompt
                self._print(f"[dim]Ready[/dim]")
                prompt_str = f"[{dir_name}{mode_hint}] › "

                if session:
                    user_input = session.prompt(prompt_str).strip()
                else:
                    print(prompt_str, end="", flush=True)
                    user_input = input().strip()

                if not user_input:
                    continue
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue

                # Increment message ID and show
                self.msg_id += 1
                self.chat(user_input)
            except KeyboardInterrupt:
                print("\n[Interrupted - /exit to quit]")
            except EOFError:
                break

        self._close_session()
        print("Goodbye!")

    def _interactive_fullscreen(self):
        """Full screen mode with fixed input at bottom"""
        # Message ID counter
        self.msg_id = len(self.session_messages) // 2  # Start from existing count

        # Output buffer
        self.output_lines = []
        self._add_output(f"╭─ DeepSeek CLI v0.1.0 ─────────────────────────────────╮")
        self._add_output(f"│ Model: {self.config['model'][:45]:<45} │")
        self._add_output(f"│ Dir:   {Path(self.working_dir).name[:45]:<45} │")
        self._add_output(f"│ Session: {self.session_id:<43} │")
        self._add_output(f"╰─ /help · /status · /exit ─────────────────────────────╯")
        self._add_output("")

        # Load previous messages if resuming
        if self.session_messages:
            self._add_output(f"[Resumed session with {len(self.session_messages)} messages]")
            for i, msg in enumerate(self.session_messages[-10:]):  # Show last 10
                prefix = "You" if msg['role'] == 'user' else "AI"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                self._add_output(f"  {prefix}: {content}")
            self._add_output("")

        # History
        history = InMemoryHistory()
        history_file = CONFIG_DIR / "history.txt"
        if history_file.exists():
            try:
                for line in history_file.read_text().splitlines()[-100:]:
                    if line.startswith('+'):
                        history.append_string(line[1:])
            except:
                pass

        # State
        self._should_exit = False
        self._pending_input = None
        self._input_text = ""

        # Styles
        style = PTStyle.from_dict({
            'output': '#ffffff',
            'input': '#00aaff',
            'input-area': 'bg:#1a1a1a',
            'border': '#333333',
            'toolbar': 'bg:#0a0a0a #666666',
            'id': '#666666',
        })

        while not self._should_exit:
            try:
                # Create output area
                output_text = '\n'.join(self.output_lines[-500:])
                output_area = TextArea(
                    text=output_text,
                    read_only=True,
                    scrollbar=True,
                    style='class:output',
                    wrap_lines=True,
                )
                output_area.buffer.cursor_position = len(output_text)

                # Create expandable input area (multiline, grows with content)
                input_area = TextArea(
                    text=self._input_text,
                    height=Dimension(min=2, max=10, preferred=3),  # Grows from 2 to 10 lines
                    prompt=f'› ',
                    style='class:input',
                    multiline=True,
                    wrap_lines=True,
                    scrollbar=False,
                )

                # Key bindings for this app instance
                kb = KeyBindings()

                @kb.add('escape', 'enter')  # Alt+Enter or Escape then Enter for newline
                def newline(event):
                    event.app.current_buffer.insert_text('\n')

                @kb.add('enter')
                def submit(event):
                    self._pending_input = event.app.current_buffer.text
                    self._input_text = ""
                    event.app.exit()

                @kb.add('c-c')
                def ctrl_c(event):
                    self._pending_input = None
                    self._input_text = ""
                    event.app.exit()

                @kb.add('c-d')
                def ctrl_d(event):
                    self._should_exit = True
                    event.app.exit()

                # Toolbar
                dir_name = Path(self.working_dir).name
                def get_toolbar():
                    return HTML(
                        f'<b>[{dir_name}]</b> '
                        f'<style fg="#444444">│</style> '
                        f'<style fg="#666666">Enter: send · Esc+Enter: newline · Ctrl+D: exit</style>'
                    )

                # Layout
                layout = Layout(
                    HSplit([
                        output_area,
                        Window(height=1, char='─', style='class:border'),
                        input_area,
                        Window(height=1, content=FormattedTextControl(get_toolbar), style='class:toolbar'),
                    ])
                )

                # Create and run app
                app = Application(
                    layout=layout,
                    key_bindings=kb,
                    style=style,
                    full_screen=True,
                    mouse_support=True,
                )
                app.layout.focus(input_area)
                app.run()

                # Process input
                if self._pending_input is not None:
                    user_input = self._pending_input.strip()
                    self._pending_input = None

                    if not user_input:
                        continue

                    # Save to history file
                    try:
                        with open(history_file, 'a') as f:
                            f.write(f'+{user_input}\n')
                    except:
                        pass

                    # Increment message ID
                    self.msg_id += 1

                    # Show user input with ID
                    self._add_output(f"\n[#{self.msg_id}] You › {user_input}")

                    if user_input.startswith('/'):
                        self._handle_command_fullscreen(user_input)
                        continue

                    # Process chat
                    self._chat_fullscreen(user_input)

            except KeyboardInterrupt:
                self._add_output("\n[Interrupted - Ctrl+D to exit]")
            except Exception as e:
                self._add_output(f"\n[Error: {e}]")

        self._close_session()
        print("\033[2J\033[H")  # Clear screen
        print("Goodbye!")

    def _add_output(self, text):
        """Add text to output buffer"""
        self.output_lines.append(text)

    def _chat_fullscreen(self, message):
        """Chat in fullscreen mode"""
        provider = self.config.get("provider", "ollama")
        ai_name = "Claude" if provider == "anthropic" else "Groq" if provider == "groq" else "DeepSeek"
        self._add_output(f"\n[#{self.msg_id}] {ai_name} ›")

        # Get response
        messages = [{"role": "system", "content": self._get_system_prompt(message)}]
        messages.extend(self.session_messages)
        messages.append({"role": "user", "content": message})

        try:
            if OLLAMA_SDK:
                response_text = ""
                stream = ollama.chat(
                    model=self.config["model"],
                    messages=messages,
                    stream=True,
                    options={"temperature": self.config["temperature"]}
                )

                current_line = ""
                for chunk in stream:
                    if chunk.message.content:
                        text = chunk.message.content
                        response_text += text
                        current_line += text

                        # Split on newlines
                        while '\n' in current_line:
                            line, current_line = current_line.split('\n', 1)
                            self._add_output(line)

                # Add remaining text
                if current_line:
                    self._add_output(current_line)

                self._save_message("user", message)
                self._save_message("assistant", response_text)

                # Extract and process actions
                actions = self._extract_actions(response_text)
                if actions:
                    self._add_output(f"\n[Actions: {len(actions)}]")
                    for action in actions:
                        if action[0] == 'cmd':
                            self._add_output(f"  › Run: {action[1][:50]}")
                            if self.config.get("auto_execute"):
                                output, code = self._run_command(action[1])
                                self._add_output(f"  Exit: {code}")
                        elif action[0] == 'file':
                            self._add_output(f"  › File: {action[1]}")

        except Exception as e:
            self._add_output(f"[Error: {e}]")

    def _handle_command_fullscreen(self, cmd):
        """Handle slash commands in fullscreen mode"""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ('/exit', '/quit', '/q'):
            self._should_exit = True
        elif command == '/help':
            self._add_output("\nCommands:")
            self._add_output("  /status  - Check last task")
            self._add_output("  /auto    - Toggle auto-execute")
            self._add_output("  /clear   - Clear output")
            self._add_output("  /cd      - Change directory")
            self._add_output("  /exit    - Quit")
        elif command == '/status':
            status, msg = self._get_task_status()
            if status:
                self._add_output(f"\n[Task Status: {status}]")
                self._add_output(msg)
            else:
                self._add_output("\n[No recent task]")
        elif command == '/auto':
            self.config["auto_execute"] = not self.config.get("auto_execute", False)
            status = "ON" if self.config["auto_execute"] else "OFF"
            self._add_output(f"\n[Auto-execute: {status}]")
            self._save_config()
        elif command == '/clear':
            self.output_lines = ["[Cleared]"]
        elif command == '/cd':
            if arg:
                try:
                    os.chdir(os.path.expanduser(arg))
                    self.working_dir = os.getcwd()
                    self._add_output(f"\n[Changed to: {self.working_dir}]")
                except Exception as e:
                    self._add_output(f"\n[Error: {e}]")
            else:
                self._add_output(f"\n[Current: {self.working_dir}]")

        # Agent modes
        elif command == '/review':
            self.config["mode"] = "review"
            self._add_output(f"\n[Review mode - checking quality, security, performance]")
        elif command == '/tdd':
            self.config["mode"] = "tdd"
            self._add_output(f"\n[TDD mode - RED > GREEN > REFACTOR]")
        elif command == '/plan':
            self.config["mode"] = "plan"
            self._add_output(f"\n[Plan mode - will plan before coding]")
        elif command == '/security':
            self.config["mode"] = "security"
            self._add_output(f"\n[Security mode - checking for vulnerabilities]")
        elif command == '/refactor':
            self.config["mode"] = "refactor"
            self._add_output(f"\n[Refactor mode - improving structure]")
        elif command == '/normal':
            self.config["mode"] = "normal"
            self._add_output(f"\n[Normal mode]")

        # Feature toggles
        elif command == '/hooks':
            current = self.config.get("hooks_enabled", True)
            self.config["hooks_enabled"] = not current
            self._save_config()
            status = "ON" if not current else "OFF"
            self._add_output(f"\n[Hooks: {status}]")
        elif command == '/learn':
            current = self.config.get("continuous_learning", True)
            self.config["continuous_learning"] = not current
            self._save_config()
            status = "ON" if not current else "OFF"
            self._add_output(f"\n[Learning: {status}]")
        elif command == '/format':
            current = self.config.get("auto_format", True)
            self.config["auto_format"] = not current
            self._save_config()
            status = "ON" if not current else "OFF"
            self._add_output(f"\n[Auto-format: {status}]")
        elif command == '/mode':
            if arg in ('normal', 'concise', 'explain', 'review', 'tdd', 'plan', 'security', 'refactor'):
                self.config["mode"] = arg
                self._add_output(f"\n[Mode: {arg}]")
            else:
                self._add_output(f"\n[Mode: {self.config.get('mode', 'normal')}]")
                self._add_output(f"[Options: normal, concise, explain, review, tdd, plan, security, refactor]")

        else:
            self._add_output(f"\n[Unknown command: {command}]")

    def _handle_command(self, cmd):
        """Handle slash commands"""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ('/exit', '/quit', '/q'):
            self._close_session()
            self._print("[cyan]Goodbye![/cyan]")
            sys.exit(0)

        elif command == '/cd':
            if arg:
                try:
                    os.chdir(os.path.expanduser(arg))
                    self.working_dir = os.getcwd()
                    self._print(f"[green]{sym('check')} Changed to {self.working_dir}[/green]")
                except Exception as e:
                    self._print(f"[red]{sym('cross')} {e}[/red]")
            else:
                self._print(f"  {self.working_dir}")

        elif command == '/ls':
            files = sorted(os.listdir())[:30]
            for f in files:
                icon = "[DIR]" if os.path.isdir(f) else "[FILE]"
                self._print(f"  {icon} {f}")
            if len(os.listdir()) > 30:
                self._print(f"  [dim]... and {len(os.listdir()) - 30} more[/dim]")

        elif command == '/auto':
            self.config["auto_execute"] = not self.config.get("auto_execute", False)
            status = "[green]ON[/green]" if self.config["auto_execute"] else "[red]OFF[/red]"
            self._print(f"  Auto-execute: {status}")
            self._save_config()

        elif command == '/clear':
            self.session_messages = []
            self.tokens_in = 0
            self.tokens_out = 0
            self._compacted_summary = None  # Reset compaction state
            self._print(f"[green]{sym('check')} Session cleared, context reset[/green]")

        elif command == '/tokens':
            total = self.tokens_in + self.tokens_out
            context_limit = get_model_context_limit(self.config["model"])
            context_pct = min(100, (total / context_limit) * 100)
            compacted = "Yes" if getattr(self, '_compacted_summary', None) else "No"
            self._print(f"\n[bold]Token Usage:[/bold]")
            self._print(f"  Model:    {self.config['model']}")
            self._print(f"  Limit:    {format_tokens(context_limit)}")
            self._print(f"  Used:     {format_tokens(total)} ({context_pct:.0f}%)")
            self._print(f"  Messages: {len(self.session_messages)}")
            self._print(f"  Compacted: {compacted}")
            if context_pct > 70:
                self._print(f"  [yellow]{sym('warn')} Context high - auto-compaction will trigger soon[/yellow]")


        elif command == '/compact':
            # Force compaction now (Claude Code style)
            if len(self.session_messages) < 4:
                self._print(f"[dim]Only {len(self.session_messages)} messages - nothing to compact[/dim]")
            else:
                self._compact_context()

        elif command == '/sessions':
            sessions = self.list_sessions()
            if not sessions:
                self._print("[dim]No saved sessions[/dim]")
                return

            self._print("\n[bold]Recent Sessions:[/bold]")
            for i, s in enumerate(sessions, 1):
                current = " [cyan](current)[/cyan]" if s['id'] == self.session_id else ""
                self._print(f"  [cyan]{i}.[/cyan] {s['id']} [dim]{sym('vline')}[/dim] {Path(s['working_dir']).name} [dim]{sym('vline')}[/dim] {s['messages']} msgs{current}")
                if s['preview']:
                    self._print(f"      [dim]{s['preview']}[/dim]")

        elif command == '/run':
            if arg:
                self._run_command(arg)
            else:
                self._print(f"[yellow]{sym('warn')} Usage: /run <command>[/yellow]")

        elif command == '/status':
            status, msg = self._get_task_status()
            if not status:
                self._print(f"[dim]No recent task to check[/dim]")
            else:
                status_colors = {
                    'running': 'yellow',
                    'running_ok': 'green',
                    'running_background': 'cyan',
                    'completed': 'green',
                    'error': 'red',
                    'failed': 'red'
                }
                color = status_colors.get(status, 'white')
                self._print(f"\n[bold]Last Task Status:[/bold] [{color}]{status}[/{color}]")
                for line in msg.split('\n'):
                    self._print(f"  [dim]{sym('vline')}[/dim] {line}")
                # Show recent output
                if DSK._last_task and DSK._last_task.get('output'):
                    self._print(f"\n[bold]Recent Output:[/bold]")
                    for line in DSK._last_task['output'][-5:]:
                        self._print(f"  [dim]{sym('vline')}[/dim] {line}")

        elif command == '/kill':
            if DSK._last_task and DSK._last_task.get('process'):
                proc = DSK._last_task['process']
                if proc.poll() is None:
                    proc.kill()
                    DSK._last_task['status'] = 'killed'
                    self._print(f"[yellow]{sym('warn')} Killed background task[/yellow]")
                else:
                    self._print(f"[dim]Task already finished[/dim]")
            else:
                self._print(f"[dim]No background task to kill[/dim]")

        elif command == '/mode':
            valid_modes = ('concise', 'explain', 'normal', 'review', 'tdd', 'plan', 'security', 'refactor')
            if arg in valid_modes:
                self.config["mode"] = arg
                self._print(f"[green]{sym('check')} Mode: {arg}[/green]")
                if arg in AGENT_MODES and AGENT_MODES[arg]:
                    self._print(f"[dim]{AGENT_MODES[arg][:100]}...[/dim]")
            else:
                current = self.config.get('mode', 'normal')
                self._print(f"  Mode: {current}")
                self._print(f"  [dim]Options: /mode normal | concise | explain | review | tdd | plan | security | refactor[/dim]")

        # Quick agent mode switches
        elif command == '/review':
            self.config["mode"] = "review"
            self._print(f"[green]{sym('check')} Code Review mode - analyzing for quality, security, performance[/green]")

        elif command == '/tdd':
            self.config["mode"] = "tdd"
            self._print(f"[green]{sym('check')} TDD mode - RED > GREEN > REFACTOR cycle[/green]")

        elif command == '/plan':
            self.config["mode"] = "plan"
            self._print(f"[green]{sym('check')} Planning mode - will create plan before implementation[/green]")

        elif command == '/security':
            self.config["mode"] = "security"
            self._print(f"[green]{sym('check')} Security Review mode - checking for vulnerabilities[/green]")

        elif command == '/refactor':
            self.config["mode"] = "refactor"
            self._print(f"[green]{sym('check')} Refactor mode - improving code structure without changing behavior[/green]")

        elif command == '/normal':
            self.config["mode"] = "normal"
            self._print(f"[green]{sym('check')} Normal mode[/green]")

        elif command == '/hooks':
            # Toggle hooks
            current = self.config.get("hooks_enabled", True)
            self.config["hooks_enabled"] = not current
            self._save_config()
            status = "enabled" if not current else "disabled"
            self._print(f"[green]{sym('check')} Hooks {status}[/green]")

        elif command == '/learn':
            # Toggle continuous learning
            current = self.config.get("continuous_learning", True)
            self.config["continuous_learning"] = not current
            self._save_config()
            status = "enabled" if not current else "disabled"
            self._print(f"[green]{sym('check')} Continuous learning {status}[/green]")

        elif command == '/format':
            # Toggle auto-format
            current = self.config.get("auto_format", True)
            self.config["auto_format"] = not current
            self._save_config()
            status = "enabled" if not current else "disabled"
            self._print(f"[green]{sym('check')} Auto-format {status}[/green]")

        elif command == '/model':
            if arg:
                self.config["model"] = arg
                self._save_config()
                self._print(f"[green]{sym('check')} Model: {arg}[/green]")
            else:
                self._print(f"  Model: {self.config['model']}")

        elif command == '/models':
            selected = show_model_menu(self.config.get("model"))
            if selected and selected != self.config.get("model"):
                self.config["model"] = selected
                self._save_config()
                self._print(f"[green]{sym('check')} Switched to: {selected}[/green]")

        elif command == '/provider':
            # Show provider selection menu
            provider = show_provider_menu(self.config.get("provider", "ollama"))
            self.config["provider"] = provider

            if provider == "groq":
                groq_model = show_groq_model_menu(self.config.get("groq_model", "compound-beta"))
                self.config["groq_model"] = groq_model
                self._save_config()
                self._print(f"[green]{sym('check')} Switched to Groq: {groq_model}[/green]")
            elif provider == "anthropic":
                anthropic_model = show_anthropic_model_menu(self.config.get("anthropic_model", "claude-sonnet-4-20250514"))
                self.config["anthropic_model"] = anthropic_model
                self._save_config()
                self._print(f"[green]{sym('check')} Switched to Anthropic: {anthropic_model}[/green]")
            else:
                self._save_config()
                self._print(f"[green]{sym('check')} Switched to Ollama: {self.config['model']}[/green]")

        elif command == '/groq':
            # Quick switch to Groq
            self.config["provider"] = "groq"
            self._save_config()
            self._print(f"[green]{sym('check')} Provider: Groq ({self.config.get('groq_model', 'compound-beta')})[/green]")

        elif command == '/ollama':
            # Quick switch to Ollama
            self.config["provider"] = "ollama"
            self._save_config()
            self._print(f"[green]{sym('check')} Provider: Ollama ({self.config['model']})[/green]")

        elif command in ('/anthropic', '/claude'):
            # Quick switch to Anthropic
            self.config["provider"] = "anthropic"
            self._save_config()
            self._print(f"[green]{sym('check')} Provider: Anthropic ({self.config.get('anthropic_model', 'claude-sonnet-4-20250514')})[/green]")

        elif command == '/help':
            self._print("""
[bold]Provider:[/bold]
  /provider      Select provider (Ollama/Groq/Anthropic) + model
  /ollama        Quick switch to Ollama
  /groq          Quick switch to Groq
  /claude        Quick switch to Anthropic (Claude)
  /models        Select model from list
  /model <name>  Set model directly

[bold]Agent Modes:[/bold]
  /review        Code review mode (quality, security, perf)
  /tdd           Test-driven development (RED > GREEN > REFACTOR)
  /plan          Planning mode (plan before code)
  /security      Security review (OWASP, vulnerabilities)
  /refactor      Refactor mode (structure, no behavior change)
  /normal        Back to normal mode
  /mode <m>      Set any mode directly

[bold]Commands:[/bold]
  /auto          Toggle auto-execute
  /tokens        Show token/context usage
  /compact       Force context compaction now
  /status        Check last task status
  /kill          Kill background task
  /cd <path>     Change directory
  /run <cmd>     Run command directly
  /sessions      List saved sessions
  /clear         Clear session + reset tokens
  /hooks         Toggle hooks (format, lint warnings)
  /learn         Toggle continuous learning
  /exit          Quit

[bold]Context:[/bold]
  Auto-compacts at ~70% - keeps summary + last 10 turns
  Strategic suggestions after 50 tool calls
  Session state persisted for continuity
""")

        else:
            self._print(f"[yellow]{sym('warn')} Unknown: {command}[/yellow]")


def get_available_models():
    """Get list of available models (local + cloud)"""
    models = []

    # Ollama cloud models (from ollama.com)
    cloud_models = [
        ("deepseek-v3.1:671b-cloud", "DeepSeek V3.1 671B"),
        ("gpt-oss:120b-cloud", "GPT-OSS 120B"),
        ("gpt-oss:20b-cloud", "GPT-OSS 20B"),
        ("qwen3-coder:480b-cloud", "Qwen3 Coder 480B"),
        ("qwen3-vl:235b-cloud", "Qwen3 VL 235B"),
        ("minimax-m2:cloud", "MiniMax M2"),
        ("glm-4.6:cloud", "GLM 4.6"),
    ]

    # Get local models from ollama
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    if 'cloud' not in model_name:  # Don't duplicate cloud models
                        models.append((model_name, f"{model_name} (local)"))
    except:
        pass

    # Add cloud models
    models.extend(cloud_models)

    return models


def show_model_menu(current_model=None):
    """Show model selection menu"""
    models = get_available_models()

    if RICH_AVAILABLE:
        console.print("\n[bold cyan]Select Model[/bold cyan]\n")

        for i, (model_id, desc) in enumerate(models, 1):
            current = " [green](current)[/green]" if model_id == current_model else ""
            console.print(f"  [cyan]{i}.[/cyan] {desc}{current}")

        console.print(f"\n  [cyan]0.[/cyan] Enter custom model name")
        console.print(f"  [dim]Enter[/dim] Keep current ({current_model})\n")

    try:
        choice = input("  Select: ").strip()
        if not choice:
            return current_model
        if choice == "0":
            custom = input("  Model name: ").strip()
            return custom if custom else current_model
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx][0]
    except (ValueError, IndexError, KeyboardInterrupt, EOFError):
        pass
    return current_model


def show_resume_menu():
    """Show session selection menu"""
    sessions = DSK.list_sessions()
    if not sessions:
        return None

    if RICH_AVAILABLE:
        console.print("\n[bold cyan]DSK[/bold cyan] [dim]- Resume Session[/dim]\n")
        console.print("[bold]Recent Sessions:[/bold]\n")

        for i, s in enumerate(sessions, 1):
            console.print(f"  [cyan]{i}.[/cyan] {s['id']} [dim]{sym('vline')}[/dim] {Path(s['working_dir']).name} [dim]{sym('vline')}[/dim] {s['messages']} msgs")
            if s['preview']:
                console.print(f"      [dim]{s['preview']}[/dim]")

        console.print(f"\n  [cyan]0.[/cyan] Start new session\n")

    try:
        choice = input("  Select [0]: ").strip()
        if not choice or choice == "0":
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            return sessions[idx]['id']
    except (ValueError, IndexError, KeyboardInterrupt, EOFError):
        pass
    return None


def show_provider_menu(current_provider="ollama"):
    """Show provider selection menu (Ollama vs Groq vs Anthropic)"""
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]DSK[/bold cyan] [dim]- Select Provider[/dim]\n")

    providers = [
        ("ollama", "Ollama (local/cloud models via ollama)"),
        ("groq", "Groq API (compound-beta, llama, etc.)"),
        ("anthropic", "Anthropic API (Claude Sonnet/Opus)"),
    ]

    for i, (pid, desc) in enumerate(providers, 1):
        marker = "[green]*[/green]" if pid == current_provider else " "
        if RICH_AVAILABLE:
            console.print(f"  {marker} [cyan]{i}.[/cyan] {desc}")
        else:
            print(f"  {'*' if pid == current_provider else ' '} {i}. {desc}")

    if RICH_AVAILABLE:
        console.print()

    try:
        choice = input(f"  Select [1]: ").strip()
        if not choice:
            return current_provider
        idx = int(choice) - 1
        if 0 <= idx < len(providers):
            return providers[idx][0]
    except (ValueError, IndexError, KeyboardInterrupt, EOFError):
        pass
    return current_provider


def show_groq_model_menu(current_model="compound-beta"):
    """Show Groq model selection menu"""
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]DSK[/bold cyan] [dim]- Select Groq Model[/dim]\n")

    for i, model in enumerate(GROQ_MODELS, 1):
        marker = "[green]*[/green]" if model == current_model else " "
        if RICH_AVAILABLE:
            console.print(f"  {marker} [cyan]{i}.[/cyan] {model}")
        else:
            print(f"  {'*' if model == current_model else ' '} {i}. {model}")

    if RICH_AVAILABLE:
        console.print()

    try:
        choice = input(f"  Select [1]: ").strip()
        if not choice:
            return current_model
        idx = int(choice) - 1
        if 0 <= idx < len(GROQ_MODELS):
            return GROQ_MODELS[idx]
    except (ValueError, IndexError, KeyboardInterrupt, EOFError):
        pass
    return current_model


def show_anthropic_model_menu(current_model="claude-sonnet-4-20250514"):
    """Show Anthropic model selection menu"""
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]DSK[/bold cyan] [dim]- Select Anthropic Model[/dim]\n")

    for i, model in enumerate(ANTHROPIC_MODELS, 1):
        marker = "[green]*[/green]" if model == current_model else " "
        if RICH_AVAILABLE:
            console.print(f"  {marker} [cyan]{i}.[/cyan] {model}")
        else:
            print(f"  {'*' if model == current_model else ' '} {i}. {model}")

    if RICH_AVAILABLE:
        console.print()

    try:
        choice = input(f"  Select [1]: ").strip()
        if not choice:
            return current_model
        idx = int(choice) - 1
        if 0 <= idx < len(ANTHROPIC_MODELS):
            return ANTHROPIC_MODELS[idx]
    except (ValueError, IndexError, KeyboardInterrupt, EOFError):
        pass
    return current_model


def main():
    parser = argparse.ArgumentParser(description="DSK - DeepSeek Terminal Agent")
    parser.add_argument("prompt", nargs="*", help="Quick prompt")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume session")
    parser.add_argument("--auto", "-a", action="store_true", help="Auto-execute")
    parser.add_argument("--concise", "-c", action="store_true", help="Concise mode - minimal explanation")
    parser.add_argument("--explain", "-e", action="store_true", help="Explain mode - teaching style")
    parser.add_argument("--select", "-s", action="store_true", help="Select provider/model at startup")
    parser.add_argument("--model", "-m", type=str, help="Model name directly")
    parser.add_argument("--groq", "-g", action="store_true", help="Use Groq API")
    parser.add_argument("--ollama", "-o", action="store_true", help="Use Ollama (default)")
    parser.add_argument("--anthropic", "--claude", action="store_true", help="Use Anthropic API (Claude)")
    parser.add_argument("--dir", "-d", type=str, help="Working directory")

    args = parser.parse_args()

    session_id = None

    # Check for active/crashed session
    if args.resume:
        session_id = show_resume_menu()
    elif not args.prompt:
        active = DSK.get_active_session()
        if active:
            if RICH_AVAILABLE:
                console.print(f"\n[yellow]{sym('warn')} Found interrupted session: {active}[/yellow]")

            options = ["Resume session", "Start fresh"]
            if PROMPT_TOOLKIT:
                from prompt_toolkit.shortcuts import radiolist_dialog
                try:
                    # Simple menu
                    print("  1. Resume session")
                    print("  2. Start fresh")
                    choice = input("  Choice [1]: ").strip()
                    if choice != "2":
                        session_id = active
                except:
                    pass
            else:
                choice = input("  Resume? [Y/n]: ").strip().lower()
                if choice in ('', 'y', 'yes'):
                    session_id = active

    dsk = DSK(working_dir=args.dir, session_id=session_id)

    if args.auto:
        dsk.config["auto_execute"] = True
    if args.concise:
        dsk.config["mode"] = "concise"
    elif args.explain:
        dsk.config["mode"] = "explain"

    # Provider selection via flags
    if args.groq:
        dsk.config["provider"] = "groq"
    elif args.anthropic:
        dsk.config["provider"] = "anthropic"
    elif args.ollama:
        dsk.config["provider"] = "ollama"

    # Interactive provider/model selection
    if args.select:
        # First select provider
        provider = show_provider_menu(dsk.config.get("provider", "ollama"))
        dsk.config["provider"] = provider

        if provider == "groq":
            # Select Groq model
            groq_model = show_groq_model_menu(dsk.config.get("groq_model", "compound-beta"))
            dsk.config["groq_model"] = groq_model
        elif provider == "anthropic":
            # Select Anthropic model
            anthropic_model = show_anthropic_model_menu(dsk.config.get("anthropic_model", "claude-sonnet-4-20250514"))
            dsk.config["anthropic_model"] = anthropic_model
        else:
            # Select Ollama model
            selected = show_model_menu(dsk.config.get("model"))
            if selected:
                dsk.config["model"] = selected

        dsk._save_config()

    # Direct model override
    if args.model:
        if dsk.config.get("provider") == "groq":
            dsk.config["groq_model"] = args.model
        else:
            dsk.config["model"] = args.model

    if args.prompt:
        dsk.chat(" ".join(args.prompt))
    else:
        dsk.interactive()


if __name__ == "__main__":
    main()
