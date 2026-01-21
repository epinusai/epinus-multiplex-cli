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

DEFAULT_CONFIG = {
    "model": "deepseek-v3.1:671b-cloud",
    "max_tokens": 4096,
    "temperature": 0.7,
    "auto_execute": False,
    "mode": "normal",  # normal, concise, explain
}


class DSK:
    def __init__(self, working_dir=None, session_id=None):
        CONFIG_DIR.mkdir(exist_ok=True)
        SESSIONS_DIR.mkdir(exist_ok=True)

        self.config = self._load_config()
        self.working_dir = working_dir or os.getcwd()
        os.chdir(self.working_dir)

        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_db = SESSIONS_DIR / f"session_{self.session_id}.db"
        self.session_messages = []
        self._init_db()

        if session_id:
            self._load_session()

    def _load_config(self):
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        return DEFAULT_CONFIG.copy()

    def _save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)

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
        try:
            conn = sqlite3.connect(self.session_db)
            conn.execute("UPDATE meta SET value='0' WHERE key='active'")
            conn.commit()
            conn.close()
        except sqlite3.Error:
            pass

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

    def _print(self, text, style=None):
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

    # Patterns for long-running/polling commands (servers + builds)
    LONG_RUNNING_CMDS = ['npm run dev', 'npm start', 'yarn dev', 'yarn start',
                         'pnpm dev', 'pnpm start', 'next dev', 'vite',
                         'python -m http.server', 'flask run', 'uvicorn',
                         'node server', 'nodemon', 'ts-node', 'webpack serve',
                         # Build commands (can take a while)
                         'npm run build', 'yarn build', 'pnpm build', 'next build',
                         'webpack', 'tsc', 'npm run lint', 'yarn lint',
                         'npm test', 'yarn test', 'pytest', 'cargo build', 'go build']

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
        return any(pattern in cmd_lower for pattern in self.LONG_RUNNING_CMDS)

    def _run_command(self, cmd):
        """Execute command with live streaming output and stats"""
        self._print(f"  [dim]$ {cmd[:70]}{'...' if len(cmd) > 70 else ''}[/dim]")

        is_server = self._is_long_running(cmd)
        if is_server:
            self._print(f"  [dim]{sym('vline')} Detected server command - polling mode[/dim]")
            return self._run_server_command(cmd)

        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=self.working_dir, bufsize=1
            )

            output_lines = []
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip()
                output_lines.append(line)
                elapsed = time.time() - start_time
                # Show live output with stats
                if len(output_lines) <= 12:
                    self._print(f"  [dim]{sym('vline')}[/dim] {line}")
                elif len(output_lines) == 13:
                    self._print(f"  [dim]{sym('vline')} ... streaming ({elapsed:.1f}s)[/dim]")

            process.wait()
            elapsed = time.time() - start_time

            # Summary stats
            if len(output_lines) > 12:
                self._print(f"  [dim]{sym('vline')} {len(output_lines)} lines in {elapsed:.1f}s[/dim]")

            if process.returncode == 0:
                self._print(f"  [green]{sym('check')} Done ({elapsed:.1f}s)[/green]")
            else:
                self._print(f"  [red]{sym('cross')} Failed (exit {process.returncode}, {elapsed:.1f}s)[/red]")

            return '\n'.join(output_lines), process.returncode

        except KeyboardInterrupt:
            process.kill()
            self._print(f"  [yellow]{sym('warn')} Interrupted by user[/yellow]")
            return "INTERRUPTED", -999  # Special code for user interruption
        except Exception as e:
            self._print(f"  [red]{sym('cross')} Error: {e}[/red]")
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

        # Preview
        self._print(f"\n  [bold {'green' if is_new else 'yellow'}]{'+ New' if is_new else '~ Edit'}:[/bold {'green' if is_new else 'yellow'}] {filepath}")

        lines = content.split('\n')
        for i, line in enumerate(lines[:10], 1):
            self._print(f"  [dim]{i:3}[/dim] [{'green' if is_new else 'yellow'}]{sym('vline')} {line}[/{'green' if is_new else 'yellow'}]")
        if len(lines) > 10:
            self._print(f"  [dim]    {sym('vline')} ... ({len(lines) - 10} more lines)[/dim]")

        if not self._ask_permission(f"Write {len(lines)} lines to {filepath}"):
            self._print(f"  [dim]{sym('corner')} Skipped[/dim]")
            return False

        os.makedirs(os.path.dirname(full_path) or '.', exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self._print(f"  [green]{sym('check')} Wrote {filepath}[/green]")
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

        if mode == "concise":
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

            # File creation
            if content.startswith('# filename:') or content.startswith('// filename:'):
                filename = content.split('\n')[0].split(':', 1)[1].strip()
                file_content = '\n'.join(content.split('\n')[1:])
                actions.append(('file', filename, file_content))
            # Shell command
            elif lang in ('bash', 'sh', 'shell', 'cmd', 'powershell', 'bat'):
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        actions.append(('cmd', line))

        return actions

    def _stream_response(self, user_message):
        """Stream AI response with live output"""
        messages = [{"role": "system", "content": self._get_system_prompt(user_message)}]
        messages.extend(self.session_messages)
        messages.append({"role": "user", "content": user_message})

        full_response = ""
        in_code_block = False
        backtick_count = 0

        # ANSI color codes
        BLUE = "\033[94m"
        WHITE = "\033[97m"
        RESET = "\033[0m"

        # Start with white for explanations
        print(WHITE, end="", flush=True)

        try:
            if OLLAMA_SDK:
                stream = ollama.chat(
                    model=self.config["model"],
                    messages=messages,
                    stream=True,
                    options={"temperature": self.config["temperature"]}
                )

                for chunk in stream:
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
            self._print(f"\n[yellow]{sym('warn')} Interrupted[/yellow]")
            return None
        except Exception as e:
            self._print(f"\n[red]{sym('cross')} Error: {e}[/red]")
            return None

        self._save_message("user", user_message)
        self._save_message("assistant", full_response)
        return full_response

    def chat(self, message, auto_continue=True):
        """Agentic chat - auto-continues after tasks until done"""
        # Show user input (only for initial message)
        msg_id = getattr(self, 'msg_id', 0)
        if not message.startswith("[AUTO]"):
            self._print(f"\n[dim]#{msg_id}[/dim] [bold white]You[/bold white] [dim]›[/dim] {message}")

        self._print(f"\n[dim]#{msg_id}[/dim] [bold cyan]DeepSeek[/bold cyan] [dim]›[/dim]")

        response = self._stream_response(message.replace("[AUTO] ", ""))
        if not response:
            return

        # Process actions
        actions = self._extract_actions(response)

        if not actions:
            # No actions - AI is done or just explaining
            return

        self._print(f"\n[bold white]  {sym('diamond')} Actions ({len(actions)})[/bold white]")

        results = []  # Collect results for auto-continue

        for action in actions:
            if action[0] == 'cmd':
                if self._ask_permission(action[1][:60]):
                    output, code = self._run_command(action[1])

                    # Check for user interruption - don't treat as error
                    if code == -999:
                        results.append(f"INTERRUPTED: {action[1][:50]}")
                        self._print(f"\n  [dim]{sym('corner')} Stopping - user interrupted[/dim]")
                        break  # Stop processing further actions

                    results.append(f"Command: {action[1][:50]}\nExit: {code}\nOutput: {output[:300]}")

                    if code != 0:
                        # Auto-fix on failure
                        self._print(f"\n  [yellow]{sym('warn')} Failed - AI will attempt fix...[/yellow]")
                        results.append(f"ERROR: Command failed with exit code {code}")
                else:
                    results.append(f"SKIPPED: {action[1][:50]}")

            elif action[0] == 'file':
                if self._write_file(action[1], action[2]):
                    results.append(f"CREATED: {action[1]}")
                else:
                    results.append(f"SKIPPED: {action[1]}")

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
                should_continue = any(word in response.lower() for word in ['next', 'then', 'now', 'after', 'continue', 'step'])

                if should_continue and len(actions) > 0:
                    self._print(f"\n[dim]  {sym('loop')} Auto-continuing...[/dim]")
                    context = "\n".join(results[-2:])
                    self.chat(f"[AUTO] Done. Results:\n{context}\n\nContinue with the next step.", auto_continue=True)

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

        self._print(f"\n[bold cyan]DeepSeek CLI[/bold cyan] v0.1.0{mode_str}")
        self._print(f"[dim]I bootstrap, refactor, and debug projects in your repo.[/dim]")
        self._print(f"[dim]Model: {self.config['model']}[/dim]")
        self._print(f"[dim]/help · /auto · /status · /exit[/dim]\n")

        # Show resumed messages
        if self.session_messages:
            self._print(f"[green]{sym('check')} Resumed session ({len(self.session_messages)} msgs)[/green]")

        # Setup prompt session for history/completion
        session = None
        if PROMPT_TOOLKIT:
            history_file = CONFIG_DIR / "history.txt"
            try:
                session = PromptSession(history=FileHistory(str(history_file)))
            except:
                session = PromptSession()

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
        self._add_output(f"\n[#{self.msg_id}] DeepSeek ›")

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
            self._print(f"[green]{sym('check')} Session cleared[/green]")

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
            if arg in ('concise', 'explain', 'normal'):
                self.config["mode"] = arg
                self._print(f"[green]{sym('check')} Mode: {arg}[/green]")
            else:
                current = self.config.get('mode', 'normal')
                self._print(f"  Mode: {current}")
                self._print(f"  [dim]Options: /mode concise | explain | normal[/dim]")

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

        elif command == '/help':
            self._print("""
[bold]Commands:[/bold]
  /models        Select model from list
  /model <name>  Set model directly
  /mode <m>      Set mode: concise | explain | normal
  /auto          Toggle auto-execute
  /status        Check last task status
  /kill          Kill background task
  /cd <path>     Change directory
  /ls            List files
  /run <cmd>     Run command directly
  /sessions      List saved sessions
  /clear         Clear session
  /exit          Quit

[bold]CLI flags:[/bold]
  deepseek --select     Select model at startup
  deepseek --model X    Use specific model
  deepseek --concise    Concise mode
  deepseek --explain    Explain mode
  deepseek --auto       Auto-execute commands
""")

        else:
            self._print(f"[yellow]{sym('warn')} Unknown: {command}[/yellow]")


def get_available_models():
    """Get list of available models (local + cloud)"""
    models = []

    # Popular cloud models (Ollama cloud)
    cloud_models = [
        ("deepseek-v3.1:671b-cloud", "DeepSeek V3.1 671B (cloud)"),
        ("deepseek-r1:671b-cloud", "DeepSeek R1 671B (cloud)"),
        ("llama3.3:70b-cloud", "Llama 3.3 70B (cloud)"),
        ("qwen2.5:72b-cloud", "Qwen 2.5 72B (cloud)"),
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


def main():
    parser = argparse.ArgumentParser(description="DSK - DeepSeek Terminal Agent")
    parser.add_argument("prompt", nargs="*", help="Quick prompt")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume session")
    parser.add_argument("--auto", "-a", action="store_true", help="Auto-execute")
    parser.add_argument("--concise", "-c", action="store_true", help="Concise mode - minimal explanation")
    parser.add_argument("--explain", "-e", action="store_true", help="Explain mode - teaching style")
    parser.add_argument("--select", "-s", action="store_true", help="Select model at startup")
    parser.add_argument("--model", "-m", type=str, help="Model name directly")
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

    # Model selection
    if args.model:
        dsk.config["model"] = args.model
    elif args.select:
        selected = show_model_menu(dsk.config.get("model"))
        if selected:
            dsk.config["model"] = selected
            dsk._save_config()

    if args.prompt:
        dsk.chat(" ".join(args.prompt))
    else:
        dsk.interactive()


if __name__ == "__main__":
    main()
