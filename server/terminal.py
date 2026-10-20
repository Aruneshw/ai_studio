"""Real interactive shell over a WebSocket, backed by a POSIX pseudo-terminal."""
import asyncio
import fcntl
import json
import os
import pty
import signal
import struct
import sys
import termios
from pathlib import Path

from .config import ROOT_DIR


def _shell_env():
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["COLORTERM"] = "truecolor"
    # Put the studio's Python (the one Unsloth is installed into) first on PATH.
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    if sys.prefix != sys.base_prefix:
        env["VIRTUAL_ENV"] = sys.prefix
    env["AI_STUDIO"] = "1"
    return env


async def run_terminal(ws, cols=120, rows=32):
    shell = os.environ.get("SHELL", "/bin/bash")
    pid, fd = pty.fork()
    if pid == 0:  # child
        try:
            os.chdir(str(ROOT_DIR))
            os.execvpe(shell, [shell, "-i"], _shell_env())
        finally:
            os._exit(1)

    def resize(c, r):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", int(r), int(c), 0, 0))

    resize(cols, rows)
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def on_readable():
        try:
            data = os.read(fd, 65536)
        except OSError:
            data = b""
        queue.put_nowait(data)
        if not data:
            loop.remove_reader(fd)

    loop.add_reader(fd, on_readable)

    async def pump_out():
        while True:
            data = await queue.get()
            if not data:
                await ws.send_text(json.dumps({"type": "exit"}))
                return
            await ws.send_bytes(data)

    out_task = asyncio.create_task(pump_out())
    await ws.send_text(json.dumps({"type": "ready", "pid": pid, "shell": shell}))
    try:
        while True:
            recv = asyncio.create_task(ws.receive())
            done, _ = await asyncio.wait({recv, out_task}, return_when=asyncio.FIRST_COMPLETED)
            if out_task in done:
                recv.cancel()
                break
            msg = recv.result()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                os.write(fd, msg["bytes"])
                continue
            data = json.loads(msg.get("text") or "{}")
            if data.get("type") == "input":
                os.write(fd, data["data"].encode())
            elif data.get("type") == "resize":
                resize(data["cols"], data["rows"])
    finally:
        out_task.cancel()
        try:
            loop.remove_reader(fd)
        except Exception:  # noqa: BLE001
            pass
        for sig in (signal.SIGHUP, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        os.close(fd)
