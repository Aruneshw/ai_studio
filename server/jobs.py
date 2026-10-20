"""Runs long subprocesses (installs, training, exports, downloads) and keeps their logs."""
import os
import signal
import subprocess
import threading
import time
import uuid

MAX_LINES = 20000
_jobs = {}
_lock = threading.Lock()


class Job:
    def __init__(self, kind, title, cmd, cwd=None, env=None, log_file=None, meta=None, on_exit=None):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.title, self.cmd = kind, title, cmd
        self.cwd, self.env, self.meta = cwd, env, meta or {}
        self.log_file = log_file
        self.on_exit = on_exit
        self.lines, self.partial = [], ""
        self.dropped = 0
        self.status, self.returncode = "starting", None
        self.started, self.ended = time.time(), None
        self.proc = None

    def start(self):
        env = dict(os.environ, **(self.env or {}), PYTHONUNBUFFERED="1")
        shell = isinstance(self.cmd, str)
        self.proc = subprocess.Popen(
            self.cmd, shell=shell, executable="/bin/bash" if shell else None, cwd=self.cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
        self.status = "running"
        threading.Thread(target=self._pump, daemon=True).start()

    def _append(self, line):
        self.lines.append(line)
        if len(self.lines) > MAX_LINES:
            cut = len(self.lines) - MAX_LINES
            del self.lines[:cut]
            self.dropped += cut

    def _pump(self):
        fh = open(self.log_file, "a", encoding="utf-8") if self.log_file else None
        buf = ""
        while True:
            chunk = self.proc.stdout.read1(65536) if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            *done, buf = buf.split("\n")
            for line in done:
                line = line.rstrip("\r").rsplit("\r", 1)[-1]
                self._append(line)
                if fh:
                    fh.write(line + "\n")
            self.partial = buf.rsplit("\r", 1)[-1]
            if fh:
                fh.flush()
        if buf:
            self._append(buf.rstrip("\r").rsplit("\r", 1)[-1])
        self.partial = ""
        self.returncode = self.proc.wait()
        if self.status != "stopped":
            self.status = "succeeded" if self.returncode == 0 else "failed"
        self.ended = time.time()
        self._append(f"[process exited with code {self.returncode}]")
        if fh:
            fh.write(f"[process exited with code {self.returncode}]\n")
            fh.close()
        if self.on_exit:
            try:
                self.on_exit(self)
            except Exception:  # noqa: BLE001
                pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.status = "stopped"
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return

            def force():
                time.sleep(10)
                if self.proc.poll() is None:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            threading.Thread(target=force, daemon=True).start()

    def summary(self):
        return {"id": self.id, "kind": self.kind, "title": self.title,
                "command": self.cmd if isinstance(self.cmd, str) else " ".join(map(str, self.cmd)),
                "status": self.status, "returncode": self.returncode, "started": self.started,
                "ended": self.ended, "meta": self.meta, "lineCount": self.dropped + len(self.lines),
                "lastLine": self.partial or (self.lines[-1] if self.lines else "")}

    def log(self, since=0):
        start = max(0, since - self.dropped)
        return {**self.summary(), "lines": self.lines[start:],
                "next": self.dropped + len(self.lines), "partial": self.partial}


def start(kind, title, cmd, **kw):
    job = Job(kind, title, cmd, **kw)
    with _lock:
        _jobs[job.id] = job
    job.start()
    return job


def get(job_id):
    return _jobs[job_id]


def all_jobs():
    return sorted(_jobs.values(), key=lambda j: j.started, reverse=True)


def running(kind=None):
    return [j for j in _jobs.values() if j.status == "running" and (kind is None or j.kind == kind)]


def stop_all():
    for j in list(_jobs.values()):
        j.stop()
