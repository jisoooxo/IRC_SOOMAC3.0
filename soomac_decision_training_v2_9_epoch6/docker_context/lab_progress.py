"""Training Image Spec 2.7 progress writer. Standard-library only."""
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid


def format_eta(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    total = math.ceil(seconds)
    return f"{total // 3600:03d}:{total // 60 % 60:02d}:{total % 60:02d}"


class Progress:
    def __init__(
        self,
        start_step=0,
        target_step=None,
        sample_unit="sequences",
        enabled=True,
        interval=2,
        console=True,
    ):
        self.path = (
            Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
            / ".lab-monitor"
            / "progress.json"
        )
        self.enabled = enabled
        self.interval = interval
        self.last = -float("inf")
        self.console = console
        self.warned = set()
        self.history = deque(maxlen=120)
        self._lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat = None
        self.value = {
            "schema_version": 1,
            "attempt_id": uuid.uuid4().hex,
            "sequence": 0,
            "phase": "loading",
            "updated_at": time.time(),
            "start_step": start_step,
            "global_step": start_step,
            "target_step": target_step,
            "training_seconds": 0.0,
            "samples_seen": None,
            "tokens_seen": None,
            "sample_unit": sample_unit,
            "scope": "job",
            "skipped_updates": 0,
            "loss": None,
            "learning_rate": None,
        }

    def _warn(self, channel):
        if channel in self.warned:
            return
        self.warned.add(channel)
        try:
            print(
                f"Lab progress {channel} unavailable; training continues.",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass

    def _console_line(self):
        latest = self.value
        earlier = [
            item
            for item in self.history
            if item["training_seconds"] < latest["training_seconds"]
        ]
        baseline = next(
            (
                item
                for item in reversed(earlier)
                if latest["training_seconds"] - item["training_seconds"] >= 30
            ),
            earlier[0] if earlier else None,
        )
        step_rate = sample_rate = eta = None
        if baseline:
            elapsed = latest["training_seconds"] - baseline["training_seconds"]
            step_rate = (latest["global_step"] - baseline["global_step"]) / elapsed
            if latest["samples_seen"] is not None and baseline["samples_seen"] is not None:
                sample_rate = (
                    latest["samples_seen"] - baseline["samples_seen"]
                ) / elapsed
        end = latest["target_step"]
        if latest["phase"] == "training" and end is not None and step_rate and step_rate > 0:
            eta = (end - latest["global_step"]) / step_rate

        def rate(value):
            if value is None or not math.isfinite(value) or value < 0:
                return "—"
            return f"{value:.2f}"

        goal = str(end) if end is not None else "?"
        run_goal = str(end - latest["start_step"]) if end is not None else "?"
        return (
            f"[LAB_PROGRESS] phase={latest['phase']} "
            f"step={latest['global_step']}/{goal} "
            f"run={latest['global_step'] - latest['start_step']}/{run_goal} "
            f"step/s={rate(step_rate)} sample/s={rate(sample_rate)} "
            f"ETA={format_eta(eta)}"
        )

    def start_heartbeat(self, interval=10):
        if not self.enabled:
            return
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Heartbeat interval must be positive")
        with self._lock:
            if self._heartbeat is not None:
                return
            self.write(force=True)

            def report():
                while not self._heartbeat_stop.wait(interval):
                    self.write(force=True)

            self._heartbeat = threading.Thread(
                target=report,
                daemon=True,
                name="lab-progress",
            )
            self._heartbeat.start()

    def close(self):
        self._heartbeat_stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1)

    def write(self, *, force=False, **values):
        with self._lock:
            self._write(force=force, **values)

    def _write(self, *, force=False, **values):
        if not self.enabled:
            return
        try:
            protected = {"schema_version", "attempt_id", "sequence", "updated_at", "scope"}
            if set(values) - set(self.value) or set(values) & protected:
                raise ValueError("Unsupported progress fields")
            changed_phase = values.get("phase", self.value["phase"]) != self.value["phase"]
            self.value.update(values)
            now = time.monotonic()
            if not force and not changed_phase and now - self.last < self.interval:
                return
            self.value.update(
                sequence=self.value["sequence"] + 1,
                updated_at=time.time(),
            )
            raw = json.dumps(self.value, allow_nan=False)
            self.last = now
        except Exception:
            self._warn("values")
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.value["attempt_id"] + ".tmp")
            temporary.write_text(raw, encoding="utf-8")
            temporary.replace(self.path)
        except Exception:
            self._warn("file write")

        try:
            if self.console:
                print(self._console_line(), flush=True)
        except Exception:
            self._warn("console")
        self.history.append(self.value.copy())

