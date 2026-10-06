"""Resume-safe JSONL task log: one line per state change of a patch export; the latest line per patch wins."""
from __future__ import annotations

import datetime as dt
import json
import os

COMPLETED = "COMPLETED"
FAILED = "FAILED"
PENDING_STATES = frozenset({"SUBMITTED", "READY", "RUNNING", "UNSUBMITTED"})
DONE_STATES = frozenset({COMPLETED})


class TaskLog:
    """Append-only JSONL log of export tasks (patch id, backend, task id, state, UTC timestamp, info)."""

    def __init__(self, path: str) -> None:
        """Open (or lazily create) the log at `path`."""
        self.path = path

    def append(self, patch_id: str, state: str, backend: str, task_id: str = "", info: str = "") -> None:
        """Append one state record and flush it to disk immediately."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        record = {"patch_id": patch_id, "state": state, "backend": backend, "task_id": task_id,
                  "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "info": info}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    def records(self) -> list[dict]:
        """All parseable records in file order (a torn last line from a crash is ignored)."""
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def latest(self) -> dict[str, dict]:
        """Latest record per patch id."""
        return {r["patch_id"]: r for r in self.records()}

    def failures(self, patch_id: str) -> int:
        """Number of FAILED records for a patch."""
        return sum(1 for r in self.records() if r["patch_id"] == patch_id and r["state"] == FAILED)

    def should_skip(self, patch_id: str, max_attempts: int) -> bool:
        """True if the patch is done, still pending in a batch queue, or has used up its attempts."""
        last = self.latest().get(patch_id)
        if last is None:
            return False
        if last["state"] in DONE_STATES or last["state"] in PENDING_STATES:
            return True
        return self.failures(patch_id) >= max_attempts

    def pending_tasks(self) -> dict[str, str]:
        """Map task id -> patch id for batch tasks whose latest state is still pending."""
        return {r["task_id"]: pid for pid, r in self.latest().items() if r["state"] in PENDING_STATES and r["task_id"]}
