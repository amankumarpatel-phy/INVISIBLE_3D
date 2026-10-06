"""Lightweight asynchronous job manager for INVISIBLE³D.

The manager intentionally keeps the first deployment simple. Jobs are run in
one background worker to protect the web server from concurrent CPU-heavy
reconstructions. Results are written to the local runtime directory; a later
production deployment can replace this storage with object storage.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np


class JobManager:
    """Thread-safe job registry with local NPZ/JSON result storage."""

    def __init__(self, root: str = "runtime/jobs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="invisible3d")
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def submit(self, kind: str, fn: Callable[[], Dict[str, Any]]) -> str:
        job_id = uuid.uuid4().hex
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)

        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "kind": kind,
                "status": "queued",
                "created_at": self._now(),
                "error": None,
                "result": None,
            }

        future = self._executor.submit(self._run, job_id, fn)
        with self._lock:
            self._jobs[job_id]["future"] = future
        return job_id

    def _run(self, job_id: str, fn: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_at"] = self._now()

        try:
            result = fn()
            metadata = result.get("metadata", {})
            with self._lock:
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["result"] = metadata
                self._jobs[job_id]["completed_at"] = self._now()
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = error_text
                self._jobs[job_id]["completed_at"] = self._now()

    def get(self, job_id: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {k: v for k, v in job.items() if k != "future"}

    def save_arrays(self, job_id: str, **arrays: np.ndarray) -> str:
        path = self.root / job_id / "data.npz"
        np.savez_compressed(path, **arrays)
        return str(path)

    def save_metadata(self, job_id: str, metadata: Dict[str, Any]) -> str:
        path = self.root / job_id / "metadata.json"
        serializable = json.loads(json.dumps(metadata, default=str))
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return str(path)

    def load_arrays(self, job_id: str) -> Dict[str, np.ndarray]:
        path = self.root / job_id / "data.npz"
        if not path.exists():
            raise FileNotFoundError(f"No stored data for job {job_id}")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def load_result_file(self, job_id: str) -> Path:
        path = self.root / job_id / "results.npz"
        if not path.exists():
            raise FileNotFoundError(f"No reconstruction result for job {job_id}")
        return path

    def save_result(self, job_id: str, **arrays: np.ndarray) -> str:
        path = self.root / job_id / "results.npz"
        np.savez_compressed(path, **arrays)
        return str(path)


jobs = JobManager()
