# app/services/job_store.py
"""
Simple in-process background job store.

Provides a thread-safe job registry for tracking long-running ingestion
and OCR operations. Jobs are executed in a thread-pool executor.

⚠️  PRODUCTION NOTE:
This implementation uses a single-process in-memory store. For
multi-worker or multi-instance deployments, replace this with a
proper distributed task queue such as:
  - Celery + Redis/RabbitMQ
  - ARQ (asyncio-native Redis queue)
  - RQ (Redis Queue)
  - Dramatiq

The public API (submit_job, get_job, cancel_job) is designed to be
a drop-in abstraction point for such a replacement.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Job model
# ─────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0          # 0.0 – 1.0
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    idempotency_key: Optional[str] = None
    _future: Optional[Future] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# ─────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────

class JobStore:
    """
    Thread-safe in-memory job registry with a bounded thread pool.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._idempotency_index: Dict[str, str] = {}   # key → job_id
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="leo-worker"
        )

    # ── Submit ───────────────────────────────────────────────

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        idempotency_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """
        Submit a callable to run in the background.

        Args:
            fn:               The callable to run.
            *args, **kwargs:  Arguments forwarded to fn.
            idempotency_key:  If provided, a second submission with the same
                              key returns the existing job_id without re-running.
            timeout_seconds:  If provided, the job will be marked failed if it
                              does not finish within this many seconds.

        Returns the job_id string.
        """
        with self._lock:
            if idempotency_key:
                existing_id = self._idempotency_index.get(idempotency_key)
                if existing_id and existing_id in self._jobs:
                    existing = self._jobs[existing_id]
                    if existing.status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.DONE):
                        log.debug("Idempotency hit for key=%s → job_id=%s", idempotency_key, existing_id)
                        return existing_id

            job_id = str(uuid.uuid4())
            job = Job(job_id=job_id, idempotency_key=idempotency_key)
            self._jobs[job_id] = job
            if idempotency_key:
                self._idempotency_index[idempotency_key] = job_id

        def _run() -> None:
            with self._lock:
                j = self._jobs.get(job_id)
                if j is None or j.status == JobStatus.CANCELLED:
                    return
                j.status = JobStatus.RUNNING
                j.started_at = time.time()

            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    j = self._jobs.get(job_id)
                    if j and j.status == JobStatus.RUNNING:
                        j.status = JobStatus.DONE
                        j.progress = 1.0
                        j.result = result if isinstance(result, dict) else {"value": result}
                        j.finished_at = time.time()
            except Exception as exc:  # noqa: BLE001
                log.exception("Job %s failed: %s", job_id, exc)
                with self._lock:
                    j = self._jobs.get(job_id)
                    if j:
                        j.status = JobStatus.FAILED
                        j.error = str(exc)
                        j.finished_at = time.time()

        future = self._executor.submit(_run)

        with self._lock:
            self._jobs[job_id]._future = future

        if timeout_seconds:
            self._schedule_timeout(job_id, timeout_seconds)

        log.info("Job submitted: job_id=%s idempotency_key=%s", job_id, idempotency_key)
        return job_id

    def _schedule_timeout(self, job_id: str, timeout_seconds: float) -> None:
        def _check() -> None:
            time.sleep(timeout_seconds)
            with self._lock:
                j = self._jobs.get(job_id)
                if j and j.status == JobStatus.RUNNING:
                    log.warning("Job %s timed out after %.1fs", job_id, timeout_seconds)
                    j.status = JobStatus.FAILED
                    j.error = f"Timed out after {timeout_seconds:.0f}s"
                    j.finished_at = time.time()
                    if j._future:
                        j._future.cancel()

        t = threading.Thread(target=_check, daemon=True)
        t.start()

    # ── Read ─────────────────────────────────────────────────

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(self, job_id: str, progress: float, message: Optional[str] = None) -> None:
        """Call this from inside a running job to report progress."""
        with self._lock:
            j = self._jobs.get(job_id)
            if j and j.status == JobStatus.RUNNING:
                j.progress = max(0.0, min(1.0, progress))
                if message is not None:
                    j.message = message

    # ── Cancel ───────────────────────────────────────────────

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return False
            if j.status == JobStatus.PENDING:
                j.status = JobStatus.CANCELLED
                if j._future:
                    j._future.cancel()
                return True
            return False  # Cannot cancel running/done/failed jobs

    # ── Cleanup ──────────────────────────────────────────────

    def purge_old(self, max_age_seconds: float = 3600.0) -> int:
        """Remove completed/failed jobs older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        removed = 0
        with self._lock:
            to_remove = [
                jid for jid, j in self._jobs.items()
                if j.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)
                and j.finished_at is not None
                and j.finished_at < cutoff
            ]
            for jid in to_remove:
                j = self._jobs.pop(jid)
                if j.idempotency_key:
                    self._idempotency_index.pop(j.idempotency_key, None)
                removed += 1
        return removed

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


# ─────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────
_job_store: Optional[JobStore] = None
_store_lock = threading.Lock()


def get_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        with _store_lock:
            if _job_store is None:
                _job_store = JobStore(max_workers=4)
    return _job_store


__all__ = [
    "Job",
    "JobStatus",
    "JobStore",
    "get_job_store",
]
