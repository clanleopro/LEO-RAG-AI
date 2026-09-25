# tests/test_jobs.py
import time
from app.services.job_store import JobStore, JobStatus

def sample_task(x: int) -> int:
    time.sleep(0.1)
    return x * 2

def failing_task():
    raise ValueError("Task failed")

def test_job_lifecycle():
    store = JobStore(max_workers=1)
    job_id = store.submit(sample_task, 5)
    
    job = store.get(job_id)
    assert job is not None
    assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)
    
    time.sleep(0.2)
    job = store.get(job_id)
    assert job.status == JobStatus.DONE
    assert job.result == {"value": 10}

def test_job_failure():
    store = JobStore(max_workers=1)
    job_id = store.submit(failing_task)
    
    time.sleep(0.1)
    job = store.get(job_id)
    assert job.status == JobStatus.FAILED
    assert "Task failed" in job.error
