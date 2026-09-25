# app/routers/jobs.py
"""
Background job status endpoint.

GET /api/jobs/{job_id}  — requires admin scope
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import require_admin
from app.models.schemas import JobStatusResponse
from app.services.job_store import get_job_store

router = APIRouter(prefix="/api", tags=["jobs"])
log = logging.getLogger(__name__)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    _role: str = Depends(require_admin),
) -> JobStatusResponse:
    """
    Retrieve the current status of a background job.

    Only accessible with the `admin` scope.
    Returns 404 if the job does not exist or has been purged.
    """
    store = get_job_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found. It may have completed and been purged.",
        )
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        result=job.result,
        error=job.error,
    )
