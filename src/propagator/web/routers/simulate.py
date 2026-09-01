from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from propagator.web.deps import get_job_manager
from propagator.web.jobs import JobBusyError, JobManager
from propagator.web.render import (
    bounds_wgs84,
    fire_probability_png,
    isochrones_wgs84,
)
from propagator.web.runner import run_job
from propagator.web.schemas import (
    FrameOut,
    FrameStats,
    Isochrone,
    JobFrames,
    JobSummary,
    SimulateRequest,
)

router = APIRouter(prefix="/api/simulate", tags=["simulate"])


def _get_job_or_404(job_id: str, manager: JobManager):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job


@router.post("", status_code=202)
def start_simulation(
    request: SimulateRequest, manager: JobManager = Depends(get_job_manager)
) -> dict:
    try:
        job_id = manager.submit(request, run_job)
    except JobBusyError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return {"job_id": job_id}


@router.get("/{job_id}")
def get_status(
    job_id: str, manager: JobManager = Depends(get_job_manager)
) -> JobSummary:
    job = _get_job_or_404(job_id, manager)
    return JobSummary(
        id=job.id,
        status=job.status.value,
        current_time_s=job.current_time_s,
        time_limit_s=job.time_limit_s,
        warning=job.warning,
        error=job.error,
    )


@router.get("/{job_id}/frames")
def get_frames(
    job_id: str, manager: JobManager = Depends(get_job_manager)
) -> JobFrames:
    job = _get_job_or_404(job_id, manager)
    bounds = bounds_wgs84(job.geo_info) if job.geo_info is not None else None
    stats_history = [
        FrameStats(time_s=t, **job.frames[t].stats)
        for t in job.frame_times
        if t in job.frames
    ]
    return JobFrames(
        id=job.id,
        status=job.status.value,
        bounds_wgs84=bounds,
        frame_times_s=list(job.frame_times),
        stats_history=stats_history,
    )


def _get_frame_or_404(job_id: str, time_s: int, manager: JobManager):
    job = _get_job_or_404(job_id, manager)
    frame = job.frames.get(time_s)
    if frame is None or job.geo_info is None:
        raise HTTPException(status_code=404, detail="Unknown frame")
    return job, frame


@router.get("/{job_id}/frame/{time_s}")
def get_frame(
    job_id: str,
    time_s: int,
    manager: JobManager = Depends(get_job_manager),
) -> FrameOut:
    job, frame = _get_frame_or_404(job_id, time_s, manager)
    if frame.isochrone_cache is None:
        raw = isochrones_wgs84(
            frame.fire_probability,
            job.geo_info,
            job.request.isochrone_thresholds,
        )
        frame.isochrone_cache = [
            Isochrone(
                threshold=t,
                coordinates=[[list(pt) for pt in line] for line in coords],
            )
            for t, coords in raw
        ]
    isochrones = frame.isochrone_cache
    return FrameOut(
        time_s=time_s,
        isochrones=isochrones,
        stats=FrameStats(time_s=time_s, **frame.stats),
    )


@router.get("/{job_id}/frame/{time_s}/image.png")
def get_frame_png(
    job_id: str,
    time_s: int,
    manager: JobManager = Depends(get_job_manager),
) -> Response:
    _, frame = _get_frame_or_404(job_id, time_s, manager)
    if frame.png_cache is None:
        frame.png_cache = fire_probability_png(frame.fire_probability)
    return Response(content=frame.png_cache, media_type="image/png")


@router.post("/{job_id}/cancel")
def cancel_simulation(
    job_id: str, manager: JobManager = Depends(get_job_manager)
) -> dict:
    if not manager.cancel(job_id):
        raise HTTPException(status_code=404, detail="Unknown job id")
    return {"cancelled": True}


@router.delete("/{job_id}")
def delete_simulation(
    job_id: str, manager: JobManager = Depends(get_job_manager)
) -> dict:
    if not manager.delete(job_id):
        raise HTTPException(status_code=404, detail="Unknown job id")
    return {"deleted": True}
