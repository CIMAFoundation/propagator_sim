from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from propagator.web.deps import get_job_manager
from propagator.web.jobs import FrameData, JobBusyError, JobManager, JobState
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
    POIArrivalOut,
    POIOut,
    SimulateRequest,
)

router = APIRouter(prefix="/api/simulate", tags=["simulate"])


def _get_job_or_404(job_id: str, manager: JobManager):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job


def _poi_out_list(job: JobState) -> list[POIOut]:
    return [
        POIOut(
            id=p.key,
            name=p.name,
            category=p.category,
            lat=p.lat,
            lon=p.lon,
            voltage=p.voltage,
            operator=p.operator,
            geometry=list(p.geometry) if p.geometry else None,
        )
        for p in job.pois
    ]


def _base_poi_key(sample_key: str) -> str:
    """Strip the "#<index>" suffix `runner.build_sample_cells` appends to
    every sample (one or more per POI, for line/polygon geometries)."""
    return sample_key.rsplit("#", 1)[0]


def _poi_arrival_out_list(
    job: JobState, frame: FrameData
) -> list[POIArrivalOut]:
    if frame.poi_arrival_cache is None:
        by_key = {p.key: p for p in job.pois}
        grouped: dict[str, list] = {}
        for sample in frame.poi_arrival:
            grouped.setdefault(_base_poi_key(sample.key), []).append(sample)

        out = []
        for base_key, samples in grouped.items():
            poi = by_key.get(base_key)
            if poi is None:
                continue
            reached = any(s.reached for s in samples)
            arrivals_h = [
                s.mean_arrival_time / 3600.0 for s in samples if s.reached
            ]
            out.append(
                POIArrivalOut(
                    id=base_key,
                    name=poi.name,
                    category=poi.category,
                    lat=poi.lat,
                    lon=poi.lon,
                    voltage=poi.voltage,
                    operator=poi.operator,
                    reached=reached,
                    arrival_time_h=min(arrivals_h) if arrivals_h else None,
                )
            )
        frame.poi_arrival_cache = out
    return frame.poi_arrival_cache


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
        poi_warning=job.poi_warning,
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
        pois=_poi_out_list(job),
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
        poi_arrival=_poi_arrival_out_list(job, frame),
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
