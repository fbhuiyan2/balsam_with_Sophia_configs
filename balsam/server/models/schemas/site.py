from datetime import datetime
from uuid import UUID
from typing import List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, validator, AnyUrl, Field
from pathlib import Path


class QueuedJobState(str, Enum):
    queued = "queued"
    running = "running"


class TransferProtocol(str, Enum):
    globus = "globus"
    rsync = "rsync"


class BackfillWindow(BaseModel):
    queue: str = Field(..., example="default")
    num_nodes: int = Field(..., example=130)
    wall_time_min: int = Field(..., example=40)


class QueuedJob(BaseModel):
    queue: str = Field(..., example="default")
    num_nodes: int = Field(..., example=128)
    wall_time_min: int = Field(..., example=60)
    state: QueuedJobState = Field(..., example=QueuedJobState.queued)


class AllowedQueue(BaseModel):
    max_nodes: int = Field(..., example=8)
    max_walltime: int = Field(..., example=60)
    max_queued_jobs: int = Field(..., example=1)


class SiteBase(BaseModel):
    hostname: str = Field(..., example="thetalogin3.alcf.anl.gov")
    path: Path = Field(..., example="/projects/datascience/user/mySite")
    globus_endpoint_id: Optional[UUID] = Field(None)
    num_nodes: int = Field(0, example=4096)
    backfill_windows: List[BackfillWindow] = Field([])
    queued_jobs: List[QueuedJob] = Field([])
    optional_batch_job_params: Dict[str, str] = Field({}, example={"enable_ssh": 1})
    allowed_projects: List[str] = Field([], example=["datascience", "materials-adsp"])
    allowed_queues: Dict[str, AllowedQueue] = Field(
        {},
        example={
            "debug-cache-quad": {
                "max_nodes": 8,
                "max_walltime": 60,
                "max_queued_jobs": 1,
            }
        },
    )
    transfer_locations: Dict[str, AnyUrl] = Field(
        {},
        example={
            "APS-DTN": "globus://ddb59aef-6d04-11e5-ba46-22000b92c6ec",
            "MyCluster": "rsync://user@hostname.mycluster",
        },
    )

    @validator("path")
    def path_is_absolute(cls, v: Path):
        if not v.is_absolute():
            raise ValueError("path must be absolute")
        return v


class SiteCreate(SiteBase):
    pass


class SiteUpdate(SiteBase):
    hostname: Optional[str] = Field(None, example="thetalogin3.alcf.anl.gov")
    path: Optional[Path] = Field(None, example="/projects/datascience/user/mySite")


class SiteOut(SiteBase):
    class Config:
        orm_mode = True

    id: int
    last_refresh: datetime
    creation_date: datetime
