"""Ingestion layer (03-ingestion.md): pipeline, dedup, cost gates, reindex."""

from ragx.ingestion.costs import (
    BatchConfig,
    DegradationConfig,
    batch_describe,
    describe_with_cache,
    describe_with_degradation,
    process_atoms,
)
from ragx.ingestion.dedup import submit_ingest
from ragx.ingestion.pipeline import IngestionPipeline, RetryPolicy, run_stage
from ragx.ingestion.queue import InProcessQueue, TaskQueue
from ragx.ingestion.queue_redis import RedisStreamsQueue, make_queue
from ragx.ingestion.reindex import incremental_reindex
from ragx.ingestion.store import MetadataStore

__all__ = [
    "BatchConfig",
    "DegradationConfig",
    "IngestionPipeline",
    "InProcessQueue",
    "MetadataStore",
    "RedisStreamsQueue",
    "RetryPolicy",
    "TaskQueue",
    "batch_describe",
    "describe_with_cache",
    "describe_with_degradation",
    "incremental_reindex",
    "make_queue",
    "process_atoms",
    "run_stage",
    "submit_ingest",
]
