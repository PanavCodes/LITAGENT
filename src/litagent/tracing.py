"""Local JSONL tracing and observability for LitAgent.

Per ARCHITECTURE.md §7 & FR-OBS-1:
Logs execution events for every agent node to results/traces/<run_id>.jsonl.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from litagent.schemas import TraceEvent

logger = logging.getLogger(__name__)


class Tracer:
    """Manages recording of trace events to local JSONL files."""

    def __init__(self, run_id: str | None = None, traces_dir: str | Path = "results/traces") -> None:
        self.run_id = run_id or f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.traces_dir / f"{self.run_id}.jsonl"
        self.events: list[TraceEvent] = []

    def record(
        self,
        node: str,
        start_time: float,
        input_summary: str,
        output_summary: str,
        error: str | None = None,
    ) -> TraceEvent:
        duration = time.time() - start_time
        event = TraceEvent(
            node=node,
            start=start_time,
            duration=duration,
            input_summary=input_summary,
            output_summary=output_summary,
            error=error,
        )
        self.events.append(event)
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except OSError as e:
            logger.warning(f"Failed to write trace event to {self.trace_file}: {e}")
        return event

    def get_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_events": len(self.events),
            "nodes": [e.node for e in self.events],
            "total_duration": sum(e.duration for e in self.events),
        }
