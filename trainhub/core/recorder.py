"""Append-only JSONL metric log.

Training runs are long and users close the app; persisting every metric row lets
the chart widget rebuild a run's curves later instead of losing them.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .events import MetricEvent

METRICS_FILENAME = "metrics.jsonl"


class MetricRecorder:
    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / METRICS_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, event: MetricEvent) -> None:
        self._fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass

    def __enter__(self) -> MetricRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_metrics(path: str | Path) -> list[MetricEvent]:
    path = Path(path)
    if path.is_dir():
        path = path / METRICS_FILENAME
    if not path.exists():
        return []
    events: list[MetricEvent] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(
                MetricEvent(
                    epoch=int(raw.get("epoch", 0)),
                    total_epochs=int(raw.get("total_epochs", 0)),
                    metrics={
                        str(k): float(v) for k, v in (raw.get("metrics") or {}).items()
                    },
                    phase=str(raw.get("phase", "train")),
                )
            )
    return events


def series_names(events: list[MetricEvent]) -> list[str]:
    names: list[str] = []
    for event in events:
        for key in event.metrics:
            if key not in names:
                names.append(key)
    return names
