"""Teaching-scale continuous batching scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class RequestLifecycle(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True)
class ContinuousBatchRequest:
    request_id: str
    prompt: str
    max_new_tokens: int = 32
    arrival_step: int = 0


@dataclass(frozen=True)
class SchedulerStepStats:
    step: int
    admitted_request_ids: list[str]
    running_request_ids: list[str]
    completed_request_ids: list[str]
    active_batch_size: int


@dataclass
class ScheduledRequestState:
    request: ContinuousBatchRequest
    lifecycle: RequestLifecycle = RequestLifecycle.WAITING
    admitted_step: int | None = None
    finished_step: int | None = None


@dataclass
class ContinuousBatchScheduler:
    requests: list[ContinuousBatchRequest]
    max_batch_size: int | None = None
    waiting: deque[ScheduledRequestState] = field(init=False)
    running: list[ScheduledRequestState] = field(default_factory=list, init=False)
    finished: list[ScheduledRequestState] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.max_batch_size is not None and self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")

        seen: set[str] = set()
        indexed_states: list[tuple[int, ScheduledRequestState]] = []
        for index, request in enumerate(self.requests):
            if request.request_id in seen:
                raise ValueError(f"duplicate request_id: {request.request_id}")
            seen.add(request.request_id)
            if request.arrival_step < 0:
                raise ValueError("arrival_step must be non-negative")
            if request.max_new_tokens < 1:
                raise ValueError("max_new_tokens must be at least 1")
            if not request.prompt:
                raise ValueError("prompt must not be empty")
            indexed_states.append((index, ScheduledRequestState(request=request)))

        indexed_states.sort(key=lambda item: (item[1].request.arrival_step, item[0]))
        self.waiting = deque(state for _, state in indexed_states)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def next_arrival_step(self) -> int | None:
        if not self.waiting:
            return None
        return self.waiting[0].request.arrival_step

    def admit(self, step: int) -> list[ScheduledRequestState]:
        admitted: list[ScheduledRequestState] = []
        while self.waiting and self.waiting[0].request.arrival_step <= step:
            if self.max_batch_size is not None and len(self.running) >= self.max_batch_size:
                break
            state = self.waiting.popleft()
            state.lifecycle = RequestLifecycle.RUNNING
            state.admitted_step = step
            self.running.append(state)
            admitted.append(state)
        return admitted

    def finish(self, request_ids: set[str], step: int) -> list[ScheduledRequestState]:
        completed: list[ScheduledRequestState] = []
        still_running: list[ScheduledRequestState] = []
        for state in self.running:
            if state.request.request_id in request_ids:
                state.lifecycle = RequestLifecycle.FINISHED
                state.finished_step = step
                completed.append(state)
            else:
                still_running.append(state)
        self.running = still_running
        self.finished.extend(completed)
        return completed

    def record_step(
        self,
        *,
        step: int,
        admitted: list[ScheduledRequestState],
        running_request_ids: list[str],
        completed: list[ScheduledRequestState],
    ) -> SchedulerStepStats:
        return SchedulerStepStats(
            step=step,
            admitted_request_ids=[state.request.request_id for state in admitted],
            running_request_ids=running_request_ids,
            completed_request_ids=[state.request.request_id for state in completed],
            active_batch_size=len(running_request_ids),
        )
