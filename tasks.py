"""
Generic async task polling utilities for long-running API operations.

This module provides reusable utilities for handling async task patterns
where an API returns a task ID immediately and the client polls for completion.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TaskStatus(Enum):
    """Status of an async task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskProgress:
    """Progress information for an async task."""

    status: TaskStatus
    progress_percent: int | None = None
    eta_seconds: int | None = None
    message: str | None = None
    error: str | None = None


class TaskPollingError(Exception):
    """Raised when task polling fails."""

    pass


class AsyncTaskPoller:
    """
    Generic poller for async task patterns.

    Handles polling a remote API for task completion with configurable
    retry logic, timeout handling, and progress callbacks.
    """

    def __init__(
        self,
        poll_interval: float = 15.0,
        max_poll_duration: float = 3600.0,
        http_timeout: float = 30.0,
    ):
        """
        Initialize the task poller.

        Args:
            poll_interval: Seconds to wait between status checks
            max_poll_duration: Maximum total time to poll before giving up
            http_timeout: Timeout for individual HTTP requests
        """
        self.poll_interval = poll_interval
        self.max_poll_duration = max_poll_duration
        self.http_timeout = http_timeout

    async def poll_until_complete(
        self,
        task_id: str,
        status_url: str,
        result_url: str | None = None,
        on_progress: Callable[[TaskProgress], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """
        Poll a task until it completes or fails.

        Args:
            task_id: The task identifier
            status_url: URL to check task status (can include {task_id} placeholder)
            result_url: URL to fetch final result (if different from status_url)
            on_progress: Optional async callback for progress updates

        Returns:
            The final result data from the API

        Raises:
            TaskPollingError: If polling fails or times out
        """
        status_url = status_url.format(task_id=task_id)
        if result_url:
            result_url = result_url.format(task_id=task_id)

        start_time = asyncio.get_event_loop().time()
        poll_count = 0

        logger.info(f"Starting to poll task {task_id} at {status_url}")

        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > self.max_poll_duration:
                    raise TaskPollingError(
                        f"Task {task_id} polling timed out after {elapsed:.1f}s"
                    )

                poll_count += 1

                try:
                    # Check task status
                    response = await client.get(status_url)
                    response.raise_for_status()
                    data = response.json()

                    # Parse status
                    progress = self._parse_progress(data)

                    logger.debug(
                        f"Task {task_id} poll #{poll_count}: {progress.status.value} "
                        f"(elapsed: {elapsed:.1f}s)"
                    )

                    # Send progress update
                    if on_progress:
                        await on_progress(progress)

                    # Check if complete
                    if progress.status == TaskStatus.COMPLETED:
                        logger.info(
                            f"Task {task_id} completed after {elapsed:.1f}s "
                            f"({poll_count} polls)"
                        )
                        # Fetch result if separate endpoint
                        if result_url and result_url != status_url:
                            result_response = await client.get(result_url)
                            result_response.raise_for_status()
                            return result_response.json()
                        return data

                    # Check if failed
                    if progress.status == TaskStatus.FAILED:
                        error_msg = progress.error or "Unknown error"
                        logger.error(f"Task {task_id} failed: {error_msg}")
                        raise TaskPollingError(f"Task failed: {error_msg}")

                    # Wait before next poll
                    await asyncio.sleep(self.poll_interval)

                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"HTTP error polling task {task_id}: "
                        f"{e.response.status_code} - {e.response.text}"
                    )
                    raise TaskPollingError(
                        f"HTTP {e.response.status_code}: {e.response.text}"
                    )
                except httpx.ConnectError as e:
                    logger.error(f"Connection error polling task {task_id}: {e}")
                    raise TaskPollingError(f"Cannot connect to API: {e}")
                except Exception as e:
                    logger.exception(f"Unexpected error polling task {task_id}")
                    raise TaskPollingError(f"Polling error: {e}")

    def _parse_progress(self, data: dict[str, Any]) -> TaskProgress:
        """
        Parse progress information from API response.

        Override this method to customize parsing for different API formats.

        Args:
            data: Raw API response data

        Returns:
            Parsed TaskProgress object
        """
        # Default parsing - handles common patterns
        status_str = data.get("status", "unknown").lower()

        # Map common status strings to TaskStatus enum
        status_map = {
            "pending": TaskStatus.PENDING,
            "queued": TaskStatus.PENDING,
            "processing": TaskStatus.PROCESSING,
            "running": TaskStatus.PROCESSING,
            "in_progress": TaskStatus.PROCESSING,
            "completed": TaskStatus.COMPLETED,
            "succeeded": TaskStatus.COMPLETED,
            "done": TaskStatus.COMPLETED,
            "failed": TaskStatus.FAILED,
            "error": TaskStatus.FAILED,
            "canceled": TaskStatus.FAILED,
            "cancelled": TaskStatus.FAILED,
        }

        status = status_map.get(status_str, TaskStatus.PENDING)

        return TaskProgress(
            status=status,
            progress_percent=data.get("progress"),
            eta_seconds=data.get("eta_seconds") or data.get("eta"),
            message=data.get("message") or data.get("status_message"),
            error=data.get("error") or data.get("error_message"),
        )


async def simple_poll_task(
    task_id: str,
    base_url: str,
    status_endpoint: str = "/task/{task_id}",
    result_endpoint: str | None = None,
    poll_interval: float = 15.0,
    max_duration: float = 3600.0,
    on_progress: Callable[[TaskProgress], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """
    Convenience function for simple task polling.

    Args:
        task_id: Task identifier
        base_url: Base URL of the API (e.g., "http://localhost:8888")
        status_endpoint: Status endpoint path (can include {task_id} placeholder)
        result_endpoint: Result endpoint path (if different from status)
        poll_interval: Seconds between polls
        max_duration: Maximum polling duration
        on_progress: Optional progress callback

    Returns:
        Final result data

    Raises:
        TaskPollingError: If polling fails
    """
    poller = AsyncTaskPoller(
        poll_interval=poll_interval,
        max_poll_duration=max_duration,
    )

    status_url = f"{base_url.rstrip('/')}{status_endpoint}"
    result_url = f"{base_url.rstrip('/')}{result_endpoint}" if result_endpoint else None

    return await poller.poll_until_complete(
        task_id=task_id,
        status_url=status_url,
        result_url=result_url,
        on_progress=on_progress,
    )
