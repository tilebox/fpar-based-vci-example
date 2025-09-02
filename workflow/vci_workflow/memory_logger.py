from guppy import hpy
from humanize import naturalsize
from tilebox.workflows.observability.logging import get_logger

logger = get_logger()


class MemoryLogger:
    def __init__(self) -> None:
        self.heap_tracker = hpy()
        # set current memory as baseline, to track only usage from now on
        self.heap_tracker.setrelheap()

    def log_snapshot(self, n_largest: int = 7) -> None:
        heap = self.heap_tracker.heap()

        sizes = {}
        for i in range(n_largest):
            sizes[str(heap[i].kind)] = naturalsize(heap[i].size)

        logger.info(f"memory snapshot: total heap size = {naturalsize(heap.size)}, top {n_largest} = {sizes}")
