import tracemalloc
import logging
import psutil
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryLogger:
    def __init__(self):
        tracemalloc.start()
        self.process = psutil.Process()
        
    def log_snapshot(self, label=""):
        # Current memory usage
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        # Top memory allocations
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        logger.info(f"=== Memory Snapshot {label} ===")
        logger.info(f"Total Memory: {memory_mb:.1f} MB")
        
        for i, stat in enumerate(top_stats[:5]):
            logger.info(f"  #{i+1}: {stat}")
            
    def compare_snapshots(self, old_snapshot, label=""):
        current = tracemalloc.take_snapshot()
        top_stats = current.compare_to(old_snapshot, 'lineno')
        
        logger.info(f"=== Memory Diff {label} ===")
        for stat in top_stats[:5]:
            logger.info(f"  {stat}")
            
        return current

# Usage
memory_logger = MemoryLogger()