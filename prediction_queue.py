"""
Asynchronous prediction queue management for NightScan.

Provides a Redis-based queue system for handling ML predictions
asynchronously, preventing request thread blocking.
"""

import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Prediction task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PredictionTask:
    """Represents a prediction task."""
    task_id: str
    user_id: Optional[int]
    file_path: str
    file_hash: str
    filename: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    progress: int = 0
    priority: int = 0  # Higher number = higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionTask':
        """Create from dictionary."""
        data['status'] = TaskStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


class PredictionQueue:
    """Redis-based prediction queue manager."""
    
    def __init__(self, redis_url: str, ttl: int = 3600):
        """
        Initialize prediction queue.
        
        Args:
            redis_url: Redis connection URL
            ttl: Time to live for completed tasks (seconds)
        """
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
        
        # Redis key prefixes
        self.QUEUE_KEY = "prediction:queue"
        self.TASK_PREFIX = "prediction:task:"
        self.USER_TASKS_PREFIX = "prediction:user:"
        self.STATS_KEY = "prediction:stats"
        
        # Test connection
        try:
            self.redis.ping()
            logger.info("Prediction queue initialized")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def create_task(
        self,
        file_path: str,
        file_hash: str,
        filename: str,
        user_id: Optional[int] = None,
        priority: int = 0
    ) -> str:
        """
        Create a new prediction task.
        
        Args:
            file_path: Path to the file to process
            file_hash: Hash of the file
            filename: Original filename
            user_id: Optional user ID
            priority: Task priority (higher = more important)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = PredictionTask(
            task_id=task_id,
            user_id=user_id,
            file_path=file_path,
            file_hash=file_hash,
            filename=filename,
            status=TaskStatus.PENDING,
            created_at=datetime.utcnow(),
            priority=priority
        )
        
        # Store task data
        task_key = f"{self.TASK_PREFIX}{task_id}"
        self.redis.setex(
            task_key,
            self.ttl,
            json.dumps(task.to_dict())
        )
        
        # Add to queue (priority queue using sorted set)
        # Score = priority * 1000000 - timestamp (higher priority, older tasks first)
        score = priority * 1000000 - datetime.utcnow().timestamp()
        self.redis.zadd(self.QUEUE_KEY, {task_id: score})
        
        # Track user's tasks
        if user_id:
            user_key = f"{self.USER_TASKS_PREFIX}{user_id}"
            self.redis.sadd(user_key, task_id)
            self.redis.expire(user_key, self.ttl)
        
        # Update stats
        self.redis.hincrby(self.STATS_KEY, "tasks_created", 1)
        
        logger.info(f"Created prediction task {task_id}")
        return task_id
    
    def get_next_task(self) -> Optional[PredictionTask]:
        """
        Get the next task from the queue.
        
        Returns:
            Next task or None if queue is empty
        """
        # Get highest priority task
        task_ids = self.redis.zrange(self.QUEUE_KEY, 0, 0)
        if not task_ids:
            return None
        
        task_id = task_ids[0]
        
        # Remove from queue
        self.redis.zrem(self.QUEUE_KEY, task_id)
        
        # Get task data
        task_key = f"{self.TASK_PREFIX}{task_id}"
        task_data = self.redis.get(task_key)
        
        if not task_data:
            logger.warning(f"Task {task_id} not found")
            return None
        
        # Update task status
        task = PredictionTask.from_dict(json.loads(task_data))
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.utcnow()
        
        # Save updated task
        self.redis.setex(
            task_key,
            self.ttl,
            json.dumps(task.to_dict())
        )
        
        # Update stats
        self.redis.hincrby(self.STATS_KEY, "tasks_processing", 1)
        
        return task
    
    def update_task_progress(self, task_id: str, progress: int) -> bool:
        """
        Update task progress (0-100).
        
        Args:
            task_id: Task ID
            progress: Progress percentage
            
        Returns:
            True if updated successfully
        """
        task_key = f"{self.TASK_PREFIX}{task_id}"
        task_data = self.redis.get(task_key)
        
        if not task_data:
            return False
        
        task = PredictionTask.from_dict(json.loads(task_data))
        task.progress = min(100, max(0, progress))
        
        self.redis.setex(
            task_key,
            self.ttl,
            json.dumps(task.to_dict())
        )
        
        return True
    
    def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        Mark task as completed with results.
        
        Args:
            task_id: Task ID
            result: Prediction results
            
        Returns:
            True if updated successfully
        """
        task_key = f"{self.TASK_PREFIX}{task_id}"
        task_data = self.redis.get(task_key)
        
        if not task_data:
            return False
        
        task = PredictionTask.from_dict(json.loads(task_data))
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.result = result
        task.progress = 100
        
        # Calculate processing time
        if task.started_at:
            processing_time = (task.completed_at - task.started_at).total_seconds()
            result['processing_time'] = processing_time
        
        # Save with extended TTL for completed tasks
        self.redis.setex(
            task_key,
            self.ttl * 2,  # Keep completed tasks longer
            json.dumps(task.to_dict())
        )
        
        # Update stats
        self.redis.hincrby(self.STATS_KEY, "tasks_completed", 1)
        self.redis.hincrby(self.STATS_KEY, "tasks_processing", -1)
        
        logger.info(f"Completed task {task_id}")
        return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """
        Mark task as failed.
        
        Args:
            task_id: Task ID
            error: Error message
            
        Returns:
            True if updated successfully
        """
        task_key = f"{self.TASK_PREFIX}{task_id}"
        task_data = self.redis.get(task_key)
        
        if not task_data:
            return False
        
        task = PredictionTask.from_dict(json.loads(task_data))
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.utcnow()
        task.error = error
        
        self.redis.setex(
            task_key,
            self.ttl,
            json.dumps(task.to_dict())
        )
        
        # Update stats
        self.redis.hincrby(self.STATS_KEY, "tasks_failed", 1)
        self.redis.hincrby(self.STATS_KEY, "tasks_processing", -1)
        
        logger.error(f"Task {task_id} failed: {error}")
        return True
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status and progress.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status dict or None
        """
        task_key = f"{self.TASK_PREFIX}{task_id}"
        task_data = self.redis.get(task_key)
        
        if not task_data:
            return None
        
        task = PredictionTask.from_dict(json.loads(task_data))
        
        status = {
            'task_id': task.task_id,
            'status': task.status.value,
            'progress': task.progress,
            'created_at': task.created_at.isoformat(),
            'filename': task.filename
        }
        
        if task.started_at:
            status['started_at'] = task.started_at.isoformat()
        
        if task.completed_at:
            status['completed_at'] = task.completed_at.isoformat()
            
        if task.error:
            status['error'] = task.error
            
        if task.result and task.status == TaskStatus.COMPLETED:
            status['result_available'] = True
            
        return status
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task results if completed.
        
        Args:
            task_id: Task ID
            
        Returns:
            Results dict or None
        """
        task_key = f"{self.TASK_PREFIX}{task_id}"
        task_data = self.redis.get(task_key)
        
        if not task_data:
            return None
        
        task = PredictionTask.from_dict(json.loads(task_data))
        
        if task.status != TaskStatus.COMPLETED:
            return None
            
        return task.result
    
    def get_user_tasks(
        self,
        user_id: int,
        include_completed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all tasks for a user.
        
        Args:
            user_id: User ID
            include_completed: Include completed tasks
            
        Returns:
            List of task status dicts
        """
        user_key = f"{self.USER_TASKS_PREFIX}{user_id}"
        task_ids = self.redis.smembers(user_key)
        
        tasks = []
        for task_id in task_ids:
            status = self.get_task_status(task_id)
            if status:
                if include_completed or status['status'] != TaskStatus.COMPLETED.value:
                    tasks.append(status)
        
        # Sort by creation time
        tasks.sort(key=lambda x: x['created_at'], reverse=True)
        
        return tasks
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled successfully
        """
        # Remove from queue if still pending
        removed = self.redis.zrem(self.QUEUE_KEY, task_id)
        
        # Update task status
        task_key = f"{self.TASK_PREFIX}{task_id}"
        task_data = self.redis.get(task_key)
        
        if not task_data:
            return False
        
        task = PredictionTask.from_dict(json.loads(task_data))
        
        if task.status not in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        
        self.redis.setex(
            task_key,
            self.ttl,
            json.dumps(task.to_dict())
        )
        
        # Update stats
        if removed:
            self.redis.hincrby(self.STATS_KEY, "tasks_cancelled", 1)
        
        return True
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = self.redis.hgetall(self.STATS_KEY)
        
        # Convert to integers
        for key in stats:
            try:
                stats[key] = int(stats[key])
            except ValueError:
                pass
        
        # Add current queue size
        stats['queue_size'] = self.redis.zcard(self.QUEUE_KEY)
        
        return stats
    
    def cleanup_expired_tasks(self) -> int:
        """
        Clean up expired tasks.
        
        Returns:
            Number of tasks cleaned up
        """
        # This is handled automatically by Redis TTL
        # But we can clean up orphaned user task references
        
        cleaned = 0
        
        # Get all user keys
        for key in self.redis.scan_iter(f"{self.USER_TASKS_PREFIX}*"):
            user_id = key.split(":")[-1]
            task_ids = self.redis.smembers(key)
            
            for task_id in task_ids:
                task_key = f"{self.TASK_PREFIX}{task_id}"
                if not self.redis.exists(task_key):
                    self.redis.srem(key, task_id)
                    cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} orphaned task references")
        
        return cleaned


# Singleton instance
_queue_instance = None


def get_prediction_queue(redis_url: Optional[str] = None) -> PredictionQueue:
    """
    Get or create prediction queue instance.
    
    Args:
        redis_url: Redis URL (uses default from config if not provided)
        
    Returns:
        PredictionQueue instance
    """
    global _queue_instance
    
    if _queue_instance is None:
        if redis_url is None:
            from config import get_config
            config = get_config()
            redis_url = config.redis.url
        
        _queue_instance = PredictionQueue(redis_url)
    
    return _queue_instance