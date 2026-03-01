"""
Logging and monitoring utilities for Waste Classification System
"""
import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps
import threading
from collections import deque
import numpy as np

# Configure base logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_path = LOG_DIR / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class InferenceLogger:
    """Logger for inference requests and performance metrics"""
    
    def __init__(self, log_file: str = "inference.log"):
        self.logger = setup_logger("inference", log_file)
        self.metrics_file = LOG_DIR / "metrics.jsonl"
        
    def log_inference(self, 
                      request_id: str,
                      image_size: tuple,
                      predictions: Dict[str, float],
                      inference_time_ms: float,
                      model_version: str,
                      success: bool = True,
                      error: Optional[str] = None):
        """Log inference request details"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "image_size": image_size,
            "top_prediction": max(predictions, key=predictions.get) if predictions else None,
            "confidence": max(predictions.values()) if predictions else 0,
            "inference_time_ms": round(inference_time_ms, 2),
            "model_version": model_version,
            "success": success,
            "error": error
        }
        
        if success:
            self.logger.info(f"Inference completed: {json.dumps(log_entry)}")
        else:
            self.logger.error(f"Inference failed: {json.dumps(log_entry)}")
        
        # Write to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_batch_inference(self, batch_size: int, total_time_ms: float, success_count: int):
        """Log batch inference metrics"""
        self.logger.info(
            f"Batch inference: size={batch_size}, "
            f"total_time={total_time_ms:.2f}ms, "
            f"avg_time={total_time_ms/batch_size:.2f}ms, "
            f"success_rate={success_count/batch_size*100:.1f}%"
        )


class PerformanceMonitor:
    """Monitor and track system performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.request_counts = deque(maxlen=window_size)
        self.error_counts = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._start_time = time.time()
        
    def record_inference(self, inference_time_ms: float, success: bool = True):
        """Record an inference event"""
        with self._lock:
            self.inference_times.append(inference_time_ms)
            self.request_counts.append(1)
            self.error_counts.append(0 if success else 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self._lock:
            if not self.inference_times:
                return {
                    "total_requests": 0,
                    "avg_inference_time_ms": 0,
                    "p50_inference_time_ms": 0,
                    "p95_inference_time_ms": 0,
                    "p99_inference_time_ms": 0,
                    "error_rate": 0,
                    "uptime_seconds": time.time() - self._start_time
                }
            
            times = list(self.inference_times)
            errors = list(self.error_counts)
            
            return {
                "total_requests": len(times),
                "avg_inference_time_ms": round(np.mean(times), 2),
                "p50_inference_time_ms": round(np.percentile(times, 50), 2),
                "p95_inference_time_ms": round(np.percentile(times, 95), 2),
                "p99_inference_time_ms": round(np.percentile(times, 99), 2),
                "min_inference_time_ms": round(min(times), 2),
                "max_inference_time_ms": round(max(times), 2),
                "error_rate": round(sum(errors) / len(errors) * 100, 2),
                "uptime_seconds": round(time.time() - self._start_time, 2)
            }
    
    def is_healthy(self, threshold_ms: float = 100) -> bool:
        """Check if system is healthy based on inference time threshold"""
        stats = self.get_stats()
        return stats["p95_inference_time_ms"] < threshold_ms


def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        
        logger = logging.getLogger("timing")
        logger.debug(f"{func.__name__} executed in {execution_time:.2f}ms")
        
        return result
    return wrapper


def async_timing_decorator(func):
    """Decorator to measure async function execution time"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        
        logger = logging.getLogger("timing")
        logger.debug(f"{func.__name__} executed in {execution_time:.2f}ms")
        
        return result
    return wrapper


class ModelVersionTracker:
    """Track model versions and deployment history"""
    
    def __init__(self, tracking_file: str = "model_versions.json"):
        self.tracking_file = LOG_DIR / tracking_file
        self._load_history()
    
    def _load_history(self):
        """Load version history from file"""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {"versions": [], "current": None}
    
    def _save_history(self):
        """Save version history to file"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def register_version(self, 
                         version: str, 
                         model_path: str,
                         metrics: Dict[str, float],
                         description: str = ""):
        """Register a new model version"""
        version_entry = {
            "version": version,
            "model_path": model_path,
            "metrics": metrics,
            "description": description,
            "registered_at": datetime.utcnow().isoformat(),
            "is_active": False
        }
        
        self.history["versions"].append(version_entry)
        self._save_history()
        
        logger = setup_logger("model_version")
        logger.info(f"Registered model version: {version}")
    
    def set_active_version(self, version: str):
        """Set the active model version"""
        for v in self.history["versions"]:
            v["is_active"] = (v["version"] == version)
        
        self.history["current"] = version
        self._save_history()
        
        logger = setup_logger("model_version")
        logger.info(f"Activated model version: {version}")
    
    def get_active_version(self) -> Optional[Dict]:
        """Get the currently active model version"""
        for v in self.history["versions"]:
            if v["is_active"]:
                return v
        return None
    
    def get_all_versions(self) -> list:
        """Get all registered versions"""
        return self.history["versions"]


# Global instances
inference_logger = InferenceLogger()
performance_monitor = PerformanceMonitor()
model_version_tracker = ModelVersionTracker()
