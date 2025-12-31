"""
Dynamic Concurrency Manager for Talent Search System
Intelligently adjusts thread pool sizes based on system resources and task characteristics
"""

import os
import psutil
import threading
from typing import Optional, Literal
from enum import Enum

class TaskType(Enum):
    """Task types with different concurrency characteristics"""
    IO_BOUND = "io_bound"          # Network requests, file I/O
    CPU_BOUND = "cpu_bound"         # LLM processing, parsing
    MIXED = "mixed"                 # Both IO and CPU
    LIGHTWEIGHT = "lightweight"     # Quick operations

class DynamicConcurrencyManager:
    """
    Manages optimal concurrency levels based on system resources and task type
    """
    
    def __init__(self):
        # System info
        self.cpu_count = os.cpu_count() or 4
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Track current usage
        self._lock = threading.Lock()
        self._active_workers = {}  # task_type -> count
        
        # ⚖️ 平衡并发策略：适度降低整体并发，避免内存压力
        self.min_workers = 3  # 适度并发（原来4）
        self.max_workers_io = min(100, self.cpu_count * 12)  # IO-bound 适度降低（原来cpu*20）
        self.max_workers_cpu = self.cpu_count * 3  # CPU-bound 保持（LLM处理，API限速更重要）
        self.max_workers_mixed = self.cpu_count * 6  # 混合任务适度降低（原来*8）- 候选人处理
        self.max_workers_llm_api = min(15, self.cpu_count * 2) 
        
        print(f"[DynamicConcurrency] Initialized: {self.cpu_count} CPUs, {self.total_memory_gb:.1f}GB RAM")
    
    def get_optimal_workers(
        self,
        task_count: int,
        task_type: TaskType = TaskType.MIXED,
        prefer_speed: bool = True,
        memory_per_task_mb: int = 500
    ) -> int:
        """
        Calculate optimal number of workers for given task
        
        Args:
            task_count: Number of tasks to process
            task_type: Type of task (IO_BOUND, CPU_BOUND, MIXED, LIGHTWEIGHT)
            prefer_speed: If True, prioritize speed over resource conservation
            memory_per_task_mb: Estimated memory per task in MB
            
        Returns:
            Optimal number of workers
        """
        
        # 1. Get current system state
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # 2. ⚖️ 平衡基础计算：适度并发，避免内存压力
        if task_type == TaskType.IO_BOUND:
            # IO-bound: 网络请求、爬取等
            base_workers = min(task_count, self.cpu_count * 10)  # 适度降低：15 → 10
            max_allowed = self.max_workers_io
            
        elif task_type == TaskType.CPU_BOUND:
            # CPU-bound: LLM处理等，受API限速影响更大
            base_workers = min(task_count, self.cpu_count * 2)  # 保持：2
            max_allowed = self.max_workers_cpu
            
        elif task_type == TaskType.LIGHTWEIGHT:
            # Lightweight: 轻量级任务
            base_workers = min(task_count, self.cpu_count * 10)  # 适度降低：12 → 10
            max_allowed = self.max_workers_io
            
        else:  # MIXED
            # Mixed: 候选人处理等混合任务（易出现内存问题）
            base_workers = min(task_count, self.cpu_count * 4)  # 适度降低：6 → 4
            max_allowed = self.max_workers_mixed
        
        # 3. ⚖️ 平衡调整策略：适度并发，避免过载
        if cpu_percent > 90:
            # 极高CPU: 适度降低
            adjustment_factor = 0.6 if task_type == TaskType.CPU_BOUND else 0.7
            
        elif cpu_percent > 70:
            # 高CPU: 轻微降低
            adjustment_factor = 0.85
            
        elif cpu_percent < 40:
            # 低CPU: 适度提升
            adjustment_factor = 1.6 if prefer_speed else 1.3
            
        else:
            # 正常范围: 保持或轻微提升
            adjustment_factor = 1.2 if prefer_speed else 1.0
        
        # 4. Memory constraint check
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        memory_constrained_workers = int(
            (available_memory_gb * 1024 * 0.8) / memory_per_task_mb
        )
        
        # 5. Calculate final worker count
        adjusted_workers = int(base_workers * adjustment_factor)
        
        # Apply all constraints
        optimal = max(
            self.min_workers,
            min(
                adjusted_workers,
                memory_constrained_workers,
                max_allowed,
                task_count  # Never more workers than tasks
            )
        )
        
        # 6. Log decision
        print(f"[DynamicConcurrency] Optimal workers: {optimal}")
        print(f"  Task: {task_count} {task_type.value} tasks")
        print(f"  System: CPU={cpu_percent:.1f}%, MEM={memory_percent:.1f}%")
        print(f"  Decision: base={base_workers}, adjusted={adjusted_workers}, final={optimal}")
        
        return optimal
    
    def get_candidate_processing_workers(
        self, 
        candidate_count: int,
        required_count: int,
        has_homepage: bool = True
    ) -> int:
        """
        ✅ 激进并发策略：尽可能处理所有候选人
        
        Args:
            candidate_count: Total number of candidates to process
            required_count: Number of candidates actually needed
            has_homepage: Whether candidates have homepages (more IO-intensive)
            
        Returns:
            Optimal worker count for candidate processing
        """
        
        # Candidate processing is MIXED: IO (web crawling) + CPU (LLM)
        # If processing homepages, more IO-bound
        task_type = TaskType.IO_BOUND if has_homepage else TaskType.MIXED
        
        # ✅ 更实际的内存估计（之前1GB太保守了）
        memory_per_task = 600 if has_homepage else 300
        
        # ✅ 激进策略：尽可能处理所有候选人，只受系统资源限制
        # 不再限制为 required * 2，而是处理全部候选人
        smart_count = candidate_count
        
        print(f"[DynamicConcurrency] Candidate processing: {candidate_count} total, {required_count} required")
        print(f"[DynamicConcurrency] Strategy: Process ALL candidates (激进模式)")
        
        return self.get_optimal_workers(
            task_count=smart_count,
            task_type=task_type,
            prefer_speed=True,  # Users are waiting
            memory_per_task_mb=memory_per_task
        )
    
    def get_extraction_workers(self, url_count: int) -> int:
        """
        Optimal workers for URL content extraction
        
        Args:
            url_count: Number of URLs to fetch
            
        Returns:
            Optimal worker count
        """
        # URL fetching is IO-bound
        return self.get_optimal_workers(
            task_count=url_count,
            task_type=TaskType.IO_BOUND,
            prefer_speed=True,
            memory_per_task_mb=50  # URL fetching uses little memory
        )
    
    def get_llm_processing_workers(self, task_count: int) -> int:
        """
        🚀 高并发策略：LLM API调用（纯网络IO，可以更高）
        
        Args:
            task_count: Number of LLM calls to make
            
        Returns:
            Optimal worker count
        """
        # 🚀 LLM API调用是纯网络IO，不消耗CPU，可以开很高的并发
        # 主要受API rate limit限制，内存使用很低
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # 基础并发数更高（专门为LLM API优化）
        base_workers = min(task_count, self.cpu_count * 12)  # 🚀 提升：原来通过IO_BOUND只有10
        max_allowed = self.max_workers_llm_api  # 使用专门的LLM API上限
        
        # 更激进的调整策略（LLM API不消耗资源）
        if cpu_percent > 90:
            adjustment_factor = 0.8  # 即使CPU高，也只轻微降低
        elif cpu_percent < 50:
            adjustment_factor = 2.0  # CPU空闲时大幅提升
        else:
            adjustment_factor = 1.5  # 正常情况也提升
        
        # 内存约束检查（LLM API内存使用很低）
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        memory_constrained_workers = int((available_memory_gb * 1024 * 0.8) / 100)  # 只需100MB/task
        
        adjusted_workers = int(base_workers * adjustment_factor)
        optimal = max(
            self.min_workers,
            min(adjusted_workers, memory_constrained_workers, max_allowed, task_count)
        )
        
        print(f"[DynamicConcurrency] LLM API workers: {optimal}")
        print(f"  Task: {task_count} LLM API calls")
        print(f"  System: CPU={cpu_percent:.1f}%, MEM={memory_percent:.1f}%")
        print(f"  Decision: base={base_workers}, adjusted={adjusted_workers}, final={optimal}")
        
        return optimal
    
    def register_active_task(self, task_type: TaskType, count: int):
        """Register active workers (for monitoring)"""
        with self._lock:
            self._active_workers[task_type] = count
    
    def unregister_active_task(self, task_type: TaskType):
        """Unregister completed task"""
        with self._lock:
            self._active_workers.pop(task_type, None)
    
    def get_system_status(self) -> dict:
        """Get current system status"""
        return {
            "cpu_count": self.cpu_count,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "active_workers": dict(self._active_workers)
        }
    
    def should_throttle(self) -> tuple[bool, str]:
        """
        Check if system should throttle due to resource constraints
        
        Returns:
            (should_throttle, reason)
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Critical: Memory above 90%
        if memory.percent > 90:
            return True, f"Critical memory usage: {memory.percent:.1f}%"
        
        # Critical: CPU above 95%
        if cpu_percent > 95:
            return True, f"Critical CPU usage: {cpu_percent:.1f}%"
        
        # Warning: Memory above 85%
        if memory.percent > 85:
            return True, f"High memory usage: {memory.percent:.1f}%"
        
        # No throttling needed
        return False, ""

# Global instance
_manager = None

def get_manager() -> DynamicConcurrencyManager:
    """Get or create the global concurrency manager"""
    global _manager
    if _manager is None:
        _manager = DynamicConcurrencyManager()
    return _manager

# Convenience functions
def get_optimal_workers(
    task_count: int,
    task_type: str = "mixed",
    **kwargs
) -> int:
    """
    Quick helper to get optimal worker count
    
    Args:
        task_count: Number of tasks
        task_type: One of 'io_bound', 'cpu_bound', 'mixed', 'lightweight'
        **kwargs: Additional parameters for get_optimal_workers
        
    Returns:
        Optimal number of workers
    """
    type_map = {
        'io_bound': TaskType.IO_BOUND,
        'io': TaskType.IO_BOUND,
        'cpu_bound': TaskType.CPU_BOUND,
        'cpu': TaskType.CPU_BOUND,
        'mixed': TaskType.MIXED,
        'lightweight': TaskType.LIGHTWEIGHT,
        'light': TaskType.LIGHTWEIGHT
    }
    
    task_type_enum = type_map.get(task_type.lower(), TaskType.MIXED)
    manager = get_manager()
    
    return manager.get_optimal_workers(task_count, task_type_enum, **kwargs)

# Specific helpers for common tasks
def get_candidate_workers(candidates: int, required: int) -> int:
    """Get optimal workers for candidate processing"""
    manager = get_manager()
    return manager.get_candidate_processing_workers(candidates, required)

def get_extraction_workers(urls: int) -> int:
    """Get optimal workers for URL extraction"""
    manager = get_manager()
    return manager.get_extraction_workers(urls)

def get_llm_workers(tasks: int) -> int:
    """Get optimal workers for LLM processing"""
    manager = get_manager()
    return manager.get_llm_processing_workers(tasks)
