"""
Task Manager for Incremental Search
Handles task state persistence and recovery
Now supports async background search with user decision checkpoints
"""
import json
import time
import uuid
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from enum import Enum

from . import schemas
from . import config


# ============================ TASK STATUS ENUM ============================

class TaskStatus(str, Enum):
    """Task execution status"""
    RUNNING = "running"  # 正在执行搜索
    WAITING_USER = "waiting_user"  # 等待用户决策（continue/finish）
    COMPLETED = "completed"  # 搜索已完成
    ERROR = "error"  # 发生错误
    CANCELLED = "cancelled"  # 用户取消


# Task storage directory
TASK_DIR = Path(config.DATA_DIR) / "search_tasks"
TASK_DIR.mkdir(parents=True, exist_ok=True)

# Task expiration time (24 hours)
TASK_EXPIRATION_HOURS = 24


def generate_task_id() -> str:
    """Generate a unique task ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"task_{timestamp}_{unique_id}"


def _get_task_path(task_id: str) -> Path:
    """Get the file path for a task"""
    return TASK_DIR / f"{task_id}.json"


def save_task_state(state: schemas.SearchTaskState) -> bool:
    """
    Save task state to disk using JSON
    
    Args:
        state: SearchTaskState to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Update timestamp
        state.updated_at = time.time()
        
        # ✅ Debug: print state before saving
        print(f"[TaskManager] Preparing to save state:")
        print(f"[TaskManager]   - candidates_accum keys: {list(state.candidates_accum.keys())}")
        print(f"[TaskManager]   - enhanced_profiles_accum keys: {list(getattr(state, 'enhanced_profiles_accum', {}).keys())}")
        print(f"[TaskManager]   - all_scored_papers count: {len(state.all_scored_papers)}")
        print(f"[TaskManager]   - search_candidate_set count: {len(state.search_candidate_set)}")
        
        # Convert to dict for JSON serialization
        state_dict = {
            "task_id": state.task_id,
            "spec": state.spec.model_dump(),
            "pos": state.pos,
            "terms": state.terms,
            "rounds_completed": state.rounds_completed,
            "candidates_accum": {
                name: candidate.model_dump(by_alias=True) 
                for name, candidate in state.candidates_accum.items()
            },
            "enhanced_profiles_accum": {
                name: profile.model_dump() 
                for name, profile in getattr(state, 'enhanced_profiles_accum', {}).items()
            },
            "all_serp": state.all_serp,
            "sources": state.sources,
            "all_scored_papers": {
                url: paper.model_dump() 
                for url, paper in state.all_scored_papers.items()
            },
            "search_candidate_set": state.search_candidate_set,
            "selected_urls_set": list(state.selected_urls_set),
            "selected_serp_url_set": list(state.selected_serp_url_set),
            "accumulated_logs": getattr(state, 'accumulated_logs', ''),
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }
        
        print(f"[TaskManager] State dict created, candidates_accum keys: {list(state_dict['candidates_accum'].keys())}")
        
        # Save to JSON file
        task_path = _get_task_path(state.task_id).with_suffix('.json')
        with open(task_path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)
        
        print(f"[TaskManager] Saved task state: {state.task_id}")
        print(f"[TaskManager]   - Rounds completed: {state.rounds_completed}")
        print(f"[TaskManager]   - Candidates found: {len(state.candidates_accum)}")
        print(f"[TaskManager]   - Search position: {state.pos}/{len(state.terms)}")
        
        return True
    except Exception as e:
        print(f"[TaskManager] Error saving task state: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_task_state(task_id: str) -> Optional[schemas.SearchTaskState]:
    """
    Load task state from disk
    
    Args:
        task_id: Task ID to load
        
    Returns:
        SearchTaskState if found and valid, None otherwise
    """
    try:
        # Try JSON file first
        task_path = _get_task_path(task_id).with_suffix('.json')
        
        if not task_path.exists():
            print(f"[TaskManager] Task not found: {task_id}")
            return None
        
        # Check if task is expired
        file_mtime = task_path.stat().st_mtime
        if time.time() - file_mtime > TASK_EXPIRATION_HOURS * 3600:
            print(f"[TaskManager] Task expired: {task_id}")
            # Clean up expired task
            task_path.unlink(missing_ok=True)
            return None
        
        # Load from JSON
        with open(task_path, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
        
        # ✅ Debug: print loaded data
        print(f"[TaskManager] Loaded JSON, reconstructing objects...")
        print(f"[TaskManager]   - candidates_accum keys in JSON: {list(state_dict['candidates_accum'].keys())}")
        print(f"[TaskManager]   - enhanced_profiles_accum keys in JSON: {list(state_dict.get('enhanced_profiles_accum', {}).keys())}")
        print(f"[TaskManager]   - all_scored_papers count in JSON: {len(state_dict['all_scored_papers'])}")
        
        # Reconstruct objects from dicts using model_validate for proper Pydantic v2 handling
        # This handles nested models correctly (e.g., IntroductionInfo inside EnhancedAuthorProfile)
        def safe_validate_candidate(cand_dict):
            """Safely validate CandidateOverview from dict"""
            try:
                return schemas.CandidateOverview.model_validate(cand_dict)
            except Exception as e:
                print(f"[TaskManager] Warning: Failed to validate CandidateOverview: {e}")
                # Fallback to **dict approach
                return schemas.CandidateOverview(**cand_dict)
        
        def safe_validate_profile(profile_dict):
            """Safely validate EnhancedAuthorProfile from dict"""
            try:
                # If already an EnhancedAuthorProfile instance, return it directly
                if isinstance(profile_dict, schemas.EnhancedAuthorProfile):
                    return profile_dict
                return schemas.EnhancedAuthorProfile.model_validate(profile_dict)
            except Exception as e:
                print(f"[TaskManager] Warning: Failed to validate EnhancedAuthorProfile: {e}")
                # Try with empty defaults for missing nested models
                try:
                    return schemas.EnhancedAuthorProfile(**profile_dict)
                except:
                    # Return empty profile as last resort
                    return schemas.EnhancedAuthorProfile()
        
        def safe_validate_paper(paper_dict):
            """Safely validate PaperWithScore from dict"""
            try:
                return schemas.PaperWithScore.model_validate(paper_dict)
            except Exception as e:
                print(f"[TaskManager] Warning: Failed to validate PaperWithScore: {e}")
                return schemas.PaperWithScore(**paper_dict)
        
        state = schemas.SearchTaskState(
            task_id=state_dict["task_id"],
            spec=schemas.QuerySpec.model_validate(state_dict["spec"]),
            pos=state_dict["pos"],
            terms=state_dict["terms"],
            rounds_completed=state_dict["rounds_completed"],
            candidates_accum={
                name: safe_validate_candidate(cand_dict)
                for name, cand_dict in state_dict["candidates_accum"].items()
            },
            enhanced_profiles_accum={
                name: safe_validate_profile(profile_dict)
                for name, profile_dict in state_dict.get("enhanced_profiles_accum", {}).items()
            },
            all_serp=state_dict["all_serp"],
            sources=state_dict["sources"],
            all_scored_papers={
                url: safe_validate_paper(paper_dict)
                for url, paper_dict in state_dict["all_scored_papers"].items()
            },
            search_candidate_set=state_dict["search_candidate_set"],
            selected_urls_set=set(state_dict["selected_urls_set"]),
            selected_serp_url_set=set(state_dict["selected_serp_url_set"]),
            accumulated_logs=state_dict.get("accumulated_logs", ""),
            created_at=state_dict["created_at"],
            updated_at=state_dict["updated_at"],
        )
        
        print(f"[TaskManager] State object created, candidates_accum: {len(state.candidates_accum)}")
        
        print(f"[TaskManager] Loaded task state: {task_id}")
        print(f"[TaskManager]   - Rounds completed: {state.rounds_completed}")
        print(f"[TaskManager]   - Candidates found: {len(state.candidates_accum)}")
        print(f"[TaskManager]   - Search position: {state.pos}/{len(state.terms)}")
        
        return state
    except Exception as e:
        print(f"[TaskManager] Error loading task state: {e}")
        import traceback
        traceback.print_exc()
        return None


def delete_task_state(task_id: str) -> bool:
    """
    Delete task state from disk
    
    Args:
        task_id: Task ID to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        task_path = _get_task_path(task_id)
        if task_path.exists():
            task_path.unlink()
            print(f"[TaskManager] Deleted task: {task_id}")
            return True
        return False
    except Exception as e:
        print(f"[TaskManager] Error deleting task: {e}")
        return False


def cleanup_expired_tasks() -> int:
    """
    Clean up expired task files
    
    Returns:
        Number of tasks cleaned up
    """
    try:
        count = 0
        current_time = time.time()
        expiration_threshold = TASK_EXPIRATION_HOURS * 3600
        
        for task_file in TASK_DIR.glob("task_*.json"):
            file_mtime = task_file.stat().st_mtime
            if current_time - file_mtime > expiration_threshold:
                task_file.unlink()
                count += 1
                print(f"[TaskManager] Cleaned up expired task: {task_file.stem}")
        
        if count > 0:
            print(f"[TaskManager] Cleaned up {count} expired task(s)")
        
        return count
    except Exception as e:
        print(f"[TaskManager] Error during cleanup: {e}")
        return 0


def list_active_tasks() -> list:
    """
    List all active (non-expired) tasks
    
    Returns:
        List of task IDs
    """
    try:
        tasks = []
        current_time = time.time()
        expiration_threshold = TASK_EXPIRATION_HOURS * 3600
        
        for task_file in TASK_DIR.glob("task_*.json"):
            file_mtime = task_file.stat().st_mtime
            if current_time - file_mtime <= expiration_threshold:
                tasks.append(task_file.stem)
        
        return tasks
    except Exception as e:
        print(f"[TaskManager] Error listing tasks: {e}")
        return []

def create_task_state_from_spec(
    spec: schemas.QuerySpec,
    terms: list,
) -> schemas.SearchTaskState:
    """
    Create a new task state from a query specification
    Args:
        spec: Query specification
        terms: Search terms to use
    Returns:
        New SearchTaskState
    """
    task_id = generate_task_id()
    
    state = schemas.SearchTaskState(
        task_id=task_id,
        spec=spec,
        pos=0,
        terms=terms,
        rounds_completed=0,
        candidates_accum={},
        enhanced_profiles_accum={},
        all_serp=[],
        sources={},
        all_scored_papers={},
        search_candidate_set=[],
        selected_urls_set=set(),
        selected_serp_url_set=set(),
    )
    return state
# ============================ ASYNC SEARCH MANAGER ============================

class AsyncSearchManager:
    """
    Manages background search tasks with user decision checkpoints
    
    每轮搜索完成后：
    1. 暂停搜索，等待用户决策
    2. 用户可以查看当前累积结果
    3. 用户选择 Continue（后台继续）或 Finish（停止搜索）
    """
    
    def __init__(self):
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def start_background_search(
        self,
        task_id: str,
        state: schemas.SearchTaskState,
        api_key: str,
        search_function: Callable,
        max_rounds: int = 10,
        progress_callback: Optional[Callable] = None
    ):
        """
        启动后台异步搜索
        
        Args:
            task_id: 任务ID
            state: 搜索状态
            api_key: API密钥
            search_function: 搜索函数（通常是 agent_execute_search）
            max_rounds: 最大搜索轮数
            progress_callback: 进度回调函数
        """
        def run_search():
            try:
                current_state = state
                
                while current_state.rounds_completed < max_rounds:
                    # 更新任务状态为运行中
                    with self._lock:
                        if task_id not in self._active_tasks:
                            break  # 任务已被取消
                        self._active_tasks[task_id]["status"] = TaskStatus.RUNNING
                        self._active_tasks[task_id]["current_round"] = current_state.rounds_completed + 1
                    
                    print(f"\n{'='*80}")
                    print(f"[AsyncSearch] Starting Round {current_state.rounds_completed + 1}")
                    print(f"{'='*80}\n")
                    
                    # 执行一轮搜索
                    result = search_function(
                        spec=current_state.spec,
                        api_key=api_key,
                        max_rounds_per_run=1,  # 每次只运行一轮
                        resume_state=current_state,
                        progress_callback=progress_callback
                    )
                    
                    # 检查是否为部分结果
                    is_partial = hasattr(result, 'is_partial') and result.is_partial
                    
                    # 更新状态
                    if is_partial:
                        # 获取最新状态用于下一轮
                        current_state = result.task_state
                        
                        # 保存状态到磁盘
                        save_task_state(current_state)
                        
                        # 更新内存中的任务信息
                        with self._lock:
                            if task_id not in self._active_tasks:
                                break  # 任务已被取消
                            
                            self._active_tasks[task_id].update({
                                "status": TaskStatus.WAITING_USER,
                                "rounds_completed": current_state.rounds_completed,
                                "candidates": dict(current_state.candidates_accum),
                                "enhanced_profiles": dict(getattr(current_state, 'enhanced_profiles_accum', {})),
                                "candidate_count": len(current_state.candidates_accum),
                                "last_update": time.time(),
                                "task_state": current_state,
                                "user_decision": None  # 重置用户决策
                            })
                        
                        print(f"\n{'='*80}")
                        print(f"[AsyncSearch] Round {current_state.rounds_completed} completed")
                        print(f"[AsyncSearch] Total candidates: {len(current_state.candidates_accum)}")
                        print(f"[AsyncSearch] Waiting for user decision...")
                        print(f"{'='*80}\n")
                        
                        # 等待用户决策
                        while True:
                            time.sleep(1)
                            
                            with self._lock:
                                if task_id not in self._active_tasks:
                                    return  # 任务已被取消
                                
                                decision = self._active_tasks[task_id].get("user_decision")
                                
                                if decision == "finish":
                                    print(f"[AsyncSearch] User chose to finish search")
                                    self._active_tasks[task_id]["status"] = TaskStatus.COMPLETED
                                    return
                                elif decision == "continue":
                                    print(f"[AsyncSearch] User chose to continue search")
                                    break  # 继续下一轮
                                elif decision == "cancel":
                                    print(f"[AsyncSearch] User cancelled search")
                                    self._active_tasks[task_id]["status"] = TaskStatus.CANCELLED
                                    return
                    else:
                        # 搜索自然完成（所有轮次已完成）
                        with self._lock:
                            if task_id in self._active_tasks:
                                self._active_tasks[task_id]["status"] = TaskStatus.COMPLETED
                                self._active_tasks[task_id]["candidates"] = dict(result.candidates)
                                self._active_tasks[task_id]["enhanced_profiles"] = dict(getattr(result, 'enhanced_profiles', {}))
                        
                        print(f"\n[AsyncSearch] Search completed naturally (all rounds finished)")
                        break
                
                # 达到最大轮数
                with self._lock:
                    if task_id in self._active_tasks:
                        self._active_tasks[task_id]["status"] = TaskStatus.COMPLETED
                
                print(f"\n[AsyncSearch] Reached maximum rounds ({max_rounds})")
                
            except Exception as e:
                print(f"\n[AsyncSearch] Error in background search: {e}")
                import traceback
                traceback.print_exc()
                
                with self._lock:
                    if task_id in self._active_tasks:
                        self._active_tasks[task_id]["status"] = TaskStatus.ERROR
                        self._active_tasks[task_id]["error_message"] = str(e)
        
        # 创建并启动后台线程
        thread = threading.Thread(target=run_search, daemon=True)
        
        with self._lock:
            self._active_tasks[task_id] = {
                "status": TaskStatus.RUNNING,
                "current_round": 0,
                "rounds_completed": 0,
                "max_rounds": max_rounds,
                "candidates": {},
                "enhanced_profiles": {},
                "candidate_count": 0,
                "task_state": state,
                "thread": thread,
                "user_decision": None,
                "created_at": time.time(),
                "last_update": time.time(),
                "error_message": None
            }
        
        thread.start()
        print(f"[AsyncSearch] Background search started for task {task_id}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Returns:
            {
                "status": TaskStatus,
                "current_round": int,
                "rounds_completed": int,
                "candidate_count": int,
                "needs_user_input": bool,
                "candidates": dict,
                "error_message": str | None
            }
        """
        with self._lock:
            if task_id not in self._active_tasks:
                return None
            
            task = self._active_tasks[task_id]
            return {
                "status": task["status"],
                "current_round": task.get("current_round", 0),
                "rounds_completed": task.get("rounds_completed", 0),
                "max_rounds": task.get("max_rounds", 10),
                "candidate_count": task.get("candidate_count", 0),
                "needs_user_input": task["status"] == TaskStatus.WAITING_USER,
                "candidates": task.get("candidates", {}),
                "enhanced_profiles": task.get("enhanced_profiles", {}),
                "error_message": task.get("error_message"),
                "last_update": task.get("last_update", 0)
            }
    
    def set_user_decision(self, task_id: str, decision: str) -> bool:
        """
        设置用户决策
        
        Args:
            task_id: 任务ID
            decision: "continue" | "finish" | "cancel"
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if task_id not in self._active_tasks:
                return False
            
            if decision not in ["continue", "finish", "cancel"]:
                return False
            
            self._active_tasks[task_id]["user_decision"] = decision
            self._active_tasks[task_id]["last_update"] = time.time()
            
            print(f"[AsyncSearch] User decision set for {task_id}: {decision}")
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self.set_user_decision(task_id, "cancel")
    
    def cleanup_task(self, task_id: str):
        """清理任务（从内存中移除）"""
        with self._lock:
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
                print(f"[AsyncSearch] Task {task_id} cleaned up from memory")
    
    def list_active_tasks(self) -> list:
        """列出所有活跃任务"""
        with self._lock:
            return list(self._active_tasks.keys())


# Global instance
_async_search_manager = None

def get_async_search_manager() -> AsyncSearchManager:
    """获取全局异步搜索管理器实例"""
    global _async_search_manager
    if _async_search_manager is None:
        _async_search_manager = AsyncSearchManager()
    return _async_search_manager
