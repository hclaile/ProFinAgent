'trainingMCPservice - AlphaQCM type: rl autogeneratetime: 2024'

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import uvicorn
import uuid
import logging
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import threading
from collections import deque

# config log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_mcp. log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Training MCP - AlphaQCM",
    description="trainingservice - AlphaQCM (Quantile-based Distributional RL for Alpha Mining)",
    version="1.0.0"
)

# ========== config ==========

REPO_PATH = Path(r"./workspace/AlphaQCM").resolve()
TRAIN_SCRIPT = 'train_qcm.py'
MAX_LOG_LINES = 1000  # maximumkeeplog

# task (use Redis)
tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()


# ========== Pydantic model ==========

class TrainRequest(BaseModel):
    """trainingrequestmodel"""
    
    model: str = Field(
        default="qrdqn",
        description="model type: qrdqn, iqn, fqf"
    )
    seed: int = Field(
        default=0,
        description="text",
        ge=0
    )
    pool: int = Field(
        default=20,
        description="Alpha pool",
        ge=1
    )
    std_lam: float = Field(
        default=1.0,
        description='standard deviation lambda parameter',
        gt=0
    )
    config_file: Optional[str] = Field(
        None,
        description='config file path(root directory)'
    )
    task_name: str = Field(
        default="training",
        description="task name",
        min_length=1,
        max_length=100
    )
    
    @validator('model')
    def validate_model(cls, v: str) -> str:
        """model type"""
        allowed_models = ['qrdqn', 'iqn', 'fqf']
        if v not in allowed_models:
            raise ValueError(f"model: {allowed_models}")
        return v
    
    @validator('pool')
    def validate_pool(cls, v: int) -> int:
        """size"""
        recommended_pools = [10, 20, 50, 100]
        if v not in recommended_pools:
            logger.warning(f"size {v} {recommended_pools}")
        return v
    
    @validator('std_lam')
    def validate_std_lam(cls, v: float) -> float:
        """std_lam parameter"""
        recommended_lams = [0.5, 1.0, 2.0]
        if v not in recommended_lams:
            logger.warning(f"std_lam {v} {recommended_lams}")
        return v


class TrainResponse(BaseModel):
    """trainingresponsemodel"""
    
    task_id: str = Field(..., description="task text")
    status: str = Field(..., description="task status: started, running, completed, failed")
    message: str = Field(..., description="status text")
    command: str = Field(..., description="training text")
    log_preview: str = Field(default="", description='log ()')
    task_name: str = Field(..., description="task name")
    created_at: str = Field(..., description="time")


class TaskStatus(BaseModel):
    """task statusmodel"""
    
    task_id: str
    status: str
    command: str
    task_name: str
    created_at: str
    pid: Optional[int] = None
    log_lines: int = 0
    last_log: str = ""
    error: Optional[str] = None


class LogResponse(BaseModel):
    """log response model"""
    
    task_id: str
    total_lines: int
    logs: List[str]
    status: str


# ========== tool ==========

def setup_environment() -> Dict[str, str]:
    'training environment variable Returns: environment variabledictionary'
    env = os.environ.copy()
    
    # path PYTHONPATH
    env["PYTHONPATH"] = str(REPO_PATH) + os.pathsep + env.get("PYTHONPATH", "")
    
    # Python output, getlog
    env["PYTHONUNBUFFERED"] = "1"
    
    # CUDArelatedenvironment variable()
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    
    logger.info(f"environment variable: PYTHONPATH={env['PYTHONPATH'][:100]}...")
    return env


def validate_repository() -> bool:
    'path training exists Returns: Raises: File Not Found Error: path file does not exist'
    if not REPO_PATH.exists():
        raise FileNotFoundError(f"path does not exist: {REPO_PATH}")
    
    train_script_path = REPO_PATH / TRAIN_SCRIPT
    if not train_script_path.exists():
        raise FileNotFoundError(f"training does not exist: {train_script_path}")
    
    logger.info(f": {REPO_PATH}")
    return True


def build_train_command(request: TrainRequest) -> List[str]:
    'request training Args: request: trainingrequest Returns: list'
    cmd = [
        sys.executable,
        str(REPO_PATH / TRAIN_SCRIPT),
        "--model", request.model,
        "--seed", str(request.seed),
        "--pool", str(request.pool),
        "--std-lam", str(request.std_lam)
    ]
    
    # config file
    # if request.config_file:
    # config_path = REPO_PATH/request.config_file
    # if not config_path. exists():
    # raise File Not Found Error(f"config file does not exist: {config_path}")
    # cmd. extend(["--config", str(config_path)])
    
    return cmd


def get_log_preview(logs: deque, lines: int = 10) -> str:
    'getlog Args: logs: log lines: Returns: log'
    if not logs:
        return ""
    
    preview_lines = list(logs)[-lines:]
    return ''.join(preview_lines)


# ========== task ==========

def _run_training(task_id: str, cmd: List[str]) -> None:
    'training task Args: task_id: task ID cmd: training list'
    try:
        env = setup_environment()
        
        logger.info(f"startup training [{task_id}]: {' '.join(cmd)}")
        
        # startup
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_PATH),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # updatetask status
        with tasks_lock:
            tasks[task_id]["pid"] = proc.pid
            tasks[task_id]["status"] = "running"
        
        logger.info(f"training startup [{task_id}], PID: {proc.pid}")
        
        # read output
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            
            line_stripped = line.rstrip()
            
            with tasks_lock:
                tasks[task_id]["log"].append(line_stripped)
                # logsize
                if len(tasks[task_id]["log"]) > MAX_LOG_LINES:
                    tasks[task_id]["log"].popleft()
            
            logger.info(f"[training {task_id[:8]}] {line_stripped}")
        
        # complete
        return_code = proc.wait()
        
        # updatefinalstatus
        with tasks_lock:
            if return_code == 0:
                tasks[task_id]["status"] = "completed"
                logger.info(f"training task complete [{task_id}]")
            else:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = f"training: {return_code}"
                logger.error(f"training task failed [{task_id}],: {return_code}")
    
    except Exception as e:
        error_msg = f"training taskexception: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}", exc_info=True)
        
        with tasks_lock:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = error_msg


# ========== module interface ==========

def run(args) -> None:
    'module interface, module call Args: args: Simple Namespace, attribute: - model: model type (qrdqn/iqn/fqf), default "qrdqn" - seed:, default 0 - pool: Alpha, default 20 - std_lam: standard deviation lambda parameter, default 1.0 - task_name: task name, default "training"'
    try:
        #
        validate_repository()
        
        # args Train Request
        request = TrainRequest(
            model=getattr(args, "model", "qrdqn"),
            seed=getattr(args, "seed", 0),
            pool=getattr(args, "pool", 20),
            std_lam=getattr(args, "std_lam", 1.0),
            task_name=getattr(args, "task_name", "training")
        )
        
        # training
        cmd = build_train_command(request)
        task_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # initialize task record
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "started",
                "command": ''.join(cmd),
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "created_at": created_at,
                "error": None
            }
        
        logger.info(f"start training task [{task_id}]: {' '.join(cmd)}")
        
        # training task
        thread = threading.Thread(target=_run_training, args=(task_id, cmd), daemon=False)
        thread.start()
        
        logger.info(f"training task startup [{task_id}], ID: {thread.ident}")
        print(f"training task startup, task ID: {task_id}")
        print(f"use: {' '.join(cmd)}")
        print(f"task status training service API query: GET/train/status/{task_id}")
        
    except Exception as e:
        error_msg = f"start training task failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


# ========== API endpoint ==========

@app.on_event("startup")
async def startup_event() -> None:
    """service startup initialization"""
    try:
        validate_repository()
        logger.info("trainingMCPservice startupsuccess")
    except Exception as e:
        logger.error(f"service startup failed: {e}", exc_info=True)
        raise


@app.get("/", tags=["basic"])
async def root() -> Dict[str, Any]:
    'service path, return service Returns: service'
    return {
        "service": "Training MCP - AlphaQCM",
        "algorithm": "AlphaQCM (Quantile-based Distributional RL for Alpha Mining)",
        "algorithm_type": "rl",
        "version": "1.0.0",
        "repository": str(REPO_PATH),
        "train_script": TRAIN_SCRIPT,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "train_start": 'POST/train/start',
            "train_status": 'GET/train/status/{task_id}',
            "train_logs": 'GET/train/logs/{task_id}',
            "train_list": 'GET/train/list'
        },
        "supported_models": ["qrdqn", "iqn", "fqf"],
        "recommended_pools": [10, 20, 50, 100],
        "recommended_std_lam": [0.5, 1.0, 2.0]
    }


@app.get("/health", tags=["basic"])
async def health_check() -> Dict[str, str]:
    'health check endpoint Returns: status'
    try:
        validate_repository()
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"service: {str(e)}")


@app.post("/train/start", response_model=TrainResponse, tags=["training"])
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
) -> TrainResponse:
    'start training task Args: request: trainingrequest parameter background_tasks: FastAPI task Returns: training task Raises: HTTPException: startup failed'
    task_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    try:
        #
        validate_repository()
        
        # training
        cmd = build_train_command(request)
        command_str = ''.join(cmd)
        
        logger.info(f"start training task [{task_id}]: {command_str}")
        
        # initialize task record
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "started",
                "command": command_str,
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "created_at": created_at,
                "error": None
            }
        
        # task
        background_tasks.add_task(_run_training, task_id, cmd)
        
        # getlog
        log_preview = f"training task, startup...\n: {command_str}"
        
        logger.info(f"training task [{task_id}]")
        
        return TrainResponse(
            task_id=task_id,
            status="started",
            message="training task successfully started",
            command=command_str,
            log_preview=log_preview,
            task_name=request.task_name,
            created_at=created_at
        )
    
    except FileNotFoundError as e:
        logger.error(f"file does not exist: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except ValueError as e:
        logger.error(f"parameter failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"startup training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"startup training failed: {str(e)}")


@app.get("/train/status/{task_id}", response_model=TaskStatus, tags=["training"])
async def get_task_status(task_id: str) -> TaskStatus:
    'gettraining task status Args: task_id: task ID Returns: task status Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        last_log = list(task["log"])[-1] if task["log"] else ""
        
        return TaskStatus(
            task_id=task_id,
            status=task["status"],
            command=task["command"],
            task_name=task["task_name"],
            created_at=task["created_at"],
            pid=task["pid"],
            log_lines=len(task["log"]),
            last_log=last_log,
            error=task.get("error")
        )


@app.get("/train/logs/{task_id}", response_model=LogResponse, tags=["training"])
async def get_task_logs(
    task_id: str,
    lines: int = 100,
    offset: int = 0
) -> LogResponse:
    'gettraining tasklog Args: task_id: task ID lines: return log offset: (start) Returns: log Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        all_logs = list(task["log"])
        total_lines = len(all_logs)
        
        # calculate log range
        start_idx = max(0, total_lines - offset - lines)
        end_idx = max(0, total_lines - offset)
        
        logs = all_logs[start_idx:end_idx]
        
        return LogResponse(
            task_id=task_id,
            total_lines=total_lines,
            logs=logs,
            status=task["status"]
        )


@app.get("/train/list", tags=["training"])
async def list_tasks() -> Dict[str, Any]:
    'training task Returns: task list'
    with tasks_lock:
        task_list = [
            {
                "task_id": task_id,
                "task_name": task["task_name"],
                "status": task["status"],
                "created_at": task["created_at"],
                "pid": task["pid"]
            }
            for task_id, task in tasks.items()
        ]
        
        return {
            "total": len(task_list),
            "tasks": task_list
        }


@app.delete("/train/{task_id}", tags=["training"])
async def delete_task(task_id: str) -> Dict[str, str]:
    'deletetraining task record() Args: task_id: task ID Returns: delete Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        if task["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail='delete task failed because the task is complete'
            )
        
        del tasks[task_id]
        logger.info(f"task delete: {task_id}")
        
        return {"message": f"task {task_id} delete"}


# ========== ==========

if __name__ == "__main__":
    'startup FastAPI service'
    logger.info("=" * 60)
    logger.info("startuptrainingMCPservice - AlphaQCM")
    logger.info(f"path: {REPO_PATH}")
    logger.info(f"training: {TRAIN_SCRIPT}")
    logger.info("=" * 60)
    
    try:
        # startup
        validate_repository()
        
        # startup service
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8010,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"service startup failed: {e}", exc_info=True)
        sys.exit(1)