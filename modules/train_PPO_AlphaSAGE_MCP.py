'trainingMCPservice - AlphaSAGE type: rl name: PPO-based Alpha Factor Generation (AlphaSAGE) autogeneratetime: 2024'

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
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

# ========== logconfig ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_mcp. log')
    ]
)
logger = logging.getLogger(__name__)

# ========== FastAPIapplication ==========
app = FastAPI(
    title="Training MCP - AlphaSAGE",
    description="PPO-based Alpha Factor Generation Training Service",
    version="1.0.0"
)

# ========== config ==========
REPO_PATH = Path(r"./workspace/AlphaSAGE").resolve()
TRAIN_SCRIPT = 'train_ppo.py'
MAX_LOG_LINES = 1000  # maximumlog

# task (use Redis)
tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()

# ========== Pydantic model ==========

class TrainRequest(BaseModel):
    """trainingrequestmodel"""
    config_file: Optional[str] = Field(None, description='config file path')
    config_params: Optional[Dict[str, Any]] = Field(
        None,
        description="trainingparameterdictionary",
        example={
            "seed": 0,
            "instruments": "csi300",
            "pool": 20,
            "steps": 200000
        }
    )
    task_name: str = Field("training", description="task name")

    class Config:
        schema_extra = {
            "example": {
                "config_params": {
                    "seed": 0,
                    "instruments": "csi300",
                    "pool": 20,
                    "steps": 200000
                },
                "task_name": "ppo_training_csi300"
            }
        }


class TrainResponse(BaseModel):
    """trainingresponsemodel"""
    task_id: str = Field(..., description="task text")
    status: str = Field(..., description="task status: started/running/completed/failed")
    message: str = Field(..., description="response text")
    command: str = Field(..., description="training text")
    log_preview: str = Field("", description='log ()')
    timestamp: str = Field(..., description="task time")


class TaskStatus(BaseModel):
    """task statusmodel"""
    task_id: str
    status: str
    command: str
    pid: Optional[int] = None
    log_lines: int = 0
    recent_logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ========== tool ==========

def setup_environment() -> dict:
    'trainingenvironment variable Returns: dict: environment variabledictionary'
    env = os.environ.copy()
    src_path = str(REPO_PATH / "src")
    pythonpath = str(REPO_PATH) + os.pathsep + src_path
    if "PYTHONPATH" in env:
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    logger.info(f"environment variable complete: PYTHONPATH={env['PYTHONPATH']}")
    env["PYTHONUNBUFFERED"] = "1"  # Python output
    return env



def build_train_command(
    config_file: Optional[str] = None,
    config_params: Optional[Dict[str, Any]] = None
) -> List[str]:
    'training Args: config_file: config file path config_params: config parameter dictionary Returns: List[str]: parameterlist Raises: Value Error: parameterinvalid'
    cmd = [sys.executable, str(REPO_PATH / TRAIN_SCRIPT)]
    
    # prefer usingconfig file
    if config_file:
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = REPO_PATH / config_path
        
        if not config_path.exists():
            logger.info(f"config file does not exist: {config_path}")
        else:
            cmd.extend(["--config", str(config_path)])
            logger.info(f"useconfig file: {config_path}")
    
    # parameter
    if config_params:
        for key, value in config_params.items():
            # parameter ()
            if not key.replace('_', '').isalnum():
                raise ValueError(f"invalid parameter: {key}")
            
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"config parameter: {config_params}")
    
    return cmd


def get_log_preview(logs: deque, lines: int = 10) -> str:
    'getlog Args: logs: log lines: Returns: str: log'
    if not logs:
        return ""
    
    recent = list(logs)[-lines:]
    return ''.join(recent)


# ========== module interface ==========

def run(args) -> None:
    'module interface, module call Args: args: Simple Namespace, attribute: - config_file: config file path (optional) - config_params: config parameter dictionary(optional),: - seed: - instruments: instrument pool, "csi300" - pool: Alpha - steps: training - task_name: task name, default "training"'
    try:
        # argsget parameter
        config_file = getattr(args, "config_file", None)
        config_params = getattr(args, "config_params", None)
        task_name = getattr(args, "task_name", "training")
        
        # training
        cmd = build_train_command(config_file=config_file, config_params=config_params)
        task_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # initialize task record
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": task_name,
                "status": "started",
                "command": ''.join(cmd),
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "created_at": created_at,
                "updated_at": created_at,
                "error": None
            }
        
        logger.info(f"start training task [{task_id}]: {' '.join(cmd)}")
        
        # training task
        thread = threading.Thread(target=_run_training_task, args=(task_id, cmd), daemon=False)
        thread.start()
        
        logger.info(f"training task startup [{task_id}], ID: {thread.ident}")
        print(f"training task startup, task ID: {task_id}")
        print(f"use: {' '.join(cmd)}")
        print(f"task status training service API query: GET/train/status/{task_id}")
        
    except Exception as e:
        error_msg = f"start training task failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


# ========== task ==========

def _run_training_task(task_id: str, cmd: List[str]) -> None:
    'training task Args: task_id: task ID cmd: parameterlist'
    try:
        env = setup_environment()
        
        # startup training
        logger.info(f"startup training [{task_id}]: {' '.join(cmd)}")
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
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"training startup [{task_id}], PID: {proc.pid}")
        
        # readoutput
        assert proc.stdout is not None
        for line in proc.stdout:
            line_stripped = line.rstrip()
            
            with tasks_lock:
                tasks[task_id]["log"].append(line_stripped)
                tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # record log file
            logger.info(f"[training {task_id[:8]}] {line_stripped}")
        
        #
        return_code = proc.wait()
        
        # updatefinalstatus
        with tasks_lock:
            if return_code == 0:
                tasks[task_id]["status"] = "completed"
                logger.info(f"training task complete [{task_id}]")
            else:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = f": {return_code}"
                logger.error(f"training task failed [{task_id}],: {return_code}")
            
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    except Exception as e:
        error_msg = f"training taskexception: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}", exc_info=True)
        
        with tasks_lock:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = error_msg
            tasks[task_id]["updated_at"] = datetime.now().isoformat()


# ========== API endpoint ==========

@app.get("/", tags=["basic"])
async def root():
    """service path"""
    return {
        "service": "Training MCP - AlphaSAGE",
        "algorithm": "PPO-based Alpha Factor Generation",
        "version": "1.0.0",
        "repository": str(REPO_PATH),
        "endpoints": {
            "docs": "/docs",
            "train_start": "/train/start",
            "train_status": "/train/status/{task_id}",
            "train_logs": "/train/logs/{task_id}",
            "train_list": "/train/list"
        }
    }


@app.get("/health", tags=["basic"])
async def health_check():
    """health check"""
    repo_exists = REPO_PATH.exists()
    script_exists = (REPO_PATH / TRAIN_SCRIPT).exists()
    
    return {
        "status": "healthy" if (repo_exists and script_exists) else "unhealthy",
        "repository_exists": repo_exists,
        "train_script_exists": script_exists,
        "active_tasks": len([t for t in tasks.values() if t["status"] == "running"])
    }


@app.post("/train/start", response_model=TrainResponse, tags=["training"])
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
) -> TrainResponse:
    'start training task configstartupPPOtraining. Args: request: trainingrequest parameter background_tasks: FastAPI task Returns: Train Response: trainingresponse, task ID status Raises: HTTPException: startup failed'
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    try:
        # path
        if not REPO_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"path does not exist: {REPO_PATH}"
            )
        
        # training
        train_script_path = REPO_PATH / TRAIN_SCRIPT
        if not train_script_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"training does not exist: {train_script_path}"
            )
        
        # training
        cmd = build_train_command(
            config_file=request.config_file,
            config_params=request.config_params
        )
        command_str = ''.join(cmd)
        
        logger.info(f"training task [{task_id}]: {request.task_name}")
        logger.info(f"training: {command_str}")
        
        # initialize task status
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "started",
                "command": command_str,
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "error": None,
                "created_at": timestamp,
                "updated_at": timestamp
            }
        
        # training task
        background_tasks.add_task(_run_training_task, task_id, cmd)
        
        return TrainResponse(
            task_id=task_id,
            status="started",
            message=f"training task startup: {request.task_name}",
            command=command_str,
            log_preview='training start...',
            timestamp=timestamp
        )
    
    except ValueError as e:
        logger.error(f"parameter failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"startup training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"startup training failed: {str(e)}")


@app.get("/train/status/{task_id}", response_model=TaskStatus, tags=["training"])
async def get_task_status(task_id: str) -> TaskStatus:
    'gettraining task status Args: task_id: task ID Returns: Task Status: task status Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        
        return TaskStatus(
            task_id=task["task_id"],
            status=task["status"],
            command=task["command"],
            pid=task.get("pid"),
            log_lines=len(task["log"]),
            recent_logs=list(task["log"])[-20:],  # 20
            error=task.get("error"),
            created_at=task["created_at"],
            updated_at=task["updated_at"]
        )


@app.get("/train/logs/{task_id}", tags=["training"])
async def get_task_logs(
    task_id: str,
    lines: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    'gettraining tasklog Args: task_id: task ID lines: return log offset: log (start) Returns: dict: log dictionary Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        all_logs = list(task["log"])
        total_lines = len(all_logs)
        
        # calculatelog
        start = max(0, total_lines - offset - lines)
        end = total_lines - offset
        
        return {
            "task_id": task_id,
            "total_lines": total_lines,
            "offset": offset,
            "lines": lines,
            "logs": all_logs[start:end]
        }


@app.get("/train/list", tags=["training"])
async def list_tasks() -> Dict[str, Any]:
    'training task Returns: dict: task list'
    with tasks_lock:
        task_list = []
        for task_id, task in tasks.items():
            task_list.append({
                "task_id": task_id,
                "task_name": task.get("task_name", "unknown"),
                "status": task["status"],
                "pid": task.get("pid"),
                "created_at": task["created_at"],
                "updated_at": task["updated_at"]
            })
        
        return {
            "total": len(task_list),
            "tasks": sorted(task_list, key=lambda x: x["created_at"], reverse=True)
        }


@app.delete("/train/{task_id}", tags=["training"])
async def delete_task(task_id: str) -> Dict[str, str]:
    'deletetraining task record: Args: task_id: task ID Returns: dict: delete result Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        if task["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail='delete task, task'
            )
        
        del tasks[task_id]
        logger.info(f"deletetask record: {task_id}")
        
        return {"message": f"task delete: {task_id}"}


# ========== ==========

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info('startuptrainingMCPservice - PPO- AlphaSAGE')
    logger.info(f"path: {REPO_PATH}")
    logger.info(f"training: {TRAIN_SCRIPT}")
    logger.info(f"Python: {sys.version}")
    logger.info("=" * 60)
    
    #
    if not REPO_PATH.exists():
        logger.error(f"error: path does not exist: {REPO_PATH}")
        sys.exit(1)
    
    if not (REPO_PATH / TRAIN_SCRIPT).exists():
        logger.error(f"error: training does not exist: {REPO_PATH / TRAIN_SCRIPT}")
        sys.exit(1)
    
    # startup service
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info",
        access_log=True
    )