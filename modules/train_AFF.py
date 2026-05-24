'trainingMCPservice - Alpha Forge type: dl name: Alpha Forge - GAN-based Alpha Factor Generation autogeneratetime: 2024'

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, List, Union
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Training MCP - AlphaForge',
    description='Alpha Forge training service - GAN-based Alpha Factor Generation',
    version="1.0.0"
)

# ========== config ==========
REPO_PATH = Path(r'./workspace/AlphaForge').resolve()
TRAIN_SCRIPT = 'train_AFF.py'
MAX_LOG_LINES = 1000  # maximumlog

# ========== task ==========
tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()

# ========== Pydantic model ==========

class TrainRequest(BaseModel):
    """trainingrequestmodel"""
    config_file: Optional[str] = Field(None, description='config file path')
    config_params: Optional[Dict[str, Any]] = Field(None, description="configparameter")
    task_name: str = Field("training", description="task name")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config_params": {
                    "instruments": "csi300",
                    "train_end_year": 2020,
                    "seeds": "[0,1,2]",
                    "save_name": "test",
                    "zoo_size": 100
                },
                "task_name": "csi300_training"
            }
        }
    )


class TrainResponse(BaseModel):
    """trainingresponsemodel"""
    task_id: str = Field(..., description="task ID")
    status: str = Field(..., description="task status")
    message: str = Field(..., description="response text")
    command: str = Field(..., description="text")
    log_preview: str = Field("", description="log text")
    created_at: str = Field(..., description="time")


class TaskStatus(BaseModel):
    """task statusmodel"""
    task_id: str
    status: str
    command: str
    pid: Optional[int] = None
    created_at: str
    updated_at: str
    log_lines: int = 0
    error: Optional[str] = None


class TaskLog(BaseModel):
    """tasklogmodel"""
    task_id: str
    status: str
    total_lines: int
    logs: List[str]


# ========== tool ==========

def setup_environment() -> dict:
    'environment variable Returns: dict: config environment variabledictionary'
    env = os.environ.copy()
    
    # path PYTHONPATH
    pythonpath = str(REPO_PATH)
    if "PYTHONPATH" in env:
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    
    # Python output
    env["PYTHONUNBUFFERED"] = "1"
    
    logger.info(f"environment variable config: PYTHONPATH={env['PYTHONPATH']}")
    return env


def build_train_command(
    config_file: Optional[str] = None,
    config_params: Optional[Dict[str, Any]] = None
) -> List[str]:
    'training Args: config_file: config file path config_params: config parameter dictionary Returns: List[str]: parameter list Raises: Value Error: parameter invalid'
    # training exists
    script_path = REPO_PATH / TRAIN_SCRIPT
    if not script_path.exists():
        raise ValueError(f"training does not exist: {script_path}")
    
    # basic
    cmd = [sys.executable, str(script_path)]
    
    # config file()
    if config_file:
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = REPO_PATH / config_path
        if not config_path.exists():
            raise ValueError(f"config file does not exist: {config_path}")
        cmd.extend(["--config", str(config_path)])
    
    # config parameter
    if config_params:
        for key, value in config_params.items():
            # list type parameter(seeds)
            if isinstance(value, list):
                cmd.append(f"--{key}={value}")
            else:
                cmd.append(f"--{key}={value}")
    
    logger.info(f"training: {' '.join(cmd)}")
    return cmd


def get_log_preview(logs: deque, lines: int = 10) -> str:
    'getlog Args: logs: log lines: Returns: str: log'
    if not logs:
        return ""
    
    preview_lines = list(logs)[-lines:]
    return ''.join(preview_lines)


def update_task_status(task_id: str, **kwargs) -> None:
    'updatetask status() Args: task_id: task ID **kwargs: update field'
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kwargs)
            tasks[task_id]["updated_at"] = datetime.now().isoformat()


# ========== module interface ==========

def run(args) -> None:
    'module interface, module call Args: args: Simple Namespace, attribute: - config_file: config file path (optional) - config_params: config parameter dictionary(optional),: - instruments: instrument pool, "csi300" - train_end_year: training year - seeds: list, "[0,1,2]" - save_name: save name - zoo_size: Alpha factor size - task_name: task name, default "training"'
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


# ========== task ==========

def _run_training(task_id: str, cmd: List[str]) -> None:
    'training task Args: task_id: task ID cmd: parameterlist'
    logger.info(f"start training task {task_id}")
    
    try:
        #
        env = setup_environment()
        
        # startup training
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
        update_task_status(
            task_id,
            pid=proc.pid,
            status="running"
        )
        
        logger.info(f"training startup, PID: {proc.pid}")
        
        # readoutput
        with tasks_lock:
            log_deque = tasks[task_id]["log"]
        
        for line in proc.stdout:
            line_stripped = line.rstrip()
            
            # log (size)
            with tasks_lock:
                log_deque.append(line_stripped)
                if len(log_deque) > MAX_LOG_LINES:
                    log_deque.popleft()
            
            # output
            logger.info(f"[training {task_id[:8]}] {line_stripped}")
        
        # complete
        return_code = proc.wait()
        
        # updatefinalstatus
        if return_code == 0:
            final_status = "completed"
            logger.info(f"training task {task_id} success complete")
        else:
            final_status = "failed"
            error_msg = f"training: {return_code}"
            logger.error(f"training task {task_id} failed: {error_msg}")
            update_task_status(task_id, error=error_msg)
        
        update_task_status(task_id, status=final_status)
    
    except Exception as e:
        error_msg = f"training taskexception: {str(e)}"
        logger.error(f"training task {task_id} exception: {e}", exc_info=True)
        update_task_status(
            task_id,
            status="failed",
            error=error_msg
        )


# ========== API endpoint ==========

@app.get("/", tags=["basic"])
async def root() -> Dict[str, Any]:
    'service endpoint Returns: service'
    return {
        "service": 'Training MCP - Alpha Forge',
        "algorithm": {
            "name": "AlphaForge",
            "type": "dl",
            "description": "GAN-based Alpha Factor Generation",
            "paradigm": "train_time_generation"
        },
        "repository": str(REPO_PATH),
        "train_script": TRAIN_SCRIPT,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "train_start": "/train/start",
            "train_status": "/train/status/{task_id}",
            "train_log": "/train/log/{task_id}",
            "train_list": "/train/list"
        },
        "status": "running"
    }


@app.post("/train/start", response_model=TrainResponse, tags=["training"])
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
) -> TrainResponse:
    'start training task README description training. supports parameter(config_params): - instruments: data name, "csi300", "csi500" - train_end_year: training year, 2020 - seeds: list, "[0,1,2,3,4]" - save_name: save name - zoo_size: save factorcount - cuda: CUDAdevice - corr_thresh: related threshold - ic_thresh: ICthreshold - icir_thresh: ICIRthreshold Args: request: trainingrequest background_tasks: FastAPI task Returns: Train Response: trainingresponse Raises: HTTPException: startup failed'
    task_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    try:
        # training
        cmd = build_train_command(
            config_file=request.config_file,
            config_params=request.config_params
        )
        command_str = ''.join(cmd)
        
        logger.info(f"training task {task_id}: {request.task_name}")
        logger.info(f"training: {command_str}")
        
        # initialize task
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "started",
                "command": command_str,
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "created_at": created_at,
                "updated_at": created_at,
                "error": None
            }
        
        # training task
        background_tasks.add_task(_run_training, task_id, cmd)
        
        return TrainResponse(
            task_id=task_id,
            status="started",
            message=f"training task startup: {request.task_name}",
            command=command_str,
            log_preview='training start...',
            created_at=created_at
        )
    
    except ValueError as e:
        logger.error(f"parametererror: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"startup training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"startup training failed: {str(e)}")


@app.get("/train/status/{task_id}", response_model=TaskStatus, tags=["training"])
async def get_task_status(task_id: str) -> TaskStatus:
    'gettask status Args: task_id: task ID Returns: Task Status: task status Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        return TaskStatus(
            task_id=task["task_id"],
            status=task["status"],
            command=task["command"],
            pid=task.get("pid"),
            created_at=task["created_at"],
            updated_at=task["updated_at"],
            log_lines=len(task["log"]),
            error=task.get("error")
        )


@app.get("/train/log/{task_id}", response_model=TaskLog, tags=["training"])
async def get_task_log(
    task_id: str,
    lines: int = 100,
    offset: int = 0
) -> TaskLog:
    'gettasklog Args: task_id: task ID lines: return log (default100) offset: log (default0, start) Returns: Task Log: tasklog Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        log_deque = task["log"]
        total_lines = len(log_deque)
        
        # calculatelog
        log_list = list(log_deque)
        start_idx = max(0, total_lines - offset - lines)
        end_idx = total_lines - offset if offset > 0 else total_lines
        
        selected_logs = log_list[start_idx:end_idx]
        
        return TaskLog(
            task_id=task_id,
            status=task["status"],
            total_lines=total_lines,
            logs=selected_logs
        )


@app.get("/train/list", tags=["training"])
async def list_tasks() -> Dict[str, Any]:
    'training task Returns: task statuslist'
    with tasks_lock:
        task_list = []
        for task_id, task in tasks.items():
            task_list.append({
                "task_id": task_id,
                "task_name": task.get("task_name", ""),
                "status": task["status"],
                "created_at": task["created_at"],
                "updated_at": task["updated_at"],
                "log_lines": len(task["log"]),
                "pid": task.get("pid")
            })
        
        # time sort
        task_list.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "total": len(task_list),
            "tasks": task_list
        }


@app.delete("/train/{task_id}", tags=["training"])
async def delete_task(task_id: str) -> Dict[str, str]:
    'deletetask record(deleterecord,) Args: task_id: task ID Returns: delete Raises: HTTPException: task does not exist'
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
        logger.info(f"deletetask record: {task_id}")
        
        return {
            "message": f"task delete: {task_id}",
            "task_id": task_id
        }


@app.get("/health", tags=["basic"])
async def health_check() -> Dict[str, Any]:
    'health check endpoint Returns: service status'
    # check path
    repo_exists = REPO_PATH.exists()
    script_exists = (REPO_PATH / TRAIN_SCRIPT).exists()
    
    with tasks_lock:
        running_tasks = sum(1 for t in tasks.values() if t["status"] == "running")
    
    return {
        "status": "healthy" if repo_exists and script_exists else "degraded",
        "repository_exists": repo_exists,
        "script_exists": script_exists,
        "repository_path": str(REPO_PATH),
        "running_tasks": running_tasks,
        "total_tasks": len(tasks)
    }


# ========== ==========

def main() -> None:
    'startup service'
    #
    if not REPO_PATH.exists():
        logger.warning(f"path does not exist: {REPO_PATH}")
    
    script_path = REPO_PATH / TRAIN_SCRIPT
    if not script_path.exists():
        logger.warning(f"training does not exist: {script_path}")
    
    # startup service
    logger.info("=" * 60)
    logger.info('trainingMCPservice - Alpha Forge')
    logger.info(f"path: {REPO_PATH}")
    logger.info(f"training: {TRAIN_SCRIPT}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info"
    )


if __name__ == "__main__":
    main()
