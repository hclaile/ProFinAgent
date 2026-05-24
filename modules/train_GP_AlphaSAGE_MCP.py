'trainingMCPservice - AlphaSAGE type: Genetic Programming for Alpha Factor Mining factor generation: train_time_generation autogeneratetime: 2024'

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
import json
import threading
from collections import deque

# ========== config log ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_mcp. log')
    ]
)
logger = logging.getLogger(__name__)

# ========== initialize FastAPIapplication ==========
app = FastAPI(
    title="Training MCP - AlphaSAGE",
    description='based on Genetic Programming Alpha factor trainingservice',
    version="1.0.0"
)

# ========== config ==========
REPO_PATH = Path(r"./workspace/AlphaSAGE").resolve()
TRAIN_SCRIPT = 'train_GP.py'
MAX_LOG_LINES = 1000  # maximumsavelog

# task (use Redis)
tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()

# ========== Pydantic model ==========

class TrainRequest(BaseModel):
    """trainingrequestmodel"""
    config_file: Optional[str] = Field(None, description='config file path')
    config_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="trainingparameterconfig",
        example={
            "instruments": "csi300",
            "seed": 0,
            "train_end_year": 2020,
            "freq": "day",
            "cuda": "0"
        }
    )
    task_name: str = Field("training", description="task name")


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
    created_at: str
    updated_at: str
    log_lines: int = 0
    log_preview: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    """task list responsemodel"""
    total: int
    tasks: List[TaskStatus]


# ========== tool ==========

def setup_environment() -> dict:
    'environment variable Returns: dict: environment variabledictionary'
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
    'training Args: config_file: config file path config_params: config parameter dictionary Returns: List[str]: list'
    cmd = [sys.executable, str(REPO_PATH / TRAIN_SCRIPT)]
    
    # config file
    if config_file:
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = REPO_PATH / config_path
        
        if not config_path.exists():
            logger.error(f"config file does not exist: {config_path}")
            #raise File Not Found Error(f"config file does not exist: {config_path}")
        else:
            cmd.extend(["--config", str(config_path)])
    
    # parameter
    if config_params:
        for key, value in config_params.items():
            # convert (argparse parameter format)
            arg_name = key.replace('_', '-')
            #
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{arg_name}")
            else:
                cmd.extend([f"--{arg_name}", str(value)])
    
    logger.info(f"training: {' '.join(cmd)}")
    return cmd


def get_log_preview(log_lines: deque, preview_lines: int = 10) -> str:
    'getlog Args: log_lines: log preview_lines: Returns: str: log'
    if not log_lines:
        return ""
    
    lines = list(log_lines)[-preview_lines:]
    return ''.join(lines)


def validate_repo_path() -> bool:
    'path training exists Returns: bool:'
    if not REPO_PATH.exists():
        logger.error(f"path does not exist: {REPO_PATH}")
        return False
    
    train_script_path = REPO_PATH / TRAIN_SCRIPT
    if not train_script_path.exists():
        logger.error(f"training does not exist: {train_script_path}")
        return False
    
    logger.info(f"path: {REPO_PATH}")
    return True


# ========== module interface ==========

def run(args) -> None:
    'module interface, module call Args: args: Simple Namespace, attribute: - config_file: config file path (optional) - config_params: config parameter dictionary(optional),: - instruments: instrument pool, "csi300" - seed: - train_end_year: training year - freq: data, "day" - cuda: CUDAdevice, "0" - task_name: task name, default "training"'
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
    'training task Args: task_id: task ID cmd: training list'
    try:
        logger.info(f"start training task {task_id}")
        
        # environment variable
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
        with tasks_lock:
            tasks[task_id]["pid"] = proc.pid
            tasks[task_id]["status"] = "running"
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"training startup, PID: {proc.pid}")
        
        # read output
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            
            line_stripped = line.rstrip()
            
            with tasks_lock:
                # usedeque logsize
                tasks[task_id]["log"].append(line_stripped)
                tasks[task_id]["updated_at"] = datetime.now().isoformat()
            
            # record log file
            logger.info(f"[training {task_id}] {line_stripped}")
        
        # complete
        return_code = proc.wait()
        
        # updatefinalstatus
        with tasks_lock:
            if return_code == 0:
                tasks[task_id]["status"] = "completed"
                logger.info(f"training task {task_id} success complete")
            else:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = f": {return_code}"
                logger.error(f"training task {task_id} failed,: {return_code}")
            
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    except Exception as e:
        logger.error(f"training task {task_id} exception: {e}", exc_info=True)
        
        with tasks_lock:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)
            tasks[task_id]["updated_at"] = datetime.now().isoformat()


# ========== API endpoint ==========

@app.on_event("startup")
async def startup_event():
    """application startup"""
    logger.info("=" * 50)
    logger.info("trainingMCPservice startup")
    logger.info(f"name: GP - AlphaSAGE")
    logger.info(f"type: Genetic Programming")
    logger.info(f"path: {REPO_PATH}")
    logger.info(f"training: {TRAIN_SCRIPT}")
    logger.info("=" * 50)
    
    # path
    if not validate_repo_path():
        logger.warning('path failed,')


@app.get("/", tags=["basic"])
async def root():
    """service path"""
    return {
        "service": "Training MCP - AlphaSAGE",
        "algorithm": "Genetic Programming for Alpha Factor Mining",
        "version": "1.0.0",
        "repository": str(REPO_PATH),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "train_start": "/train/start",
            "train_status": "/train/status/{task_id}",
            "train_logs": "/train/logs/{task_id}",
            "train_list": "/train/list",
            "train_stop": "/train/stop/{task_id}",
            "health": "/health"
        }
    }


@app.get("/health", tags=["basic"])
async def health_check():
    """health check endpoint"""
    repo_valid = validate_repo_path()
    
    return {
        "status": "healthy" if repo_valid else "degraded",
        "timestamp": datetime.now().isoformat(),
        "repository_valid": repo_valid,
        "active_tasks": sum(1 for t in tasks.values() if t["status"] == "running")
    }


@app.post("/train/start", response_model=TrainResponse, tags=["training"])
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    'start training task configstartupGPtraining. supports parameter: - instruments: instrument pool (csi300, csi500) - seed: - train_end_year: training year - freq: data (day/minute) - cuda: GPUdevice'
    # path
    if not validate_repo_path():
        raise HTTPException(
            status_code=500,
            detail='path training does not exist, check config'
        )
    
    # generatetask ID
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    try:
        # training
        cmd = build_train_command(
            config_file=request.config_file,
            config_params=request.config_params
        )
        
        command_str = ''.join(cmd)
        
        # initialize task
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "started",
                "command": command_str,
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "created_at": timestamp,
                "updated_at": timestamp,
                "error": None
            }
        
        # task
        background_tasks.add_task(_run_training, task_id, cmd)
        
        logger.info(f"training task {task_id} ({request.task_name})")
        
        return TrainResponse(
            task_id=task_id,
            status="started",
            message="training task successfully started",
            command=command_str,
            log_preview='training start...',
            timestamp=timestamp
        )
    
    except FileNotFoundError as e:
        logger.error(f"filenot found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        logger.error(f"startup training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"startup training failed: {str(e)}")


@app.get("/train/status/{task_id}", response_model=TaskStatus, tags=["training"])
async def get_task_status(task_id: str):
    'gettraining task status return task status, ID, status, log'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task {task_id} does not exist")
        
        task = tasks[task_id]
        
        return TaskStatus(
            task_id=task["task_id"],
            status=task["status"],
            command=task["command"],
            pid=task.get("pid"),
            created_at=task["created_at"],
            updated_at=task["updated_at"],
            log_lines=len(task["log"]),
            log_preview=list(task["log"])[-10:],  # 10
            error=task.get("error")
        )


@app.get("/train/logs/{task_id}", tags=["training"])
async def get_task_logs(
    task_id: str,
    lines: int = 100,
    from_end: bool = True
):
    'gettraining tasklog Args: task_id: task ID lines: return log from_end: startreturn(True log)'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task {task_id} does not exist")
        
        task = tasks[task_id]
        log_lines = list(task["log"])
        
        if from_end:
            selected_logs = log_lines[-lines:]
        else:
            selected_logs = log_lines[:lines]
        
        return {
            "task_id": task_id,
            "total_lines": len(log_lines),
            "returned_lines": len(selected_logs),
            "logs": selected_logs,
            "status": task["status"]
        }


@app.get("/train/list", response_model=TaskListResponse, tags=["training"])
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 50
):
    'training task Args: status: status task (started/running/completed/failed) limit: return maximumtask'
    with tasks_lock:
        task_list = []
        
        for task in tasks.values():
            # status
            if status and task["status"] != status:
                continue
            
            task_list.append(TaskStatus(
                task_id=task["task_id"],
                status=task["status"],
                command=task["command"],
                pid=task.get("pid"),
                created_at=task["created_at"],
                updated_at=task["updated_at"],
                log_lines=len(task["log"]),
                log_preview=list(task["log"])[-5:],  # 5
                error=task.get("error")
            ))
        
        # time sort
        task_list.sort(key=lambda x: x.created_at, reverse=True)
        
        return TaskListResponse(
            total=len(task_list),
            tasks=task_list[:limit]
        )


@app.post("/train/stop/{task_id}", tags=["training"])
async def stop_training(task_id: str):
    'training task training'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task {task_id} does not exist")
        
        task = tasks[task_id]
        
        if task["status"] not in ["running", "started"]:
            raise HTTPException(
                status_code=400,
                detail=f"task status {task['status']},"
            )
        
        pid = task.get("pid")
        
        if not pid:
            raise HTTPException(status_code=400, detail='task IDdoes not exist')
    
    try:
        #
        import signal
        os.kill(pid, signal.SIGTERM)
        
        with tasks_lock:
            tasks[task_id]["status"] = "stopped"
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"training task {task_id} (PID: {pid})")
        
        return {
            "task_id": task_id,
            "message": "training task",
            "pid": pid
        }
    
    except ProcessLookupError:
        raise HTTPException(status_code=404, detail='does not exist')
    
    except Exception as e:
        logger.error(f"task failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"task failed: {str(e)}")


@app.delete("/train/delete/{task_id}", tags=["training"])
async def delete_task(task_id: str):
    'deletetraining task record delete completefailed task record'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task {task_id} does not exist")
        
        task = tasks[task_id]
        
        if task["status"] in ["running", "started"]:
            raise HTTPException(
                status_code=400,
                detail='delete task, task'
            )
        
        del tasks[task_id]
    
    logger.info(f"task {task_id} delete")
    
    return {
        "task_id": task_id,
        "message": "task record deleted"
    }


# ========== ==========

def main():
    """text"""
    logger.info(f"startuptrainingMCPservice")
    logger.info(f"path: {REPO_PATH}")
    logger.info(f"training: {TRAIN_SCRIPT}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info"
    )


if __name__ == "__main__":
    main()