'trainingMCPservice - AlphaSAGE type: gfn autogeneratetime: 2024'

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Literal
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

# ========== FastAPIapplication ==========
app = FastAPI(
    title="Training MCP - AlphaSAGE",
    description="MCP service starts an AlphaSAGE training task",
    version="1.0.0"
)

# ========== config ==========
REPO_PATH = Path(r"./workspace/AlphaSAGE").resolve()
TRAIN_SCRIPT = 'train_gfn.py'

# task (use data)
tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()

# ========== Pydantic model ==========

class TrainConfigParams(BaseModel):
    """trainingconfigparameter"""
    seed: Optional[int] = Field(default=0, description="text")
    instrument: Optional[str] = Field(default='csi300', description="text")
    pool_capacity: Optional[int] = Field(default=50, description="Alpha pool")
    log_freq: Optional[int] = Field(default=500, description="log text")
    update_freq: Optional[int] = Field(default=64, description="update text")
    n_episodes: Optional[int] = Field(default=10000, description="training text")
    encoder_type: Optional[Literal['transformer', 'lstm', 'gnn']] = Field(
        default='gnn', description="type"
    )
    entropy_coef: Optional[float] = Field(default=0.01, description="text")
    entropy_temperature: Optional[float] = Field(default=1.0, description="text")
    mask_dropout_prob: Optional[float] = Field(default=1.0, description="dropout text")
    ssl_weight: Optional[float] = Field(default=1.0, description="text")
    nov_weight: Optional[float] = Field(default=0.3, description="text")
    weight_decay_type: Optional[Literal['linear', 'exponential', 'polynomial']] = Field(
        default='linear', description="type"
    )
    final_weight_ratio: Optional[float] = Field(default=0.0, description="final text")

    @validator('pool_capacity', 'log_freq', 'update_freq', 'n_episodes')
    def validate_positive_int(cls, v):
        if v <= 0:
            raise ValueError("text")
        return v

    @validator('entropy_coef', 'entropy_temperature', 'mask_dropout_prob', 
               'ssl_weight', 'nov_weight', 'final_weight_ratio')
    def validate_non_negative_float(cls, v):
        if v < 0:
            raise ValueError("text")
        return v


class TrainRequest(BaseModel):
    """trainingrequestmodel"""
    config_file: Optional[str] = Field(None, description='config file path(root directory)')
    config_params: Optional[TrainConfigParams] = Field(None, description="training config parameter")
    task_name: str = Field("training", description="task name")
    
    @validator('task_name')
    def validate_task_name(cls, v):
        if not v or not v.strip():
            raise ValueError('task name is empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "config_params": {
                    "seed": 0,
                    "instrument": "csi300",
                    "pool_capacity": 50,
                    "log_freq": 500,
                    "update_freq": 64,
                    "n_episodes": 10000,
                    "encoder_type": "gnn",
                    "entropy_coef": 0.01,
                    "entropy_temperature": 1.0,
                    "mask_dropout_prob": 1.0,
                    "ssl_weight": 1.0,
                    "nov_weight": 0.3,
                    "weight_decay_type": "linear",
                    "final_weight_ratio": 0.0
                },
                "task_name": "csi300_training"
            }
        }


class TrainResponse(BaseModel):
    """trainingresponsemodel"""
    task_id: str = Field(..., description="task text")
    status: str = Field(..., description="task status")
    message: str = Field(..., description="response text")
    command: str = Field(..., description="training text")
    log_preview: str = Field(default="", description="log text")
    created_at: str = Field(..., description="task time")


class TaskStatus(BaseModel):
    """task statusmodel"""
    task_id: str
    status: str
    command: str
    pid: Optional[int] = None
    created_at: str
    log_lines: int
    log_tail: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# ========== tool ==========

def setup_environment() -> dict:
    'environment variable Returns: dict: environment variabledictionary'
    env = os.environ.copy()
    src_path = str(REPO_PATH / "src")
    pythonpath = str(REPO_PATH) + os.pathsep + src_path
    if "PYTHONPATH" in env:
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    env["PYTHONUNBUFFERED"] = "1"  # Python output
    return env


def build_training_command(
    config_file: Optional[str] = None,
    config_params: Optional[TrainConfigParams] = None
) -> List[str]:
    'training Args: config_file: config file path config_params: config parameter Returns: List[str]: list Raises: Value Error: parameter invalid'
    cmd = [sys.executable, str(REPO_PATH / TRAIN_SCRIPT)]
    
    # config file
    if config_file:
        config_path = REPO_PATH / config_file
        if not config_path.exists():
            raise ValueError(f"config file does not exist: {config_path}")
        cmd.extend(["--config", str(config_path)])
    
    # config parameter
    if config_params:
        params_dict = config_params.dict(exclude_none=True)
        for key, value in params_dict.items():
            cmd.extend([f"--{key}", str(value)])
    
    return cmd


def get_log_preview(log_lines: List[str], num_lines: int = 10) -> str:
    'getlog Args: log_lines: log list num_lines: Returns: str: log'
    if not log_lines:
        return ""
    preview_lines = log_lines[-num_lines:] if len(log_lines) > num_lines else log_lines
    return ''.join(preview_lines)


# ========== task ==========

def _run_training(task_id: str, cmd: List[str]) -> None:
    'training task Args: task_id: task ID cmd: training list'
    try:
        env = setup_environment()
        
        logger.info(f"[task {task_id}] startup training: {' '.join(cmd)}")
        
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
        
        logger.info(f"[task {task_id}] startup, PID: {proc.pid}")
        
        # readoutput
        for line in proc.stdout:
            line_stripped = line.rstrip()
            with tasks_lock:
                tasks[task_id]["log"].append(line_stripped)
            logger.info(f"[task {task_id}] {line_stripped}")
        
        # complete
        return_code = proc.wait()
        
        # updatefinalstatus
        with tasks_lock:
            if return_code == 0:
                tasks[task_id]["status"] = "completed"
                logger.info(f"[task {task_id}] training success complete")
            else:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = f": {return_code}"
                logger.error(f"[task {task_id}] training failed,: {return_code}")
    
    except Exception as e:
        logger.error(f"[task {task_id}] trainingexception: {e}", exc_info=True)
        with tasks_lock:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)


# ========== API endpoint ==========

@app.get("/", tags=["path"])
async def root():
    'service path Returns: dict: service'
    return {
        "service": "Training MCP - AlphaSAGE",
        "algorithm": 'AlphaSAGE (GFlow Net for Alpha Factor Discovery)',
        "algorithm_type": "gfn",
        "repository": str(REPO_PATH),
        "endpoints": {
            "docs": "/docs",
            "openapi": '/openapi.json',
            "train_start": "/train/start",
            "train_status": "/train/status/{task_id}",
            "train_logs": "/train/logs/{task_id}",
            "train_list": "/train/list"
        }
    }


@app.post("/train/start", response_model=TrainResponse, tags=["training"])
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
) -> TrainResponse:
    'start training task Args: request: trainingrequest background_tasks: FastAPI task Returns: Train Response: trainingresponse Raises: HTTPException: startup failed'
    task_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
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
        try:
            cmd = build_training_command(
                config_file=request.config_file,
                config_params=request.config_params
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        command_str = ''.join(cmd)
        logger.info(f"[task {task_id}] startup training: {command_str}")
        
        # initialize task record
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "pending",
                "command": command_str,
                "log": [],
                "pid": None,
                "created_at": created_at,
                "error": None
            }
        
        # task
        background_tasks.add_task(_run_training, task_id, cmd)
        
        # updatestatus startup
        with tasks_lock:
            tasks[task_id]["status"] = "started"
        
        logger.info(f"[task {task_id}] training task")
        
        return TrainResponse(
            task_id=task_id,
            status="started",
            message=f"training task '{request.task_name}' startup",
            command=command_str,
            log_preview="",
            created_at=created_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[task {task_id}] startup training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"startup training failed: {str(e)}")


@app.get("/train/status/{task_id}", response_model=TaskStatus, tags=["training"])
async def get_task_status(task_id: str) -> TaskStatus:
    'querytask status Args: task_id: task ID Returns: Task Status: task status Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        log_tail = list(task["log"])[-20:]
        
        return TaskStatus(
            task_id=task["task_id"],
            status=task["status"],
            command=task["command"],
            pid=task.get("pid"),
            created_at=task["created_at"],
            log_lines=len(task["log"]),
            log_tail=log_tail,
            error=task.get("error")
        )


@app.get("/train/logs/{task_id}", tags=["training"])
async def get_task_logs(
    task_id: str,
    tail: int = 100
) -> JSONResponse:
    'gettasklog Args: task_id: task ID tail: return N log Returns: JSONResponse: logcontent Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = tasks[task_id]
        logs = task["log"]
        
        # get N
        if tail > 0:
            logs = logs[-tail:]
        
        return JSONResponse(content={
            "task_id": task_id,
            "total_lines": len(task["log"]),
            "returned_lines": len(logs),
            "logs": logs
        })


@app.get("/train/list", tags=["training"])
async def list_tasks() -> JSONResponse:
    'task Returns: JSONResponse: task list'
    with tasks_lock:
        task_list = [
            {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "status": task["status"],
                "created_at": task["created_at"],
                "pid": task.get("pid"),
                "log_lines": len(task["log"])
            }
            for task in tasks.values()
        ]
    
    return JSONResponse(content={
        "total": len(task_list),
        "tasks": task_list
    })


@app.get("/health", tags=["health check"])
async def health_check():
    'health check endpoint Returns: dict: status'
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "repository": str(REPO_PATH),
        "repository_exists": REPO_PATH.exists(),
        "train_script_exists": (REPO_PATH / TRAIN_SCRIPT).exists(),
        "active_tasks": sum(1 for t in tasks.values() if t["status"] == "running")
    }


# ========== module interface ==========

def run(args) -> None:
    'module interface, module call Args: args: Simple Namespace, attribute: - config_file: config file path (optional) - config_params: config parameter dictionary(optional),: - seed: - instrument:, "csi300" - pool_capacity: Alpha - log_freq: log - update_freq: update - n_episodes: training - encoder_type: type(transformer/lstm/gnn) - entropy_coef: - entropy_temperature: - mask_dropout_prob: dropout - ssl_weight: - nov_weight: - weight_decay_type: type - final_weight_ratio: final - task_name: task name, default "training"'
    try:
        # argsget parameter
        config_file = getattr(args, "config_file", None)
        config_params_dict = getattr(args, "config_params", None)
        task_name = getattr(args, "task_name", "training")
        
        # convert config_params TrainConfigParams text
        config_params = None
        if config_params_dict:
            if isinstance(config_params_dict, dict):
                config_params = TrainConfigParams(**config_params_dict)
            else:
                # , directly use
                config_params = config_params_dict
        
        # training
        cmd = build_training_command(config_file=config_file, config_params=config_params)
        task_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # initialize task record
        with tasks_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "task_name": task_name,
                "status": "started",
                "command": ''.join(cmd),
                "log": [],
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


# ========== exception ==========

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """global exception"""
    logger.error(f"exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "service error", "error": str(exc)}
    )


# ========== ==========

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("startup AlphaSAGE training MCP service")
    logger.info(f"path: {REPO_PATH}")
    logger.info(f"training: {TRAIN_SCRIPT}")
    logger.info(f"exists: {REPO_PATH.exists()}")
    logger.info(f"exists: {(REPO_PATH / TRAIN_SCRIPT).exists()}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info",
        access_log=True
    )
