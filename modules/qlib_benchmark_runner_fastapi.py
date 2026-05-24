'Qlib Benchmark Runner - FastAPI service using FastAPI Qlib benchmark model'

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
import yaml

# config log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qlib_benchmark_runner. log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qlib Benchmark Runner",
    description='Qlib Benchmark model service',
    version="1.0.0"
)

# ========== config ==========

BENCHMARKS_DIR = Path(os.getenv("QLIB_BENCHMARKS_DIR", "workspace/qlib_benchmark/benchmarks")).expanduser()
DEFAULT_PROVIDER_URI = os.getenv("QLIB_PROVIDER_URI", '~/.qlib/qlib_data/cn_data')
MAX_LOG_LINES = 2000  # maximumkeeplog

# task (use Redis)
run_tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()


# ========== Pydantic model ==========

class ModelInfo(BaseModel):
    """model metadata"""
    path: str = Field(..., description="YAML file path")
    filename: str = Field(..., description="file text")
    model_class: str = Field(..., description="model text")
    dataset: str = Field(..., description="data text")
    market: str = Field(..., description="market")
    benchmark: str = Field(..., description="text")
    provider_uri: str = Field(..., description="current provider_uri")
    start_time: Optional[str] = Field(None, description="datastart time")
    end_time: Optional[str] = Field(None, description="data time")
    train_period: Optional[List] = Field(None, description="training time text")
    valid_period: Optional[List] = Field(None, description="time text")
    test_period: Optional[List] = Field(None, description="time text")


class ModelListResponse(BaseModel):
    """model list response"""
    total_models: int = Field(..., description="model count")
    total_configs: int = Field(..., description="configfile text")
    models: Dict[str, List[str]] = Field(..., description="model dictionary")
    benchmarks_dir: str = Field(..., description="Benchmarks directory path")


class UpdateProviderRequest(BaseModel):
    'update Provider URI request'
    yaml_path: str = Field(..., description='YAML config file path')
    provider_uri: str = Field(..., description="provider_uri path")


class UpdateProviderResponse(BaseModel):
    'update Provider URI response'
    success: bool = Field(..., description="success")
    yaml_path: str = Field(..., description="YAML file path")
    old_uri: str = Field(..., description="provider_uri")
    new_uri: str = Field(..., description="provider_uri")
    message: str = Field(..., description="text")


class RunModelRequest(BaseModel):
    """model request"""
    yaml_path: str = Field(..., description='YAML config file path')
    provider_uri: Optional[str] = Field('~/.qlib/qlib_data/cn_data', description='optional:provider_uri(update)')
    experiment_name: Optional[str] = Field(None, description='optional: name')
    task_name: str = Field(default="qlib_training", description="task name")


class RunModelResponse(BaseModel):
    """model response"""
    task_id: str = Field(..., description="task text")
    status: str = Field(..., description="task status")
    message: str = Field(..., description="status text")
    command: str = Field(..., description="text")
    yaml_path: str = Field(..., description="YAML config path")
    provider_uri: Optional[str] = Field(None, description="useprovider_uri")
    experiment_name: Optional[str] = Field(None, description="name")
    task_name: str = Field(..., description="task name")
    created_at: str = Field(..., description="time")


class TaskStatus(BaseModel):
    """task statusmodel"""
    task_id: str
    status: str
    command: str
    task_name: str
    yaml_path: str
    created_at: str
    pid: Optional[int] = None
    log_lines: int = 0
    last_log: str = ""
    error: Optional[str] = None
    exit_code: Optional[int] = None


class LogResponse(BaseModel):
    """log response model"""
    task_id: str
    total_lines: int
    logs: List[str]
    status: str


class QrunStatusResponse(BaseModel):
    """Qrun status response"""
    available: bool = Field(..., description="qrun available")
    message: str = Field(..., description="status text")
    version_info: Optional[str] = Field(None, description="text")


# ========== tool ==========

def scan_benchmark_models() -> Dict[str, List[str]]:
    'benchmarks directory return available model Returns: model dictionary, model name, YAML file path list'
    models = {}
    
    if not BENCHMARKS_DIR.exists():
        logger.warning(f"Benchmarks directory does not exist: {BENCHMARKS_DIR}")
        return models
    
    # model directory
    for model_dir in BENCHMARKS_DIR.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith('.'):
            # yaml file
            yaml_files = list(model_dir.glob('workflow_config_*. yaml'))
            if yaml_files:
                models[model_dir.name] = [str(f) for f in yaml_files]
    
    logger.info(f"{len(models)} model type")
    return models


def get_yaml_config(yaml_path: str) -> Dict[str, Any]:
    'read YAML config file Args: yaml_path: YAML file path Returns: config dictionary Raises: File Not Found Error: file does not exist. yaml. YAMLError: YAML error'
    yaml_file = Path(yaml_path)
    
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")
    
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def extract_model_info(yaml_path: str) -> ModelInfo:
    'YAML config file extract model Args: yaml_path: YAML file path Returns: model'
    from datetime import date, datetime
    
    def convert_to_str(value):
        """convert date text"""
        if value is None:
            return None
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, list):
            return [convert_to_str(item) for item in value]
        return str(value) if value else None
    
    config = get_yaml_config(yaml_path)
    
    # extract
    task_config = config.get('task', {})
    model_config = task_config.get('model', {})
    dataset_config = task_config.get('dataset', {})
    dataset_kwargs = dataset_config.get('kwargs', {})
    handler_config = dataset_kwargs.get('handler', {})
    handler_kwargs = handler_config.get('kwargs', {})
    
    # extract convert date field
    start_time = convert_to_str(handler_kwargs.get('start_time'))
    end_time = convert_to_str(handler_kwargs.get('end_time'))
    train_period = convert_to_str(dataset_kwargs.get('segments', {}).get('train'))
    valid_period = convert_to_str(dataset_kwargs.get('segments', {}).get('valid'))
    test_period = convert_to_str(dataset_kwargs.get('segments', {}).get('test'))
    
    info = ModelInfo(
        path=yaml_path,
        filename=os.path.basename(yaml_path),
        model_class=model_config.get('class', 'Unknown'),
        dataset=handler_config.get('class', 'Unknown'),
        market=config.get('market', 'Unknown'),
        benchmark=config.get('benchmark', 'Unknown'),
        provider_uri=config.get('qlib_init', {}).get('provider_uri', 'Not specified'),
        start_time=start_time,
        end_time=end_time,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )
    
    return info


def update_yaml_provider_uri(yaml_path: str, provider_uri: str) -> tuple[str, str]:
    'updateYAML config file provider_uri Args: yaml_path: YAML file path provider_uri: provider_uri Returns: (URI, URI)'
    config = get_yaml_config(yaml_path)
    
    # get URI
    old_uri = config.get('qlib_init', {}).get('provider_uri', 'Not set')
    
    # updateprovider_uri
    if 'qlib_init' not in config:
        config['qlib_init'] = {}
    config['qlib_init']['provider_uri'] = provider_uri
    
    # file
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"update provider_uri: {yaml_path}")
    logger.info(f"URI: {old_uri}")
    logger.info(f"URI: {provider_uri}")
    
    return old_uri, provider_uri


def check_qrun_available() -> tuple[bool, str, Optional[str]]:
    'checkqrun available Returns: (available,,)'
    try:
        result = subprocess.run(
            ["qrun", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return True, "qrun available", result.stdout[:200]
        else:
            return False, "qrun returned an error", None
            
    except FileNotFoundError:
        return False, 'qrun not found, installpyqlib', None
    except Exception as e:
        return False, f"checkqrun: {str(e)}", None


def build_qrun_command(yaml_path: str, experiment_name: Optional[str] = None) -> List[str]:
    'qrun Args: yaml_path: YAML config file path experiment_name: optional name Returns: list'
    import sys
    
    # use current Python qrun module
    cmd = [sys.executable, "-m", 'qlib. cli. run', yaml_path]
    
    if experiment_name:
        cmd.extend(["--experiment_name", experiment_name])
    
    return cmd


# ========== module interface ==========

def list_models(args=None) -> None:
    'module interface, available benchmark model Args: args: Simple Namespace (optional, parameter)'
    models = scan_benchmark_models()
    total_configs = sum(len(files) for files in models.values())
    
    logger.info(f"model: {len(models)} model type, {total_configs} config")
    
    # result
    print("=" * 80)
    print("Qlib Benchmark model list")
    print("=" * 80)
    print(f"model typecount: {len(models)}")
    print(f"config file: {total_configs}")
    print(f"Benchmarks directory: {BENCHMARKS_DIR}")
    print()
    
    for model_name, yaml_files in models.items():
        print(f"model: {model_name} ({len(yaml_files)} config)")
        for yaml_file in yaml_files:
            print(f"- {yaml_file}")
        print()
    
    print("=" * 80)


def run(args) -> None:
    'module interface, module call Args: args: Simple Namespace, attribute: - yaml_path: YAML config file path(required) - provider_uri: provider_uri path(optional), update YAML file - experiment_name: name(optional) - task_name: task name, default "qlib_training"'
    try:
        # argsget parameter
        yaml_path = getattr(args, "yaml_path", None)
        if not yaml_path:
            raise ValueError("yaml_path parameter is required")
        
        provider_uri = getattr(args, "provider_uri", None)
        experiment_name = getattr(args, "experiment_name", None)
        task_name = getattr(args, "task_name", "qlib_training")
        
        # provider_uri, update YAML file
        if provider_uri:
            old_uri, new_uri = update_yaml_provider_uri(yaml_path, provider_uri)
            logger.info(f"update provider_uri: {old_uri} -> {new_uri}")
        
        #
        cmd = build_qrun_command(yaml_path, experiment_name)
        task_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # initialize task record
        with tasks_lock:
            run_tasks[task_id] = {
                "task_id": task_id,
                "task_name": task_name,
                "status": "started",
                "command": ''.join(cmd),
                "yaml_path": yaml_path,
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "created_at": created_at,
                "error": None,
                "exit_code": None
            }
        
        logger.info(f"start training task [{task_id}]: {' '.join(cmd)}")
        
        # training task
        thread = threading.Thread(target=_run_qlib_model, args=(task_id, cmd, yaml_path), daemon=False)
        thread.start()
        
        logger.info(f"training task startup [{task_id}], ID: {thread.ident}")
        print(f"training task startup, task ID: {task_id}")
        print(f"use: {' '.join(cmd)}")
        print(f"YAML config: {yaml_path}")
        print(f"task status training service API query: GET/train/status/{task_id}")
        
    except Exception as e:
        error_msg = f"start training task failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


# ========== task ==========

def _run_qlib_model(task_id: str, cmd: List[str], yaml_path: str) -> None:
    'qlibmodeltraining task Args: task_id: task ID cmd: training list yaml_path: YAML config file path'
    try:
        logger.info(f"startup training [{task_id}]: {' '.join(cmd)}")
        
        # environment variable
        env = os.environ.copy()

        yaml_file_path = Path(yaml_path)
        cwd = str(yaml_file_path.parent) if yaml_file_path.exists() else None
        
        # Python output
        env["PYTHONUNBUFFERED"] = "1"
        
        # startup
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=cwd,  # directory
            env=env   # environment variable
        )
        
        # updatetask status
        with tasks_lock:
            run_tasks[task_id]["pid"] = proc.pid
            run_tasks[task_id]["status"] = "running"
        
        logger.info(f"training startup [{task_id}], PID: {proc.pid}")
        
        # read output
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            
            line_stripped = line.rstrip()
            
            #
            with tasks_lock:
                run_tasks[task_id]["log"].append(line_stripped)
                # logsize
                if len(run_tasks[task_id]["log"]) > MAX_LOG_LINES:
                    run_tasks[task_id]["log"].popleft()
            
            # output
            logger.info(f"[training {task_id[:8]}] {line_stripped}")
        
        # complete
        return_code = proc.wait()
        
        # updatefinalstatus
        with tasks_lock:
            run_tasks[task_id]["exit_code"] = return_code
            if return_code == 0:
                run_tasks[task_id]["status"] = "completed"
                logger.info(f"training task complete [{task_id}]")
            else:
                run_tasks[task_id]["status"] = "failed"
                run_tasks[task_id]["error"] = f"training: {return_code}"
                logger.error(f"training task failed [{task_id}],: {return_code}")
    
    except Exception as e:
        error_msg = f"training taskexception: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}", exc_info=True)
        
        with tasks_lock:
            run_tasks[task_id]["status"] = "failed"
            run_tasks[task_id]["error"] = error_msg


# ========== API endpoint ==========

@app.on_event("startup")
async def startup_event() -> None:
    """service startup initialization"""
    logger.info("=" * 60)
    logger.info("Qlib Benchmark Runner service startup")
    logger.info(f"Benchmarks directory: {BENCHMARKS_DIR}")
    logger.info(f"default Provider URI: {DEFAULT_PROVIDER_URI}")
    logger.info("=" * 60)
    
    if not BENCHMARKS_DIR.exists():
        logger.warning(f"warning: Benchmarks directory does not exist: {BENCHMARKS_DIR}")
    
    # check qrun availability
    available, message, _ = check_qrun_available()
    if available:
        logger.info(f"[OK] {message}")
    else:
        logger.warning(f"[WARNING] {message}")


@app.get("/", tags=["basic"])
async def root() -> Dict[str, Any]:
    'service path, return service Returns: service'
    return {
        "service": "Qlib Benchmark Runner",
        "description": 'Qlib Benchmark model service',
        "version": "1.0.0",
        "benchmarks_dir": str(BENCHMARKS_DIR),
        "default_provider_uri": DEFAULT_PROVIDER_URI,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "list_models": 'GET/models/list',
            "get_model_info": 'GET/models/info',
            "update_provider_uri": 'POST/models/update_provider',
            "run_model": 'POST/models/run',
            "check_qrun": 'GET/qrun/status',
            "task_status": 'GET/tasks/status/{task_id}',
            "task_logs": 'GET/tasks/logs/{task_id}',
            "task_list": 'GET/tasks/list'
        }
    }


@app.get("/health", tags=["basic"])
async def health_check() -> Dict[str, str]:
    'health check endpoint Returns: status'
    status = "healthy"
    details = []
    
    # checkbenchmarks directory
    if not BENCHMARKS_DIR.exists():
        status = "degraded"
        details.append(f"Benchmarks directory does not exist: {BENCHMARKS_DIR}")
    
    # checkqrun
    available, message, _ = check_qrun_available()
    if not available:
        status = "degraded"
        details.append(message)
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "details": ';'.join(details) if details else "All OK"
    }


@app.get("/models/list", response_model=ModelListResponse, tags=["model text"])
async def list_models() -> ModelListResponse:
    'available benchmark model Returns: model list'
    models = scan_benchmark_models()
    total_configs = sum(len(files) for files in models.values())
    
    logger.info(f"model: {len(models)} model type, {total_configs} config")
    
    return ModelListResponse(
        total_models=len(models),
        total_configs=total_configs,
        models=models,
        benchmarks_dir=str(BENCHMARKS_DIR)
    )


@app.get("/models/info", response_model=ModelInfo, tags=["model text"])
async def get_model_info(yaml_path: str) -> ModelInfo:
    'get model config details Args: yaml_path: YAML config file path Returns: model details Raises: HTTPException: file does not exist error'
    try:
        info = extract_model_info(yaml_path)
        logger.info(f"get model: {info.filename}")
        return info
        
    except FileNotFoundError as e:
        logger.error(f"file does not exist: {e}")
        raise HTTPException(status_code=404, detail=str(e))
        
    except Exception as e:
        logger.error(f"get model failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"get model failed: {str(e)}")


@app.post("/models/update_provider", response_model=UpdateProviderResponse, tags=["model text"])
async def update_provider_uri(request: UpdateProviderRequest) -> UpdateProviderResponse:
    'updateYAML config file provider_uri Args: request: update request Returns: update result Raises: HTTPException: update failed'
    try:
        old_uri, new_uri = update_yaml_provider_uri(
            request.yaml_path,
            request.provider_uri
        )
        
        return UpdateProviderResponse(
            success=True,
            yaml_path=request.yaml_path,
            old_uri=old_uri,
            new_uri=new_uri,
            message='Provider URI update success'
        )
        
    except FileNotFoundError as e:
        logger.error(f"file does not exist: {e}")
        raise HTTPException(status_code=404, detail=str(e))
        
    except Exception as e:
        logger.error(f"updateprovider_urifailed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"update failed: {str(e)}")


@app.post("/models/run", response_model=RunModelResponse, tags=["model text"])
async def run_model(
    request: RunModelRequest,
    background_tasks: BackgroundTasks
) -> RunModelResponse:
    'qlib benchmark model Args: request: request background_tasks: FastAPI task Returns: task Raises: HTTPException: failed'
    task_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    try:
        # checkYAML file exists
        if not Path(request.yaml_path).exists():
            raise FileNotFoundError(f"YAML file does not exist: {request.yaml_path}")
        
        # provider_uri, update
        if request.provider_uri:
            logger.info(f"updateprovider_uri: {request.provider_uri}")
            update_yaml_provider_uri(request.yaml_path, request.provider_uri)
        
        # check qrun availability
        available, message, _ = check_qrun_available()
        if not available:
            raise RuntimeError(f"qrun unavailable: {message}")
        
        #
        cmd = build_qrun_command(request.yaml_path, request.experiment_name)
        command_str = ''.join(cmd)
        
        logger.info(f"start training task [{task_id}]: {command_str}")
        
        # initialize task record
        with tasks_lock:
            run_tasks[task_id] = {
                "task_id": task_id,
                "task_name": request.task_name,
                "status": "started",
                "command": command_str,
                "yaml_path": request.yaml_path,
                "log": deque(maxlen=MAX_LOG_LINES),
                "pid": None,
                "created_at": created_at,
                "error": None,
                "exit_code": None
            }
        
        # task
        background_tasks.add_task(_run_qlib_model, task_id, cmd, request.yaml_path)
        
        logger.info(f"training task [{task_id}]")
        
        return RunModelResponse(
            task_id=task_id,
            status="started",
            message="training task successfully started",
            command=command_str,
            yaml_path=request.yaml_path,
            provider_uri=request.provider_uri,
            experiment_name=request.experiment_name,
            task_name=request.task_name,
            created_at=created_at
        )
        
    except FileNotFoundError as e:
        logger.error(f"file does not exist: {e}")
        raise HTTPException(status_code=404, detail=str(e))
        
    except RuntimeError as e:
        logger.error(f"error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"startup training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"startup training failed: {str(e)}")


@app.get("/qrun/status", response_model=QrunStatusResponse, tags=["toolcheck"])
async def check_qrun_status() -> QrunStatusResponse:
    'checkqrun available Returns: qrunstatus'
    available, message, version_info = check_qrun_available()
    
    logger.info(f"qrunstatuscheck: {message}")
    
    return QrunStatusResponse(
        available=available,
        message=message,
        version_info=version_info
    )


@app.get("/tasks/status/{task_id}", response_model=TaskStatus, tags=["task text"])
async def get_task_status(task_id: str) -> TaskStatus:
    'gettraining task status Args: task_id: task ID Returns: task status Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in run_tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = run_tasks[task_id]
        last_log = list(task["log"])[-1] if task["log"] else ""
        
        return TaskStatus(
            task_id=task_id,
            status=task["status"],
            command=task["command"],
            task_name=task["task_name"],
            yaml_path=task["yaml_path"],
            created_at=task["created_at"],
            pid=task["pid"],
            log_lines=len(task["log"]),
            last_log=last_log,
            error=task.get("error"),
            exit_code=task.get("exit_code")
        )


@app.get("/tasks/logs/{task_id}", response_model=LogResponse, tags=["task text"])
async def get_task_logs(
    task_id: str,
    lines: int = 100,
    offset: int = 0
) -> LogResponse:
    'gettraining tasklog Args: task_id: task ID lines: return log offset: (start) Returns: log Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in run_tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = run_tasks[task_id]
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


@app.get("/tasks/list", tags=["task text"])
async def list_tasks() -> Dict[str, Any]:
    'training task Returns: task list'
    with tasks_lock:
        task_list = [
            {
                "task_id": task_id,
                "task_name": task["task_name"],
                "status": task["status"],
                "yaml_path": task["yaml_path"],
                "created_at": task["created_at"],
                "pid": task["pid"]
            }
            for task_id, task in run_tasks.items()
        ]
        
        # time sort
        task_list.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "total": len(task_list),
            "tasks": task_list
        }


@app.delete("/tasks/{task_id}", tags=["task text"])
async def delete_task(task_id: str) -> Dict[str, str]:
    'deletetraining task record() Args: task_id: task ID Returns: delete Raises: HTTPException: task does not exist'
    with tasks_lock:
        if task_id not in run_tasks:
            raise HTTPException(status_code=404, detail=f"task does not exist: {task_id}")
        
        task = run_tasks[task_id]
        if task["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail='delete task failed because the task is complete'
            )
        
        del run_tasks[task_id]
        logger.info(f"task delete: {task_id}")
        
        return {"message": f"task {task_id} delete"}


# ========== ==========

if __name__ == "__main__":
    'startup FastAPI service'
    logger.info("=" * 60)
    logger.info("startup Qlib Benchmark Runner service")
    logger.info(f"Benchmarks directory: {BENCHMARKS_DIR}")
    logger.info(f"default Provider URI: {DEFAULT_PROVIDER_URI}")
    logger.info("=" * 60)
    
    # startup service
    uvicorn.run(
        app,
        host="localhost",
        port=8010,
        log_level="info",
        access_log=True
    )

