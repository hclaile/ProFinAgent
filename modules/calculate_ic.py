import os
import re
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Qlib data
# ============================================================================

def _is_qlib_data_dir(path: str) -> bool:
    'path available Qlib data directory(cn_data).'
    if not path or not os.path.isdir(path):
        return False
    # Qlib data directory calendars/instruments
    required = ["calendars", "instruments"]
    return all(os.path.isdir(os.path.join(path, d)) for d in required)


def ensure_qlib_cn_data(provider_uri: str) -> None:
    'provider_uri cn_data exists; does not exist try download online. prefer using Qlib tool `qlib.tests.data.GetData`(). download failed, exception readable error.'
    if not provider_uri:
        raise ValueError('provider_uri is empty, Qlib data')

    provider_uri = os.path.expanduser(provider_uri)

    if _is_qlib_data_dir(provider_uri):
        return

    # parent directory, download directory does not exist failed
    os.makedirs(provider_uri, exist_ok=True)

    logger.warning(
        'available Qlib cn_data data directory, download online.'
        f"target={provider_uri}"
    )

    try:
        # Qlib example tool download data
        from qlib.tests.data import GetData  # type: ignore
        import inspect

        getter = GetData()
        fn = getter.qlib_data
        sig = inspect.signature(fn)
        kwargs = {}

        # compatible parameter/
        if "target_dir" in sig.parameters:
            kwargs["target_dir"] = provider_uri
        elif "target" in sig.parameters:
            kwargs["target"] = provider_uri
        else:
            # : position parameter target_dir
            kwargs = None

        # region parameter: use 'cn'/'us'
        if kwargs is not None and "region" in sig.parameters:
            kwargs["region"] = "cn"

        # exists skip(supports)
        if kwargs is not None and "exists_skip" in sig.parameters:
            kwargs["exists_skip"] = True

        if kwargs is None:
            fn(provider_uri, "cn")
        else:
            fn(**kwargs)

    except Exception as e:
        # download Get Data,
        raise RuntimeError(
            'Qlib data directory does not exist, auto download failed.'
            f"check network/proxy available, manual download Qlib cn_data: {provider_uri}."
            f"original error: {e}"
        ) from e

    if not _is_qlib_data_dir(provider_uri):
        raise RuntimeError(
            'try download Qlib cn_data, directory structure complete.'
            f"check download result: {provider_uri}"
        )


# ============================================================================
# Pydantic data model
# ============================================================================

class ICRequest(BaseModel):
    'IC calculate request parameter'
    formula: str = Field(..., description="Qlib format factor expression, 'Ts_Mean($close, 10)/Ts_Mean($close, 30) - 1'")
    instruments: str = Field(default="csi300", description='instrument pool name: csi300, csi500, csi800, csi1000')
    start_date: str = Field(default="2020-01-01", description="start date (YYYY-MM-DD)")
    end_date: str = Field(default="2023-12-31", description='date (YYYY-MM-DD)')
    label_expr: str = Field(
        default="Ref($close, -2)/Ref($close, -1) - 1", 
        description='label expression, calculate future return rate'
    )
    provider_uri: Optional[str] = Field(default=None, description='Qlib data path, default auto detect (qlib_bin directory)')


class ICMetrics(BaseModel):
    'IC metrics result'
    ic_mean: float = Field(..., description='IC mean - factor future return rate related mean')
    ic_std: float = Field(..., description='IC standard deviation - IC series standard deviation, stable')
    ir: float = Field(..., description='information ratio (IR) - IC mean/IC standard deviation,')
    rank_ic_mean: float = Field(..., description='Rank IC mean - use Spearman related calculate IC mean')
    rank_ic_std: float = Field(..., description='Rank IC standard deviation')
    rank_ir: float = Field(..., description='Rank IR - Rank IC mean/Rank IC standard deviation')
    ic_win_rate: float = Field(..., description='IC win rate - IC>0')


class ICResponse(BaseModel):
    """IC calculate response"""
    success: bool = Field(..., description="calculation success")
    metrics: Optional[ICMetrics] = Field(None, description="IC metrics result")
    message: str = Field(default="", description="status text")
    details: Optional[Dict[str, Any]] = Field(None, description="details")


class MCPToolParameter(BaseModel):
    'MCP tool parameter'
    name: str
    description: str
    type: str
    required: bool = True
    default: Optional[Any] = None


class MCPTool(BaseModel):
    'MCP tool'
    name: str
    description: str
    parameters: List[MCPToolParameter]


class MCPToolsResponse(BaseModel):
    """MCP tool list response"""
    tools: List[MCPTool]


class MCPToolCallRequest(BaseModel):
    'MCP tool call request'
    name: str = Field(..., description="tool name")
    arguments: Dict[str, Any] = Field(default={}, description="tool parameter")


class MCPToolCallResponse(BaseModel):
    'MCP tool call response'
    content: List[Dict[str, Any]]
    isError: bool = False


# ============================================================================
# IC calculate (, reference local file)
# ============================================================================


def preprocess_formula(formula: str) -> str:
    'factor expression, convert Qlib supports format: 1. Constant(x) replace direct x 2. format Args: formula: original factor expression Returns: convert Qlib compatible expression'
    if not formula:
        return formula
    
    original_formula = formula
    
    # 1. replace Constant() direct
    # Constant(1.0), Constant(-2.5), Constant(100)
    constant_pattern = r'Constant\s*\(\s*(-\d+\.\d*)\s*\)'
    formula = re.sub(constant_pattern, r'\1', formula)
    
    # 2. extraspace
    formula = re.sub(r'\s+', '', formula).strip()
    
    # 3.common operator ()
    # Qlib use Add, Sub, Mul, Div, supports +, -, *,/
    
    if formula != original_formula:
        logger.info(f"expression: {original_formula} -> {formula}")
    
    return formula


def compute_ic_series(
    factor_values: pd.Series,
    labels: pd.Series,
    method: str = 'pearson'
) -> pd.Series:
    "calculate IC series (panel format data) Args: factor_values: factor (Multi Index: datetime, instrument) labels: return rate label (Multi Index: datetime, instrument) method: 'pearson' 'spearman' Returns: IC series (date)"
    df = pd.concat([factor_values, labels], axis=1, keys=['factor', 'label']).dropna()
    
    def calc_ic(group):
        if len(group) < 3:
            return np.nan
        try:
            if method == 'pearson':
                ic, _ = pearsonr(group['factor'], group['label'])
            else:
                ic, _ = spearmanr(group['factor'], group['label'])
            return ic
        except Exception:
            return np.nan
    
    return df.groupby(level=0).apply(calc_ic)


def compute_factor_metrics(
    factor_values: pd.Series,
    labels: pd.Series
) -> Dict[str, Any]:
    'calculate factor metrics Args: factor_values: factor labels: return rate label Returns: metrics dictionary: - ic_mean: IC mean - ic_std: IC standard deviation - ir: information ratio - rank_ic_mean: Rank IC mean - rank_ic_std: Rank IC standard deviation - rank_ir: Rank IR - ic_win_rate: IC win rate'
    ic_series = compute_ic_series(factor_values, labels, method='pearson')
    rank_ic_series = compute_ic_series(factor_values, labels, method='spearman')
    
    # NaN
    ic_series_clean = ic_series.dropna()
    rank_ic_series_clean = rank_ic_series.dropna()
    
    # calculateIC mean standard deviation
    ic_mean = float(ic_series_clean.mean()) if len(ic_series_clean) > 0 else 0.0
    ic_std = float(ic_series_clean.std()) if len(ic_series_clean) > 0 else 0.0
    
    # calculate Rank IC mean standard deviation
    rank_ic_mean = float(rank_ic_series_clean.mean()) if len(rank_ic_series_clean) > 0 else 0.0
    rank_ic_std = float(rank_ic_series_clean.std()) if len(rank_ic_series_clean) > 0 else 0.0
    
    # information ratio
    ir = ic_mean / ic_std if ic_std > 1e-8 else 0.0
    rank_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-8 else 0.0
    
    # IC win rate
    ic_win_rate = float((ic_series_clean > 0).mean()) if len(ic_series_clean) > 0 else 0.0
    
    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ir': ir,
        'rank_ic_mean': rank_ic_mean,
        'rank_ic_std': rank_ic_std,
        'rank_ir': rank_ir,
        'ic_win_rate': ic_win_rate,
        'ic_series': ic_series_clean,
        'rank_ic_series': rank_ic_series_clean,
        'sample_count': len(ic_series_clean),
    }


def init_qlib(provider_uri: Optional[str] = None) -> tuple:
    'initialize Qlib Args: provider_uri: Qlib data path Returns: (initialize success, actually use data path)'
    try:
        import qlib
        from qlib.config import REG_CN
        
        # expand ~ path
        if provider_uri and provider_uri.startswith('~'):
            provider_uri = os.path.expanduser(provider_uri)

        # user provider_uri directory does not exist/complete, try auto download cn_data
        if provider_uri:
            try:
                if not _is_qlib_data_dir(provider_uri):
                    ensure_qlib_cn_data(provider_uri)
            except Exception as e:
                logger.error(f"Qlib data failed: {e}")
                return False, None
        
        # auto detect data path
        if provider_uri is None or not os.path.exists(provider_uri):
            possible_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qlib_bin'),
                'qlib_bin',
                os.path.abspath('qlib_bin'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    provider_uri = path
                    logger.info(f"auto detect data path: {provider_uri}")
                    break

        # auto detect qlib_bin(directory), is not cn_data structure, download;
        # directory, try download default position.
        if provider_uri is None or not os.path.exists(provider_uri):
            default_cn = os.path.expanduser('~/.qlib/qlib_data/cn_data')
            try:
                ensure_qlib_cn_data(default_cn)
                provider_uri = default_cn
                logger.info(f"use default data path auto download complete: {provider_uri}")
            except Exception as e:
                logger.error(f"not found Qlib data path auto download failed: {e}")
                return False, None
        
        if provider_uri is None or not os.path.exists(provider_uri):
            logger.error("not found Qlib data path")
            return False, None
        
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        logger.info(f"Qlib initialize success, data path: {provider_uri}")
        return True, provider_uri
        
    except Exception as e:
        logger.error(f"Qlib initialize failed: {e}")
        return False, None


def get_instruments_list(
    instruments: str,
    start_date: str,
    end_date: str,
    provider_uri: Optional[str] = None
) -> List[str]:
    'getstock list Args: instruments: instrument pool name start_date: start date end_date: date provider_uri: data path Returns: ticker list'
    instrument_list = None
    
    # 1: use D.list_instruments() getstock list
    try:
        from qlib.data import D
        
        # use list_instruments getstock list
        instrument_list = D.list_instruments(
            instruments=D.instruments(instruments),
            start_time=start_date,
            end_time=end_date,
            as_list=True
        )
        
        if instrument_list and len(instrument_list) > 0:
            logger.info(f"use D.list_instruments get {len(instrument_list)} stock")
            return list(instrument_list)
            
    except Exception as e:
        logger.warning(f"D.list_instruments get failed: {e}, try file read")
    
    # 2: file direct read
    logger.info('try file direct read stock list...')
    
    # provider_uri
    if provider_uri is None:
        # use global save path
        global _qlib_provider_uri
        provider_uri = _qlib_provider_uri
    
    if provider_uri is None:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qlib_bin'),
            'qlib_bin',
            os.path.abspath('qlib_bin'),
            r'C:\edge_download\python_project\Qlib_project\qlib_bin',
            r'C:\edge_download\python_project\Qlib_project\MCTS_QCM\qlib_bin',
        ]
        for path in possible_paths:
            test_file = os.path.join(path, 'instruments', f'{instruments}.txt')
            if os.path.exists(test_file):
                provider_uri = path
                logger.info(f"data path: {provider_uri}")
                break
    
    if provider_uri:
        file_path = os.path.join(provider_uri, 'instruments', f'{instruments}.txt')
        logger.info(f"try to read instrument pool file: {file_path}")
        
        if os.path.exists(file_path):
            stocks_from_file = set()
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('')
                        if len(parts) >= 1:
                            stock_code = parts[0]
                            if len(parts) >= 3:
                                stock_start = parts[1]
                                stock_end = parts[2]
                                # checkdate range
                                if stock_start <= end_date and stock_end >= start_date:
                                    stocks_from_file.add(stock_code)
                            else:
                                stocks_from_file.add(stock_code)
            
            if len(stocks_from_file) > 0:
                instrument_list = sorted(list(stocks_from_file))
                logger.info(f"file read {len(instrument_list)} stock")
        else:
            logger.error(f"instrument pool file does not exist: {file_path}")
    else:
        logger.error('not found data path, read instrument pool file')
    
    return instrument_list or []


def calculate_ic(
    formula: str,
    instruments: str,
    start_date: str,
    end_date: str,
    label_expr: str = 'Ref($close, -2)/Ref($close, -1) - 1',
    provider_uri: Optional[str] = None
) -> Dict[str, Any]:
    'calculate factor IC metrics Args: formula: Qlib format factor expression instruments: instrument pool name start_date: start date end_date: date label_expr: label expression provider_uri: Qlib data path Returns: IC metrics dictionary'
    try:
        from qlib.data import D
        
        # save original expression
        original_formula = formula
        original_label_expr = label_expr
        
        # expression, Constant(x) convert Qlib supports format
        formula = preprocess_formula(formula)
        label_expr = preprocess_formula(label_expr)
        
        logger.info(f"calculate factor: {formula}")
        logger.info(f"instrument pool: {instruments}")
        logger.info(f"date range: {start_date} {end_date}")
        logger.info(f"label expression: {label_expr}")
        
        # use global save provider_uri
        global _qlib_provider_uri
        if provider_uri is None:
            provider_uri = _qlib_provider_uri
        
        # getstock list
        instrument_list = get_instruments_list(instruments, start_date, end_date, provider_uri)
        
        if len(instrument_list) == 0:
            raise ValueError(f"instrument pool {instruments} get stock")
        
        logger.info(f"get {len(instrument_list)} stock")
        
        # calculate factor
        logger.info('calculate factor...')
        factor_df = D.features(
            instrument_list,
            [formula],
            start_time=start_date,
            end_time=end_date
        )
        
        # calculate label
        logger.info('calculate label...')
        label_df = D.features(
            instrument_list,
            [label_expr],
            start_time=start_date,
            end_time=end_date
        )
        
        # convert Series format
        if isinstance(factor_df, pd.DataFrame):
            factor_values = factor_df.iloc[:, 0]
        else:
            factor_values = factor_df
        
        if isinstance(label_df, pd.DataFrame):
            labels = label_df.iloc[:, 0]
        else:
            labels = label_df
        
        # name
        if isinstance(factor_values.index, pd.MultiIndex):
            factor_values.index.names = ['datetime', 'instrument']
        if isinstance(labels.index, pd.MultiIndex):
            labels.index.names = ['datetime', 'instrument']
        
        logger.info(f"factor calculation complete, {len(factor_values)} record")
        logger.info(f"label calculation complete, {len(labels)} record")
        
        # factor label
        aligned = pd.concat([factor_values, labels], axis=1, keys=['factor', 'label']).dropna()
        
        if len(aligned) < 10:
            raise ValueError(f"data: {len(aligned)}, check expression")
        
        logger.info(f"data: {len(aligned)} record")
        
        # extract data
        factor_aligned = aligned['factor']
        labels_aligned = aligned['label']
        
        # calculate IC metrics
        logger.info('calculate IC metrics...')
        metrics = compute_factor_metrics(factor_aligned, labels_aligned)
        
        return {
            'success': True,
            'metrics': {
                'ic_mean': float(metrics['ic_mean']),
                'ic_std': float(metrics['ic_std']),
                'ir': float(metrics['ir']),
                'rank_ic_mean': float(metrics['rank_ic_mean']),
                'rank_ic_std': float(metrics['rank_ic_std']),
                'rank_ir': float(metrics['rank_ir']),
                'ic_win_rate': float(metrics['ic_win_rate']),
            },
            'details': {
                'formula': formula,
                'original_formula': original_formula,
                'instruments': instruments,
                'start_date': start_date,
                'end_date': end_date,
                'label_expr': label_expr,
                'original_label_expr': original_label_expr,
                'stock_count': len(instrument_list),
                'sample_count': metrics.get('sample_count', 0),
                'data_records': len(aligned),
            }
        }
        
    except ImportError:
        logger.error('Qlib not installed, install: pip install pyqlib')
        raise HTTPException(status_code=500, detail="Qlib not installed")
    except Exception as e:
        logger.error(f"calculate IC: {e}", exc_info=True)
        raise


# ============================================================================
# FastAPI application
# ============================================================================

app = FastAPI(
    title="MCP IC calculation service",
    description='based on MCP (Model Context Protocol) factor IC calculation service: - calculate factor return rate IC - supports Qlib format factor expression - supports instrument pool (csi300, csi500, csi800, csi1000) - return complete IC metrics: IC mean, IC standard deviation, IR, Rank IC, IC win rate MCP protocol endpoints: - GET/tools: get available tool list - POST/tools/call: call tool',
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qlib initializestatus
_qlib_initialized = False
_qlib_provider_uri: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    'application startup initialize Qlib'
    global _qlib_initialized, _qlib_provider_uri
    _qlib_initialized, _qlib_provider_uri = init_qlib()
    if _qlib_initialized:
        logger.info(f"service startup success, Qlib initialize, data path: {_qlib_provider_uri}")
    else:
        logger.warning('service startup, Qlib initialize failed')


# ============================================================================
# MCP protocol endpoints
# ============================================================================

@app.get("/tools", response_model=MCPToolsResponse)
async def get_tools():
    """get available tool list (MCP protocol)"""
    tools = [
        MCPTool(
            name="calculate_ic",
            description='calculate factor IC metrics. return IC mean, IC standard deviation, information ratio (IR), Rank IC mean, Rank IC standard deviation, Rank IR, IC win rate.',
            parameters=[
                MCPToolParameter(
                    name="formula",
                    description="Qlib format factor expression, 'Ts_Mean($close, 10)/Ts_Mean($close, 30) - 1'",
                    type="string",
                    required=True
                ),
                MCPToolParameter(
                    name="instruments",
                    description='instrument pool name: csi300, csi500, csi800, csi1000, all',
                    type="string",
                    required=False,
                    default="csi300"
                ),
                MCPToolParameter(
                    name="start_date",
                    description="start date (YYYY-MM-DD)",
                    type="string",
                    required=False,
                    default="2020-01-01"
                ),
                MCPToolParameter(
                    name="end_date",
                    description='date (YYYY-MM-DD)',
                    type="string",
                    required=False,
                    default="2023-12-31"
                ),
                MCPToolParameter(
                    name="label_expr",
                    description='label expression, calculate future return rate',
                    type="string",
                    required=False,
                    default="Ref($close, -2)/Ref($close, -1) - 1"
                ),
            ]
        )
    ]
    return MCPToolsResponse(tools=tools)


@app.post("/tools/call", response_model=MCPToolCallResponse)
async def call_tool(request: MCPToolCallRequest):
    'call tool (MCP protocol)'
    global _qlib_initialized, _qlib_provider_uri
    
    if request.name == "calculate_ic":
        try:
            # Qlib initialize
            if not _qlib_initialized:
                provider_uri = request.arguments.get("provider_uri")
                _qlib_initialized, _qlib_provider_uri = init_qlib(provider_uri)
                if not _qlib_initialized:
                    return MCPToolCallResponse(
                        content=[{
                            "type": "text",
                            "text": 'error: Qlib initialize failed, check data path config'
                        }],
                        isError=True
                    )
            
            # extract parameter
            formula = request.arguments.get("formula")
            if not formula:
                return MCPToolCallResponse(
                    content=[{
                        "type": "text",
                        "text": "error: missing parameter 'formula'"
                    }],
                    isError=True
                )
            
            instruments = request.arguments.get("instruments", "csi300")
            start_date = request.arguments.get("start_date", "2020-01-01")
            end_date = request.arguments.get("end_date", "2023-12-31")
            label_expr = request.arguments.get("label_expr", "Ref($close, -2)/Ref($close, -1) - 1")
            provider_uri = request.arguments.get("provider_uri")
            
            # calculate IC
            result = calculate_ic(
                formula=formula,
                instruments=instruments,
                start_date=start_date,
                end_date=end_date,
                label_expr=label_expr,
                provider_uri=provider_uri
            )
            
            # format output
            metrics = result['metrics']
            details = result['details']
            
            output_text = f"""IC calculation result ================================================================================ factor expression: {details['formula']} instrument pool: {details['instruments']} ({details['stock_count']} stock) date range: {details['start_date']} {details['end_date']} data record: {details['data_records']} ================================================================================ [IC metrics] IC mean (IC_mean): {metrics['ic_mean']:.6f} IC standard deviation (IC_std): {metrics['ic_std']:.6f} information ratio (IR): {metrics['ir']:.6f} Rank IC mean: {metrics['rank_ic_mean']:.6f} Rank IC standard deviation: {metrics['rank_ic_std']:.6f} Rank IR: {metrics['rank_ir']:.6f} IC win rate: {metrics['ic_win_rate']:.2%} ================================================================================"""
            
            return MCPToolCallResponse(
                content=[
                    {
                        "type": "text",
                        "text": output_text
                    },
                    {
                        "type": "json",
                        "data": result
                    }
                ],
                isError=False
            )
            
        except Exception as e:
            logger.error(f"calculate IC: {e}", exc_info=True)
            return MCPToolCallResponse(
                content=[{
                    "type": "text",
                    "text": f"error: {str(e)}"
                }],
                isError=True
            )
    else:
        return MCPToolCallResponse(
            content=[{
                "type": "text",
                "text": f"tool: {request.name}"
            }],
            isError=True
        )


# ============================================================================
# module interface (module call)
# ============================================================================

def run(args) -> None:
    'module interface, module call Args: args: Simple Namespace, attribute: - formula: Qlib format factor expression (required) - instruments: instrument pool name, default "csi300" - start_date: start date, default "2020-01-01" - end_date: date, default "2023-12-31" - label_expr: label expression, default "Ref($close, -2)/Ref($close, -1) - 1" - provider_uri: Qlib data path, default None(auto detect)'
    try:
        # argsget parameter
        formula = getattr(args, "formula", None)
        if not formula:
            raise ValueError("formula parameter is required")
        
        instruments = getattr(args, "instruments", "csi300")
        start_date = getattr(args, "start_date", "2020-01-01")
        end_date = getattr(args, "end_date", "2023-12-31")
        label_expr = getattr(args, "label_expr", "Ref($close, -2)/Ref($close, -1) - 1")
        provider_uri = getattr(args, "provider_uri", None) or os.getenv("QLIB_PROVIDER_URI")
        
        # Qlib initialize
        global _qlib_initialized, _qlib_provider_uri
        if not _qlib_initialized:
            _qlib_initialized, _qlib_provider_uri = init_qlib(provider_uri)
            if not _qlib_initialized:
                raise RuntimeError('Qlib initialize failed, check data path config')
        
        # calculate IC
        result = calculate_ic(
            formula=formula,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            label_expr=label_expr,
            provider_uri=provider_uri
        )
        
        # result
        print("=" * 80)
        print('IC calculation complete')
        print("=" * 80)
        print(f"factor expression: {formula}")
        print(f"instrument pool: {instruments}")
        print(f"date range: {start_date} {end_date}")
        print()
        print('IC metrics result:')
        print(f"IC mean: {result['metrics']['ic_mean']:.6f}")
        print(f"IC standard deviation: {result['metrics']['ic_std']:.6f}")
        print(f"information ratio (IR): {result['metrics']['ir']:.6f}")
        print(f"Rank IC mean: {result['metrics']['rank_ic_mean']:.6f}")
        print(f"Rank IC standard deviation: {result['metrics']['rank_ic_std']:.6f}")
        print(f"Rank IR: {result['metrics']['rank_ir']:.6f}")
        print(f"IC win rate: {result['metrics']['ic_win_rate']:.4f}")
        print("=" * 80)
        
        logger.info("IC calculate complete")
        
    except Exception as e:
        error_msg = f"calculate IC failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"error: {error_msg}")
        raise RuntimeError(error_msg)


# ============================================================================
# REST API endpoint (directcall)
# ============================================================================

@app.post("/calculate_ic", response_model=ICResponse)
async def api_calculate_ic(request: ICRequest):
    'calculate factor IC metrics (REST API) Args: request: IC calculate request parameter Returns: IC metrics result'
    global _qlib_initialized, _qlib_provider_uri
    
    try:
        # Qlib initialize
        if not _qlib_initialized:
            _qlib_initialized, _qlib_provider_uri = init_qlib(request.provider_uri)
            if not _qlib_initialized:
                return ICResponse(
                    success=False,
                    message='Qlib initialize failed, check data path config'
                )
        
        # calculate IC
        result = calculate_ic(
            formula=request.formula,
            instruments=request.instruments,
            start_date=request.start_date,
            end_date=request.end_date,
            label_expr=request.label_expr,
            provider_uri=request.provider_uri
        )
        
        return ICResponse(
            success=True,
            metrics=ICMetrics(**result['metrics']),
            message="calculate success",
            details=result['details']
        )
        
    except Exception as e:
        logger.error(f"calculate IC: {e}", exc_info=True)
        return ICResponse(
            success=False,
            message=f"calculate failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """health check endpoint"""
    return {
        "status": "healthy",
        "qlib_initialized": _qlib_initialized,
        "provider_uri": _qlib_provider_uri,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    'path, return service'
    return {
        "service": "MCP IC calculation service",
        "version": "1.0.0",
        "description": 'based on MCP protocol factor IC calculation service',
        "endpoints": {
            "MCP": {
                'GET/tools': "get available tool list",
                'POST/tools/call': "call tool"
            },
            "REST API": {
                'POST/calculate_ic': "calculate factor IC metrics"
            },
            "text": {
                'GET/health': "health check",
                'GET/docs': 'Swagger',
                'GET/redoc': 'Re Doc'
            }
        }
    }


# ============================================================================
#
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("MCP IC calculation service")
    print("=" * 80)
    print()
    print('service: localhost:8000')
    print('Swagger:/docs')
    print('Re Doc:/redoc')
    print()
    print("MCP endpoint:")
    print('GET/tools - get available tool list')
    print('POST/tools/call - call tool')
    print()
    print("REST API endpoint:")
    print('POST/calculate_ic - calculate factor IC metrics')
    print()
    print("=" * 80)
    
    uvicorn.run(app, host="localhost", port=8000)

