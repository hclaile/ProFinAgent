'technical indicator calculation module - based on TA-Lib supports 150+ technical indicator, overwrite,,,'
import os
import sys
import json
import inspect
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Union, Tuple

import pandas as pd
import numpy as np

# try TA-Lib
try:
    import talib
except ImportError:
    talib = None
    print('warning: TA-Lib not installed, please run: pip install TA-Lib', file=sys.stderr)

# metrics
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from indicator_whitelist import SUPPORTED_INDICATORS
except ImportError:
    # failed, use list
    SUPPORTED_INDICATORS = [
        "SMA", "EMA", "MACD", "RSI", "BBANDS", "STOCH", 
        "ATR", "ADX", "CCI", "MOM", "ROC", "WILLR"
    ]


def _print(*args, **kwargs):
    'print to stderr, avoid interfering with MCP JSON-RPC communication'
    print(*args, file=sys.stderr, **kwargs)


def _is_datetime_like(obj: Any) -> bool:
    ': datetime type supports: - pandas Timestamp - Python datetime/date - pandas Period - datetime64 type - datetime (arrow, pendulum, strftime) - numpy datetime64 Args: obj: Returns: True datetime type, False'
    if obj is None:
        return False
    
    # type check
    if isinstance(obj, (pd.Timestamp, datetime, pd.Period)):
        return True
    
    # check datetime64 type
    if pd.api.types.is_datetime64_any_dtype(type(obj)):
        return True
    
    # check numpy datetime64
    if isinstance(obj, np.datetime64):
        return True
    
    # check strftime (arrow, pendulum)
    if hasattr(obj, 'strftime') and callable(getattr(obj, 'strftime', None)):
        # : trycall strftime datetime
        try:
            obj.strftime('%Y-%m-%d')
            return True
        except (AttributeError, TypeError, ValueError):
            pass
    
    # check date type
    try:
        from datetime import date
        if isinstance(obj, date):
            return True
    except ImportError:
        pass
    
    return False


def _convert_datetime_to_string(obj: Any) -> str:
    'convert: datetime convert supports _is_datetime_like type Args: obj: datetime Returns: format'
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, pd.Period):
        return str(obj)
    elif isinstance(obj, np.datetime64):
        try:
            return pd.to_datetime(obj).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return str(obj)
    elif pd.api.types.is_datetime64_any_dtype(type(obj)):
        try:
            return pd.to_datetime(obj).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return str(obj)
    elif hasattr(obj, 'strftime'):
        # datetime
        try:
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                return obj.strftime('%Y-%m-%d')
            except Exception:
                return str(obj)
    else:
        # : convert
        return str(obj)


class TimestampJSONEncoder(json.JSONEncoder):
    'JSON, auto Timestamp datetime, datetime serialize: use type convert,'
    def default(self, obj):
        # use datetime convert
        if _is_datetime_like(obj):
            return _convert_datetime_to_string(obj)
        
        # numpy numerictype
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # numpy
        elif isinstance(obj, np.ndarray):
            # (datetime)
            try:
                return obj.tolist()
            except (TypeError, ValueError):
                # tolist failed, try convert
                return [_convert_datetime_to_string(item) if _is_datetime_like(item) else item for item in obj]
        # numpy datetime64
        elif isinstance(obj, (np.datetime64,)):
            return _convert_datetime_to_string(obj)
        # type
        return super().default(obj)


def _convert_timestamps_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    'Data Frame Timestamp convert, supports JSON serialize: datetime type,: - Datetime Index - datetime64 type - object type Timestamp - Period Args: df: input Data Frame Returns: convert Data Frame(Timestamp/Period)'
    df = df.copy()
    
    # Timestamp/Period(Multi Index)
    if isinstance(df.index, pd.MultiIndex):
        # Multi Index: check
        try:
            new_levels_data = []
            for i in range(df.index.nlevels):
                level_values = df.index.get_level_values(i)
                # check datetime type
                if len(level_values) > 0 and _is_datetime_like(level_values[0]):
                    new_levels_data.append([_convert_datetime_to_string(v) if _is_datetime_like(v) else v for v in level_values])
                else:
                    new_levels_data.append(list(level_values))
            # Multi Index
            df.index = pd.MultiIndex.from_arrays(new_levels_data)
        except Exception as e:
            # Multi Index convert failed, record warning continue
            _print(f"WARN warning: Multi Index datetime convert failed: {str(e)[:100]}")
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime('%Y-%m-%d')
    elif isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.astype(str)
    elif hasattr(df.index, 'dtype'):
        if pd.api.types.is_datetime64_any_dtype(df.index.dtype):
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        elif pd.api.types.is_period_dtype(df.index.dtype):
            df.index = df.index.astype(str)
        # check datetime (object type)
        elif len(df.index) > 0:
            sample_idx = df.index[0]
            if _is_datetime_like(sample_idx):
                df.index = [_convert_datetime_to_string(v) if _is_datetime_like(v) else v for v in df.index]
    
    # Timestamp/Period
    for col in df.columns:
        col_dtype = df[col].dtype
        
        # 1: datetime64 type
        if pd.api.types.is_datetime64_any_dtype(col_dtype):
            try:
                # convert, keepdate (time format)
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
            except Exception:
                # convert failed, try direct
                df[col] = df[col].astype(str)
        
        # 2: Period type
        elif pd.api.types.is_period_dtype(col_dtype):
            df[col] = df[col].astype(str)
        
        # 3: object type(Timestamp)
        elif col_dtype == 'object':
            try:
                # use type, check
                has_datetime = False
                
                # check empty(check20, overwrite)
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample_size = min(20, len(non_null_values))
                    for val in non_null_values.head(sample_size):
                        if _is_datetime_like(val):
                            has_datetime = True
                            break
                        # check structure(list,, dictionary)
                        elif isinstance(val, (list, tuple)):
                            for item in val[:5]:  # check 5
                                if _is_datetime_like(item):
                                    has_datetime = True
                                    break
                            if has_datetime:
                                break
                        elif isinstance(val, dict):
                            for item in list(val.values())[:5]:  # check 5
                                if _is_datetime_like(item):
                                    has_datetime = True
                                    break
                            if has_datetime:
                                break
                
                # datetime, convert
                if has_datetime:
                    def _convert_object_type(x):
                        if pd.isna(x):
                            return x
                        # use convert
                        if _is_datetime_like(x):
                            return _convert_datetime_to_string(x)
                        # structure
                        elif isinstance(x, (list, tuple)):
                            converted = [_convert_datetime_to_string(item) if _is_datetime_like(item) else item for item in x]
                            return tuple(converted) if isinstance(x, tuple) else converted
                        elif isinstance(x, dict):
                            return {k: (_convert_datetime_to_string(v) if _is_datetime_like(v) else v) for k, v in x.items()}
                        else:
                            return x
                    
                    df[col] = df[col].apply(_convert_object_type)
            except Exception as e:
                # convert failed, try
                try:
                    # convert: try convert(use)
                    def _convert_aggressive(x):
                        if pd.isna(x):
                            return x
                        # use convert
                        if _is_datetime_like(x):
                            return _convert_datetime_to_string(x)
                        # structure
                        elif isinstance(x, (list, tuple)):
                            converted = [_convert_datetime_to_string(item) if _is_datetime_like(item) else item for item in x]
                            return tuple(converted) if isinstance(x, tuple) else converted
                        elif isinstance(x, dict):
                            return {k: (_convert_datetime_to_string(v) if _is_datetime_like(v) else v) for k, v in x.items()}
                        # try datetime(use, convert)
                        elif isinstance(x, str) and len(x) > 0:
                            # try date
                            if any(char.isdigit() for char in x[:10]):  # 10
                                try:
                                    dt = pd.to_datetime(x)
                                    if not pd.isna(dt):
                                        return dt.strftime('%Y-%m-%d')
                                except:
                                    pass
                        return x
                    
                    df[col] = df[col].apply(_convert_aggressive)
                except Exception as e2:
                    # failed, record warning continue
                    _print(f"WARN warning: '{col}' Timestamp convert failed: {str(e2)[:100]}")
                    pass  # failed,
    
    return df


def _convert_dict_timestamps_to_strings(obj: Any, seen: Optional[set] = None) -> Any:
    'convert dictionary, list Timestamp: use type convert, datetime type Args: obj: dict, list, Timestamp type seen: reference (use) Returns: convert'
    # reference
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return "[Circular Reference]"
    seen.add(obj_id)
    
    try:
        # use datetime convert
        if _is_datetime_like(obj):
            result = _convert_datetime_to_string(obj)
            seen.remove(obj_id)
            return result
        # dictionary
        elif isinstance(obj, dict):
            result = {k: _convert_dict_timestamps_to_strings(v, seen) for k, v in obj.items()}
            seen.remove(obj_id)
            return result
        # list
        elif isinstance(obj, (list, tuple)):
            result = [_convert_dict_timestamps_to_strings(item, seen) for item in obj]
            seen.remove(obj_id)
            # originaltype(list tuple)
            if isinstance(obj, tuple):
                return tuple(result)
            return result
        # numpy numerictype
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            seen.remove(obj_id)
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            seen.remove(obj_id)
            return float(obj)
        elif isinstance(obj, np.bool_):
            seen.remove(obj_id)
            return bool(obj)
        # numpy
        elif isinstance(obj, np.ndarray):
            result = obj.tolist()
            result = _convert_dict_timestamps_to_strings(result, seen)
            seen.remove(obj_id)
            return result
        # type(check strftime)
        elif hasattr(obj, 'strftime'):
            try:
                result = obj.strftime('%Y-%m-%d %H:%M:%S')
                seen.remove(obj_id)
                return result
            except Exception:
                seen.remove(obj_id)
                return str(obj)
        # type, direct return
        else:
            seen.remove(obj_id)
            return obj
    except Exception as e:
        # convert, return
        seen.discard(obj_id)  # seen
        try:
            return str(obj)
        except Exception:
            return f"[Conversion Error: {type(obj).__name__}]"


def _load_json_file_smart(input_path: str) -> pd.DataFrame:
    'load JSON file, supports format: 1. structure(fmp_historical_data output):{"data": [...]} 2. record format:[{...}, {...}] 3.:{...} Args: input_path: JSON file path Returns: Data Frame: load data'
    try:
        # read normal JSON, checkstructure
        _print(f"read JSON file: {input_path}")
        file_size = os.path.getsize(input_path)
        _print(f"filesize: {file_size / 1024:.2f} KB")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        _print(f"OK JSON success, datatype: {type(raw_data).__name__}")
        
        # 1: structure(fmp_historical_data output format)
        if isinstance(raw_data, dict):
            _print(f"dictionarystructure,: {list(raw_data.keys())[:10]}")
            # check 'data' field
            if 'data' in raw_data:
                data_list = raw_data['data']
                if isinstance(data_list, list):
                    if len(data_list) > 0:
                        # extract data field convert Data Frame
                        df = pd.DataFrame(data_list)
                        # convertdate, pandas auto datetime type
                        df = _convert_timestamps_to_strings(df)
                        _print(f"OK JSON structure, extract 'data' field: {len(df)} {len(df.columns)}")
                        return df
                    else:
                        raise ValueError(f"JSON structure 'data' fieldis empty list, convert Data Frame")
                else:
                    raise ValueError(f"JSON structure 'data' fieldis not a list type, actualtype: {type(data_list).__name__}")
            # , try direct convert Data Frame
            elif len(raw_data) > 0:
                df = pd.DataFrame([raw_data])
                # convertdate
                df = _convert_timestamps_to_strings(df)
                _print(f"OK JSON, convert Data Frame: {len(df)} {len(df.columns)}")
                return df
            else:
                raise ValueError(f"JSON dictionaryis empty, convert Data Frame")
        
        # 2: record format
        elif isinstance(raw_data, list):
            if len(raw_data) > 0:
                # check list type
                first_elem = raw_data[0]
                if not isinstance(first_elem, dict):
                    raise ValueError(f"JSON is notdictionary type, actualtype: {type(first_elem).__name__}. format: [{{...}}, {{...}}]")
                
                df = pd.DataFrame(raw_data)
                # convertdate, pandas auto datetime type
                df = _convert_timestamps_to_strings(df)
                _print(f"OK record JSON format: {len(df)} {len(df.columns)}")
                return df
            else:
                raise ValueError(f"JSON is empty, convert Data Frame")
        
        # , error
        raise ValueError(
            f"JSON structure: dict list, actual {type(raw_data).__name__}\n"
            f"supports format:\n"
            f"1. structure: {{\"data\": [{{...}}, {{...}}]}}\n"
            f"2. record: [{{...}}, {{...}}]\n"
            f"3.: {{...}}"
        )
        
    except json.JSONDecodeError as e:
        error_msg = f"JSON failed: {str(e)}"
        # error
        if hasattr(e, 'pos'):
            error_msg += f"\n errorposition: {e.lineno}, {e.colno}"
        error_msg += f"\n: check JSON file format (,,)"
        raise ValueError(error_msg)
    except UnicodeDecodeError as e:
        raise ValueError(
            f"file error: {str(e)}\n"
            f": JSON file UTF-8, check file format"
        )
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        raise ValueError(
            f"read JSON file failed ({error_type}): {error_msg}\n"
            f"file path: {input_path}\n"
            f": check file exists, readable, format"
        )


def _load_data(input_path: str) -> pd.DataFrame:
    'load data file(supports CSV, JSON, Parquet format) auto detect file format, content load supports JSON structure(fmp_historical_data output format) Args: input_path: input file path Returns: Data Frame: load data'
    input_path = os.path.expanduser(input_path)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"file does not exist: {input_path}")
    
    file_ext = os.path.splitext(input_path)[1].lower()
    
    # supports format read
    # JSON, use load structure
    format_readers = {
        'csv': lambda: pd.read_csv(input_path),
        'json': lambda: _load_json_file_smart(input_path),
        'parquet': lambda: pd.read_parquet(input_path),
    }
    
    # prefer trying format
    ext_to_format = {
        '.csv': 'csv',
        '.json': 'json',
        '.parquet': 'parquet',
    }
    
    primary_format = ext_to_format.get(file_ext, None)
    attempted_formats = []
    format_errors = {}  # save format error
    skipped_formats = []  # recordskip format
    
    # try formatread
    if primary_format:
        try:
            _print(f"try {primary_format.upper()} formatread file...")
            df = format_readers[primary_format]()
            _print(f"OK success load data: {len(df)} {len(df.columns)}")
            return df
        except Exception as e:
            attempted_formats.append(primary_format)
            error_msg = str(e)
            format_errors[primary_format] = error_msg
            _print(f"WARN warning: {primary_format.upper()} formatread failed: {error_msg[:200]}")
            _print(f"try auto detect file format...")
    
    # read failed, try supports format
    # :.json file, try Parquet()
    for format_name, reader_func in format_readers.items():
        if format_name in attempted_formats:
            continue  # try, skip
        
        # : file.json, try Parquet
        if file_ext == '.json' and format_name == 'parquet':
            skipped_formats.append(('parquet', f'file.json, skip Parquet format'))
            _print(f"skip Parquet format(file.json,)")
            continue
        
        # , file.parquet, try JSON()
        if file_ext == '.parquet' and format_name == 'json':
            skipped_formats.append(('json', f'file.parquet, skip JSON format'))
            _print(f"skip JSON format(file.parquet)")
            continue
        
        try:
            _print(f"try {format_name.upper()} formatread...")
            df = reader_func()
            _print(f"OK success {format_name.upper()} formatload data: {len(df)} {len(df.columns)}")
            if primary_format and format_name != primary_format:
                _print(f"WARN: file {file_ext}, actualcontent {format_name.upper()} format")
            return df
        except Exception as e:
            attempted_formats.append(format_name)
            error_msg = str(e)
            format_errors[format_name] = error_msg
            _print(f"ERR {format_name.upper()} formatread failed: {error_msg[:200]}")
            continue
    
    # format failed, error
    error_parts = []
    error_parts.append(f"data load failed: read file")
    error_parts.append(f"file path: {input_path}")
    error_parts.append(f"file: {file_ext}")
    
    # actualtry format
    if attempted_formats:
        error_parts.append(f"try format: {', '.join([f.upper() for f in attempted_formats])}")
        error_parts.append(f"formaterror:")
        for fmt in attempted_formats:
            if fmt in format_errors:
                err_msg = format_errors[fmt]
                # error (keep 300)
                if len(err_msg) > 300:
                    err_msg = err_msg[:300] + "..."
                error_parts.append(f"- {fmt.upper()}: {err_msg}")
    
    # skip format
    if skipped_formats:
        error_parts.append(f"skip format:")
        for fmt, reason in skipped_formats:
            error_parts.append(f"- {fmt.upper()}: {reason}")
    
    #
    error_parts.append(f"\n:")
    if file_ext == '.json':
        error_parts.append(f"1. check JSON file format (JSON)")
        error_parts.append(f"2. check file question(UTF-8)")
        error_parts.append(f"3. trymanual file JSON format")
        error_parts.append(f"4. fileactual format, file")
    elif file_ext == '.csv':
        error_parts.append(f"1. check CSV file format (,)")
        error_parts.append(f"2. try Excel tool file")
    elif file_ext == '.parquet':
        error_parts.append(f"1. check Parquet file complete")
        error_parts.append(f"2. try to use pandas.read_parquet() direct read")
    else:
        error_parts.append(f"1. check file format supports(currentsupports: CSV, JSON, Parquet)")
        error_parts.append(f"2. file format, convert supports format")
    
    raise RuntimeError(''.join(error_parts))


def _validate_columns(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, str]:
    "column name(supports normalize) Args: df: input Data Frame required_cols: required ('open', 'high', 'low', 'close', 'volume') Returns: column name dictionary {'open': 'actualcolumn name', ...}"
    col_mapping = {}
    
    # commoncolumn name
    name_variants = {
        # description:
        # - overwritecommonsize/English, compatible data
        # - FMP historical output "Open_AAPL/Close_AAPL/Volume_AAPL" column name,
        # " + Benchmark" auto, symbol.
        'open': ['open', 'Open', 'text', 'text', 'OPEN', 'opening_price', 'open_price', 'opening'],
        'high': ['high', 'High', 'text', 'text', 'HIGH', 'highest_price', 'high_price'],
        'low': ['low', 'Low', 'text', 'text', 'LOW', 'lowest_price', 'low_price'],
        'close': ['close', 'Close', 'text', 'text', 'CLOSE', 'closing_price', 'close_price', 'last', 'last_price'],
        'volume': ['volume', 'Volume', 'text', 'VOLUME', 'vol', 'Vol', 'VOL', 'turnover', 'text'],
    }
    
    # column name: empty, (keep column name return)
    df_cols_lower = {str(col).strip().lower(): str(col) for col in df.columns}
    
    for req_col in required_cols:
        found = False
        for variant in name_variants.get(req_col, [req_col]):
            v = str(variant).strip()
            v_lower = v.lower()

            # 1) direct()
            if v in df.columns:
                col_mapping[req_col] = v
                found = True
                break
            # 2) size
            elif v_lower in df_cols_lower:
                col_mapping[req_col] = df_cols_lower[v_lower]
                found = True
                break

            # 3): (Open_AAPL/Close_AAPL/Volume_AAPL column name)
            # -:"{variant}_{xxx}"/"{variant} {xxx}"/"{variant}-{xxx}"
            # - exists *_Benchmark, prefer benchmark
            try:
                candidates = []
                for col in df.columns:
                    col_s = str(col).strip()
                    col_l = col_s.lower()
                    if col_l.startswith(v_lower + "_") or col_l.startswith(v_lower + '') or col_l.startswith(v_lower + "-"):
                        candidates.append(col_s)
                if candidates:
                    # prefer benchmark
                    non_benchmark = [c for c in candidates if "benchmark" not in c.lower()]
                    chosen_pool = non_benchmark if non_benchmark else candidates
                    # "/" column name(symbol)
                    chosen = sorted(chosen_pool, key=lambda x: (len(x), x.lower()))[0]
                    col_mapping[req_col] = chosen
                    found = True
                    break
            except Exception:
                # compatible: failed
                pass
        
        if not found:
            # metrics, optional
            if req_col in ['volume']:
                continue
            raise ValueError(
                f"missing required: {req_col}\n"
                f"available: {list(df.columns)}\n"
                f": supports column name: {name_variants.get(req_col, [req_col])}"
            )
    
    return col_mapping


def _get_indicator_params(indicator_name: str) -> Tuple[List[str], Dict[str, Any]]:
    "get metrics input parameter Args: indicator_name: metricsname('SMA', 'MACD') Returns: tuple: (required_inputs, default_params) - required_inputs: inputcolumn name list(['close'], ['high', 'low', 'close']) - default_params: defaultparameterdictionary"
    if talib is None:
        return ([], {})
    
    # get TA-Lib
    indicator_func = getattr(talib, indicator_name, None)
    if indicator_func is None:
        return ([], {})
    
    # get
    try:
        sig = inspect.signature(indicator_func)
        params = sig.parameters
        
        required_inputs = []
        default_params = {}
        
        # common input parameter
        price_params = ['open', 'high', 'low', 'close', 'volume']
        
        for param_name, param in params.items():
            # datainput
            if param_name.lower() in price_params:
                required_inputs.append(param_name.lower())
            # default, record
            elif param.default != inspect.Parameter.empty:
                default_params[param_name] = param.default
        
        return (required_inputs, default_params)
    except Exception:
        # get, return empty
        return ([], {})


def calculate_indicator(df: pd.DataFrame, indicator_name: str, 
                       col_mapping: Dict[str, str],
                       params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    "metrics calculate, supports TA-Lib metrics Args: df: input data indicator_name: metricsname('SMA', 'MACD', 'RSI') col_mapping: column name dictionary params: parameter(optional) Returns: metrics Data Frame"
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    if indicator_name not in SUPPORTED_INDICATORS:
        _print(f"WARN warning: {indicator_name} supports metricslist, skip")
        return df
    
    # get TA-Lib
    indicator_func = getattr(talib, indicator_name, None)
    if indicator_func is None:
        _print(f"WARN warning: TA-Lib not found {indicator_name}, skip")
        return df
    
    try:
        # input data
        inputs = []
        kwargs = {}
        
        # input()
        sig = inspect.signature(indicator_func)
        param_names = list(sig.parameters.keys())
        
        # input data
        for param_name in param_names:
            param_lower = param_name.lower()
            param_obj = sig.parameters[param_name]
            
            # data(position parameter)
            if param_lower in ['open', 'high', 'low', 'close', 'volume', 'real', 'real0', 'real1']:
                # 'real' type input, default use close
                if param_lower in ['real', 'real0']:
                    price_col = 'close'
                elif param_lower == 'real1':
                    price_col = 'close'  # open,
                else:
                    price_col = param_lower
                
                if price_col in col_mapping:
                    inputs.append(df[col_mapping[price_col]].values.astype(np.float64))
                else:
                    # missing required input, skip metrics
                    _print(f"WARN warning: {indicator_name} {price_col}, data does not exist, skip")
                    return df
            # optional parameters
            elif param_obj.default != inspect.Parameter.empty:
                # user parameter, useuser
                if params and param_name in params:
                    kwargs[param_name] = params[param_name]
                # use default ()
        
        # call metrics
        if kwargs:
            result = indicator_func(*inputs, **kwargs)
        else:
            result = indicator_func(*inputs)
        
        # return result
        if isinstance(result, tuple):
            # output(MACD return macd, signal, hist)
            output_names = _get_output_names(indicator_name, len(result))
            for i, output in enumerate(result):
                col_name = output_names[i] if i < len(output_names) else f"{indicator_name}_{i}"
                df[col_name] = output
        else:
            # output
            df[indicator_name] = result
        
        return df
    
    except Exception as e:
        _print(f"WARN warning: calculate {indicator_name}: {str(e)}, skip")
        return df


def _get_output_names(indicator_name: str, num_outputs: int) -> List[str]:
    'get metrics outputcolumn name Args: indicator_name: metricsname num_outputs: outputcount Returns: outputcolumn name list'
    # metrics outputname
    output_name_map = {
        'MACD': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'MACDEXT': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'MACDFIX': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'STOCH': ['STOCH_K', 'STOCH_D'],
        'STOCHF': ['STOCHF_K', 'STOCHF_D'],
        'STOCHRSI': ['STOCHRSI_K', 'STOCHRSI_D'],
        'BBANDS': ['BBANDS_Upper', 'BBANDS_Middle', 'BBANDS_Lower'],
        'AROON': ['AROON_Down', 'AROON_Up'],
        'HT_PHASOR': ['HT_PHASOR_InPhase', 'HT_PHASOR_Quadrature'],
        'HT_SINE': ['HT_SINE', 'HT_LEADSINE'],
        'MAMA': ['MAMA', 'FAMA'],
        'MINMAX': ['MIN', 'MAX'],
        'MINMAXINDEX': ['MIN_IDX', 'MAX_IDX'],
    }
    
    if indicator_name in output_name_map:
        return output_name_map[indicator_name]
    
    # default
    if num_outputs == 1:
        return [indicator_name]
    else:
        return [f"{indicator_name}_{i}" for i in range(num_outputs)]


def calculate_ma(df: pd.DataFrame, col_mapping: Dict[str, str], 
                 periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    'calculate average (MA)'
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    close = df[col_mapping['close']].values.astype(np.float64)
    
    for period in periods:
        df[f'MA_{period}'] = talib.SMA(close, timeperiod=period)
    
    _print(f"OK calculate MA: {periods}")
    return df


def calculate_ema(df: pd.DataFrame, col_mapping: Dict[str, str],
                  periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    'calculate average (EMA)'
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    close = df[col_mapping['close']].values.astype(np.float64)
    
    for period in periods:
        df[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
    
    _print(f"OK calculate EMA: {periods}")
    return df


def calculate_macd(df: pd.DataFrame, col_mapping: Dict[str, str],
                   fast_period: int = 12, slow_period: int = 26, 
                   signal_period: int = 9) -> pd.DataFrame:
    """calculate MACD metrics"""
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    close = df[col_mapping['close']].values.astype(np.float64)
    
    macd, signal, hist = talib.MACD(
        close, 
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period
    )
    
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    _print(f"OK calculate MACD ({fast_period}, {slow_period}, {signal_period})")
    return df


def calculate_rsi(df: pd.DataFrame, col_mapping: Dict[str, str],
                  periods: List[int] = [6, 12, 24]) -> pd.DataFrame:
    'calculate metrics (RSI)'
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    close = df[col_mapping['close']].values.astype(np.float64)
    
    for period in periods:
        df[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)
    
    _print(f"OK calculate RSI: {periods}")
    return df


def calculate_bollinger_bands(df: pd.DataFrame, col_mapping: Dict[str, str],
                               period: int = 20, nbdevup: float = 2.0,
                               nbdevdn: float = 2.0) -> pd.DataFrame:
    'calculate (BOLL)'
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    close = df[col_mapping['close']].values.astype(np.float64)
    
    upper, middle, lower = talib.BBANDS(
        close,
        timeperiod=period,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn
    )
    
    df['BOLL_Upper'] = upper
    df['BOLL_Middle'] = middle
    df['BOLL_Lower'] = lower
    df['BOLL_Width'] = (upper - lower) / middle  #
    
    _print(f"OK calculate (period={period}, std={nbdevup})")
    return df


def calculate_kdj(df: pd.DataFrame, col_mapping: Dict[str, str],
                  fastk_period: int = 9, slowk_period: int = 3,
                  slowd_period: int = 3) -> pd.DataFrame:
    """calculate KDJ metrics"""
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    high = df[col_mapping['high']].values.astype(np.float64)
    low = df[col_mapping['low']].values.astype(np.float64)
    close = df[col_mapping['close']].values.astype(np.float64)
    
    # calculate K D
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=0,
        slowd_period=slowd_period,
        slowd_matype=0
    )
    
    # calculate J = 3K - 2D
    j = 3 * slowk - 2 * slowd
    
    df['KDJ_K'] = slowk
    df['KDJ_D'] = slowd
    df['KDJ_J'] = j
    
    _print(f"OK calculate KDJ ({fastk_period}, {slowk_period}, {slowd_period})")
    return df


def calculate_atr(df: pd.DataFrame, col_mapping: Dict[str, str],
                  period: int = 14) -> pd.DataFrame:
    'calculateaverage (ATR)'
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    high = df[col_mapping['high']].values.astype(np.float64)
    low = df[col_mapping['low']].values.astype(np.float64)
    close = df[col_mapping['close']].values.astype(np.float64)
    
    df['ATR'] = talib.ATR(high, low, close, timeperiod=period)
    
    _print(f"OK calculate ATR (period={period})")
    return df


def calculate_obv(df: pd.DataFrame, col_mapping: Dict[str, str]) -> pd.DataFrame:
    'calculate (OBV)'
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    close = df[col_mapping['close']].values.astype(np.float64)
    volume = df[col_mapping['volume']].values.astype(np.float64)
    
    df['OBV'] = talib.OBV(close, volume)
    
    _print(f"OK calculate OBV")
    return df


def calculate_adx(df: pd.DataFrame, col_mapping: Dict[str, str],
                  period: int = 14) -> pd.DataFrame:
    'calculateaverage metrics (ADX)'
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    high = df[col_mapping['high']].values.astype(np.float64)
    low = df[col_mapping['low']].values.astype(np.float64)
    close = df[col_mapping['close']].values.astype(np.float64)
    
    df['ADX'] = talib.ADX(high, low, close, timeperiod=period)
    
    _print(f"OK calculate ADX (period={period})")
    return df


def calculate_cci(df: pd.DataFrame, col_mapping: Dict[str, str],
                  period: int = 14) -> pd.DataFrame:
    'calculate metrics (CCI)'
    if talib is None:
        raise ImportError("TA-Lib not installed")
    
    high = df[col_mapping['high']].values.astype(np.float64)
    low = df[col_mapping['low']].values.astype(np.float64)
    close = df[col_mapping['close']].values.astype(np.float64)
    
    df['CCI'] = talib.CCI(high, low, close, timeperiod=period)
    
    _print(f"OK calculate CCI (period={period})")
    return df


def calculate_all_indicators(df: pd.DataFrame, col_mapping: Dict[str, str],
                             config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    'calculate technical indicator Args: df: input data col_mapping: column name config: config(optional) Returns: technical indicator Data Frame'
    if config is None:
        config = {}
    
    # MA
    if config.get('ma', True):
        df = calculate_ma(df, col_mapping, periods=config.get('ma_periods', [5, 10, 20, 60]))
    
    # EMA
    if config.get('ema', True):
        df = calculate_ema(df, col_mapping, periods=config.get('ema_periods', [5, 10, 20, 60]))
    
    # MACD
    if config.get('macd', True):
        df = calculate_macd(df, col_mapping)
    
    # RSI
    if config.get('rsi', True):
        df = calculate_rsi(df, col_mapping, periods=config.get('rsi_periods', [6, 12, 24]))
    
    #
    if config.get('boll', True):
        df = calculate_bollinger_bands(df, col_mapping)
    
    # KDJ
    if config.get('kdj', True):
        df = calculate_kdj(df, col_mapping)
    
    # ATR
    if config.get('atr', True):
        df = calculate_atr(df, col_mapping)
    
    # OBV
    if config.get('obv', True) and 'volume' in col_mapping:
        df = calculate_obv(df, col_mapping)
    
    # ADX
    if config.get('adx', True):
        df = calculate_adx(df, col_mapping)
    
    # CCI
    if config.get('cci', True):
        df = calculate_cci(df, col_mapping)
    
    return df


def run(args) -> None:
    'MCP tool Args: args: parameter - input_path: input data file path - output_path: output file path(optional) - indicators: calculate metricslist(optional, default 150 metrics) - config: config(optional) - validate: return data(optional, default False)'
    if talib is None:
        raise ImportError(
            'TA-Lib not installed, install:'
            'Ubuntu/Debian: sudo apt-get install ta-lib && pip install TA-Lib'
            'Mac: brew install ta-lib && pip install TA-Lib'
            'Windows: install current Python TA-Lib wheel'
        )
    
    # get parameter
    input_path = getattr(args, 'input_path', None)
    if not input_path:
        raise ValueError('missing required parameters: input_path')
    
    output_path = getattr(args, 'output_path', None)
    indicators = getattr(args, 'indicators', None)
    config_dict = getattr(args, 'config', {})
    validate = getattr(args, 'validate', False)
    
    # user metrics, use supports metrics
    if indicators is None or (isinstance(indicators, list) and 'all' in indicators):
        indicators = SUPPORTED_INDICATORS.copy()
        _print(f"calculate supports metrics ({len(indicators)})")
    elif isinstance(indicators, str):
        indicators = [indicators]
    
    _print("="*60)
    _print(f"technical indicator calculation tool - based on TA-Lib")
    _print("="*60)
    _print(f"supports metrics: {len(SUPPORTED_INDICATORS)}")
    _print(f"calculate metrics: {len(indicators)}")
    
    # load data
    _print(f"\n 1: load data file")
    _print(f"file path: {input_path}")
    df_original = _load_data(input_path)
    df = df_original.copy()
    _print(f"data: {list(df.columns)}")
    
    # convert date, auto convert question
    _print(f": convert date format...")
    df = _convert_timestamps_to_strings(df)
    df_original = df.copy()  # updateoriginal data
    
    # column name(try)
    _print(f"\n 2: data")
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    col_mapping = {}
    for col in required_cols:
        try:
            mapping = _validate_columns(df, [col])
            col_mapping.update(mapping)
        except ValueError:
            # does not exist, skip
            pass
    
    _print(f"column name: {col_mapping}")
    _print(f"available: {list(col_mapping.keys())}")
    
    # calculate metrics
    _print(f"\n 3: calculatetechnical indicator")
    _print(f"metricslist: {indicators if len(indicators) <= 10 else f'{indicators[:10]}... ({len(indicators)})'}")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, indicator_name in enumerate(indicators, 1):
        try:
            original_cols = len(df.columns)
            df = calculate_indicator(df, indicator_name, col_mapping, config_dict)
            new_cols = len(df.columns)
            
            if new_cols > original_cols:
                success_count += 1
                if (i % 20 == 0) or (i == len(indicators)):
                    _print(f": {i}/{len(indicators)} | success: {success_count} | failed: {failed_count} | skip: {skipped_count}")
            else:
                skipped_count += 1
        except Exception as e:
            failed_count += 1
            _print(f"ERR {indicator_name} calculate failed: {str(e)}")
    
    _print(f"\nmetricscalculatestatistics:")
    _print(f"OK success: {success_count}")
    _print(f"skip: {skipped_count}")
    _print(f"ERR failed: {failed_count}")
    
    # save, Timestamp convert(calculate metrics)
    _print(f"\n 3.5: finaldataclean(Timestamp)")
    df = _convert_timestamps_to_strings(df)
    
    # : use type, check datetime
    timestamp_count = 0
    problematic_cols = []
    
    # check
    if len(df.index) > 0:
        sample_idx = df.index[0]
        if _is_datetime_like(sample_idx):
            timestamp_count += 1
            problematic_cols.append("text")
    
    # check
    for col in df.columns:
        col_dtype = df[col].dtype
        
        # check datetime type
        if pd.api.types.is_datetime64_any_dtype(col_dtype) or pd.api.types.is_period_dtype(col_dtype):
            timestamp_count += 1
            problematic_cols.append(col)
        elif col_dtype == 'object':
            # use type, check object
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                # check (20)
                sample_size = min(20, len(non_null_values))
                for val in non_null_values.head(sample_size):
                    if _is_datetime_like(val):
                        timestamp_count += 1
                        problematic_cols.append(col)
                        break
                    # check structure
                    elif isinstance(val, (list, tuple)):
                        for item in val[:5]:
                            if _is_datetime_like(item):
                                timestamp_count += 1
                                problematic_cols.append(col)
                                break
                        if col in problematic_cols:
                            break
                    elif isinstance(val, dict):
                        for item in list(val.values())[:5]:
                            if _is_datetime_like(item):
                                timestamp_count += 1
                                problematic_cols.append(col)
                                break
                        if col in problematic_cols:
                            break
    
    if timestamp_count > 0:
        _print(f"WARN warning: {timestamp_count} datetime type/: {problematic_cols[:10]}")
        _print(f"clean...")
        df = _convert_timestamps_to_strings(df)  # clean
        
        # : question, clean
        timestamp_count_2 = 0
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                timestamp_count_2 += 1
            elif df[col].dtype == 'object' and len(df[col].dropna()) > 0:
                sample = df[col].dropna().head(10)
                for val in sample:
                    if _is_datetime_like(val):
                        timestamp_count_2 += 1
                        break
        
        if timestamp_count_2 > 0:
            _print(f"WARN warning: clean {timestamp_count_2} question, clean...")
            df = _convert_timestamps_to_strings(df)  # clean
    
    _print(f"OK dataclean complete, datetime convert")
    
    # save result
    _print(f"\n 4: save result")
    if output_path:
        output_path = os.path.expanduser(output_path)
    else:
        # output path, auto generate
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path) or '.'
        output_path = os.path.join(output_dir, f"{input_basename}_indicators.csv")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    file_ext = os.path.splitext(output_path)[1].lower()
    
    # JSON format, convert Timestamp
    if file_ext == '.json':
        # Timestamp convert()
        df_json = _convert_timestamps_to_strings(df.copy())
        
        # check: save serialize data(3)
        _print(f"check: serialize 3 data...")
        try:
            test_df = df_json.head(3).copy()
            # tryserialize data
            test_records = test_df.to_dict('records')
            test_records = _convert_dict_timestamps_to_strings(test_records)
            # use serialize
            test_json_str = json.dumps(test_records, indent=2, ensure_ascii=False, cls=TimestampJSONEncoder)
            #
            json.loads(test_json_str)
            _print(f"OK check: 3 data successserialize")
        except Exception as precheck_err:
            _print(f"WARN checkwarning: {str(precheck_err)[:150]}")
            # check failed, clean
            df_json = _convert_timestamps_to_strings(df_json)
        
        # use save JSON,
        try:
            # 1: try directly use to_json(convert success,)
            json_str = df_json.to_json(orient='records', indent=2, force_ascii=False, date_format='iso')
            # JSON ()
            try:
                json.loads(json_str)
                # , write file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                _print(f"OK use to_json success save")
            except json.JSONDecodeError as parse_err:
                # JSON invalid, try 2
                raise ValueError(f"generate JSON invalid: {parse_err}")
        except (TypeError, ValueError, AttributeError) as e:
            # 2: to_json failed, use
            _print(f"WARN to_json failed, use: {str(e)[:150]}")
            try:
                # convert dictionary, use
                records = df_json.to_dict('records')
                # clean Timestamp()
                records = _convert_dict_timestamps_to_strings(records)
                # use save
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=2, ensure_ascii=False, cls=TimestampJSONEncoder)
                _print(f"OK use success save")
            except Exception as e2:
                # 3: - convert
                _print(f"WARN failed, use convert: {str(e2)[:150]}")
                try:
                    # , clean
                    clean_records = []
                    for idx, row in df_json.iterrows():
                        row_dict = row.to_dict()
                        # clean
                        clean_row = _convert_dict_timestamps_to_strings(row_dict)
                        clean_records.append(clean_row)
                    # use save
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(clean_records, f, indent=2, ensure_ascii=False, cls=TimestampJSONEncoder)
                    _print(f"OK use convertsuccess save")
                except Exception as e3:
                    # failed
                    error_msg = (
                        f"save JSON file, serialize failed\n"
                        f"file path: {output_path}\n"
                        f"data: {len(df_json)}\n"
                        f"data: {len(df_json.columns)}\n"
                        f"column name: {list(df_json.columns)}\n"
                        f"error1 (to_json): {str(e)[:200]}\n"
                        f"error2 (): {str(e2)[:200]}\n"
                        f"error3 (convert): {str(e3)[:200]}\n"
                        f": check data serialize"
                    )
                    _print(f"ERR {error_msg}")
                    raise RuntimeError(error_msg)
    elif file_ext == '.csv' or not file_ext:
        df.to_csv(output_path, index=False)
    elif file_ext == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    file_size = os.path.getsize(output_path) / 1024  # KB
    _print(f"OK result save: {output_path}")
    _print(f"filesize: {file_size:.1f} KB")
    
    # generate data
    if validate:
        _print(f"\n 5: generate data")
        # use convert Data Frame(to_dict Timestamp)
        # convert, 3, date
        df_for_preview = _convert_timestamps_to_strings(df.head(3).copy())
        
        # use: convert JSON, dictionary
        # to_dict Timestamp
        preview_dict = None
        preview_error = None
        
        # 1: try to use to_json()
        try:
            json_str = df_for_preview.to_json(orient='records', date_format='iso')
            preview_dict = json.loads(json_str)  #
            _print(f"OK use to_json generate data")
        except Exception as e1:
            preview_error = str(e1)
            _print(f"WARN to_json failed: {str(e1)[:150]}")
            # 2: use to_dict + clean
            try:
                preview_dict = df_for_preview.to_dict('records')
                # clean Timestamp
                preview_dict = _convert_dict_timestamps_to_strings(preview_dict)
                _print(f"OK use to_dict + clean generate data")
            except Exception as e2:
                preview_error = f"{str(e1)}; {str(e2)}"
                _print(f"WARN to_dict failed: {str(e2)[:150]}")
                # 3:
                try:
                    preview_dict = []
                    for idx, row in df_for_preview.iterrows():
                        row_dict = row.to_dict()
                        clean_row = _convert_dict_timestamps_to_strings(row_dict)
                        preview_dict.append(clean_row)
                    _print(f"OK use generate data")
                except Exception as e3:
                    preview_error = f"{str(e1)}; {str(e2)}; {str(e3)}"
                    _print(f"ERR data generate failed")
                    # failed, useempty list
                    preview_dict = []
        
        validation_data = {
            'input_file': input_path,
            'output_file': output_path,
            'total_rows': len(df),
            'original_columns': len(df_original.columns),
            'total_columns': len(df.columns),
            'new_indicators': len(df.columns) - len(df_original.columns),
            'requested_indicators': len(indicators),
            'success_count': success_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
            'new_column_names': [col for col in df.columns if col not in df_original.columns],
            'data_preview': preview_dict
        }
        
        # save data(Timestamp convert, use)
        validation_data = _convert_dict_timestamps_to_strings(validation_data)
        validation_file = output_path.replace(file_ext, '_validation.json')
        
        try:
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_data, f, indent=2, ensure_ascii=False, cls=TimestampJSONEncoder)
            _print(f"OK data save: {validation_file}")
        except Exception as e:
            # save failed, try use (, clean)
            _print(f"WARN use save failed, try: {str(e)[:150]}")
            try:
                # clean
                validation_data = _convert_dict_timestamps_to_strings(validation_data)
                with open(validation_file, 'w', encoding='utf-8') as f:
                    json.dump(validation_data, f, indent=2, ensure_ascii=False)
                _print(f"OK data save(use): {validation_file}")
            except Exception as e2:
                _print(f"ERR data save failed: {str(e2)[:200]}")
                # exception, output file success save
                _print(f"warning: data save failed, output file success save")
        
        # summary
        _print(f"\n data:")
        _print(f"original: {validation_data['original_columns']}")
        _print(f"metrics: {validation_data['new_indicators']}")
        _print(f": {validation_data['total_columns']}")
        _print(f"data: {validation_data['total_rows']}")
    
    _print(f"\n" + "="*60)
    _print(f"OK technical indicator calculation complete!")
    _print(f"input file: {input_path}")
    _print(f"output file: {output_path}")
    _print(f"data: {len(df)}")
    _print(f"original: {len(df_original.columns)}")
    _print(f": {len(df.columns) - len(df_original.columns)}")
    _print(f": {len(df.columns)}")
    _print("="*60)
