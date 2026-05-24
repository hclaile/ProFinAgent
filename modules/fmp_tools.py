import os
import sys
import json
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Dict, Any, List, Union, Tuple

try:
    import requests
except ImportError:
    requests = None

# optional dependency: financetoolkit (tool)
try:
    from financetoolkit import Toolkit  # type: ignore
    _FINANCETOOLKIT_AVAILABLE = True
except ImportError:
    Toolkit = None  # type: ignore
    _FINANCETOOLKIT_AVAILABLE = False


FMP_BASE_URL = 'https://financialmodelingprep.com/stable'


def _print(*args, **kwargs):
    'print to stderr, avoid interfering with MCP JSON-RPC communication'
    print(*args, file=sys.stderr, **kwargs)


def _ensure_output_path(path: str) -> str:
    if not path or not isinstance(path, str):
        raise ValueError('output_path, empty')
    path = os.path.expanduser(path)
    out_dir = os.path.dirname(path)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    return path


def _get_root_dir(args) -> str:
    root = getattr(args, "root", None)
    if root:
        return os.path.expanduser(root)
    return os.getcwd()


def _get_api_key(args) -> str:
    api_key = getattr(args, "api_key", None) or os.environ.get("FMP_API_KEY")
    if not api_key:
        raise ValueError('missing FMP api_key, parameter environment variable FMP_API_KEY')
    return api_key


def _http_get(url: str, params: Dict[str, Any]) -> Any:
    if requests is None:
        raise ImportError('requests not installed, please run: pip install requests')
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _search_symbol(api_key: str, company_name: str, limit: int = 5) -> Optional[str]:
    'use search-name endpoint, result try direct return original input.'
    try:
        url = f"{FMP_BASE_URL}/search-name"
        params = {"query": company_name, "limit": limit, "apikey": api_key}
        data = _http_get(url, params)
        if isinstance(data, list) and data:
            first = data[0]
            symbol = first.get("symbol")
            if symbol:
                return symbol
    except Exception as e:
        _print(f"[WARN] search-name query failed, try directly use input: {e}")
    return company_name.upper().strip()


def _fetch_profile(api_key: str, symbol: str) -> List[Dict[str, Any]]:
    url = f"{FMP_BASE_URL}/pro file"
    params = {"symbol": symbol, "apikey": api_key}
    data = _http_get(url, params)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def run_company_profile(args) -> None:
    'get company name FMP Company Pro file required parameters: company_name: company name code output_path: output JSON file path optional parameters: api_key: FMP API Key, use environment variable FMP_API_KEY limit: search return maximum candidates (default 5) root: root directory(use, interface)'
    company_name = getattr(args, "company_name", None)
    output_path = getattr(args, "output_path", None)
    limit = getattr(args, "limit", 5)

    if not company_name:
        raise ValueError('company_name parameter')
    if not output_path:
        raise ValueError('output_path parameter')

    output_path = _ensure_output_path(output_path)
    api_key = _get_api_key(args)

    _print(f"[INFO] start query FMP company: {company_name}")
    symbol = _search_symbol(api_key, company_name, limit=limit)
    _print(f"[INFO]: {symbol}")

    profile = _fetch_profile(api_key, symbol)
    if not profile:
        _print('[WARN] get company Pro file data')

    result = {
        "query": company_name,
        "resolved_symbol": symbol,
        "profile": profile,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    _print(f"[SUCCESS] result save: {output_path}")


def _fetch_statement(api_key: str, symbol: str, endpoint: str, limit: int = 40, period: Optional[str] = None) -> Any:
    'get: income-statement/balance-sheet-statement/cash-flow-statement'
    def _normalize_symbol(sym: str) -> str:
        return (sym or "").strip().upper()

    def _normalize_period(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        v = str(p).strip().lower()
        # compatiblecommon: annual/quarter
        if v in {"a", "annual", "year", "yearly", "y"}:
            return "annual"
        if v in {"q", "quarter", "quarterly"}:
            return "quarter"
        # , future FMP user; continue
        return v

    def _looks_like_error_payload(data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        # FMP common error field/(endpoint)
        for k in ("Error Message", "error", "errors", "message"):
            if k in data and data.get(k):
                return True
        return False

    symbol_n = _normalize_symbol(symbol)
    period_n = _normalize_period(period)

    # FMP stable query parameter (profile symbol=...)
    # prefer trying:/{endpoint}symbol=XXX&...
    url_query = f"{FMP_BASE_URL}/{endpoint}"
    params_query: Dict[str, Any] = {"symbol": symbol_n, "limit": limit, "apikey": api_key}
    if period_n:
        params_query["period"] = period_n

    last_err: Optional[Exception] = None
    try:
        data = _http_get(url_query, params_query)
        if _looks_like_error_payload(data) or (isinstance(data, list) and len(data) == 0):
            raise ValueError(f"query return error empty result: endpoint={endpoint}, symbol={symbol_n}")
        return data
    except Exception as e:
        last_err = e
        _print(f"[WARN] query failed, try path: {e}")

    # compatible:/{endpoint}/{symbol}...
    url_path = f"{FMP_BASE_URL}/{endpoint}/{symbol_n}"
    params_path: Dict[str, Any] = {"limit": limit, "apikey": api_key}
    if period_n:
        params_path["period"] = period_n
    try:
        data = _http_get(url_path, params_path)
        if _looks_like_error_payload(data) or (isinstance(data, list) and len(data) == 0):
            raise ValueError(f"path return error empty result: endpoint={endpoint}, symbol={symbol_n}")
        return data
    except Exception as e2:
        e1_msg = str(last_err) if last_err else "N/A"
        raise RuntimeError(
            f"FMP statement fetch failed: endpoint={endpoint}, symbol={symbol_n},"
            f"period={period_n}, limit={limit}, query_error={e1_msg}, path_error={e2}"
        ) from e2


def _save_statement(symbol: str, data: Any, output_path: str, kind: str):
    result = {
        "symbol": symbol,
        "statement_type": kind,
        "data": data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _print(f"[SUCCESS] {kind} save: {output_path}")


def _parse_symbols(symbols: Union[str, List[str], None]) -> List[str]:
    'supports: - "AAPL"/"AAPL, MSFT"/"AAPL MSFT" - ["AAPL", "MSFT"]'
    if symbols is None:
        return []
    if isinstance(symbols, list):
        return [str(s).strip().upper() for s in symbols if str(s).strip()]
    s = str(symbols).strip()
    if not s:
        return []
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = s.split()
    return [p.upper() for p in parts if p]


def _safe_df_to_records(obj: Any, limit: Optional[int] = None) -> Any:
    'Toolkit return JSON serializestructure: - pandas. Data Frame -> list[dict] - -> return(json. dump) Period, convert supports JSON serialize.'
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore

    if pd is not None and isinstance(obj, pd.DataFrame):
        df = obj.copy()
        # Multi Index columns
        def _flatten_col(c: Any) -> str:
            if isinstance(c, tuple):
                return "_".join([str(x) for x in c if x not in (None, "", "nan")])
            return str(c)
        df.columns = [_flatten_col(c) for c in df.columns]
        
        # Period: convert
        if hasattr(df.index, 'dtype'):
            if pd.api.types.is_period_dtype(df.index.dtype):
                df.index = df.index.astype(str)
            elif isinstance(df.index, pd.PeriodIndex):
                df.index = df.index.astype(str)
        
        #
        if isinstance(limit, int) and limit > 0:
            try:
                df = df.sort_index(ascending=False).head(limit).sort_index()
            except Exception:
                df = df.head(limit)
        
        # (Period normal)
        df = df.reset_index()
        
        # Period
        for col in df.columns:
            if pd.api.types.is_period_dtype(df[col].dtype):
                df[col] = df[col].astype(str)
            else:
                # check Period (type)
                try:
                    # try convert Period
                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, pd.Period) else x)
                except Exception:
                    pass  # convert failed,
        
        # convert dictionary list
        records = df.to_dict(orient="records")
        
        # check: record, Period
        def _convert_period_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            'convert dictionary Period'
            result = {}
            for k, v in d.items():
                if isinstance(v, pd.Period):
                    result[k] = str(v)
                elif isinstance(v, dict):
                    result[k] = _convert_period_in_dict(v)
                elif isinstance(v, list):
                    result[k] = [_convert_period_in_dict(item) if isinstance(item, dict) else (str(item) if isinstance(item, pd.Period) else item) for item in v]
                else:
                    result[k] = v
            return result
        
        records = [_convert_period_in_dict(r) for r in records]
        return records
    return obj


def _toolkit_call(
    api_key: str,
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    'initialize Toolkit(start_date/end_date). Toolkit parameter, compatible.'
    if not _FINANCETOOLKIT_AVAILABLE or Toolkit is None:
        raise ImportError('financetoolkit not installed. install: pip install financetoolkit')
    if not symbols:
        raise ValueError('symbols is empty')

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        # supports end_date
        kwargs["end_date"] = end_date
    try:
        return Toolkit(symbols, **kwargs)
    except TypeError:
        # compatibleunsupported end_date
        kwargs.pop("end_date", None)
        return Toolkit(symbols, **kwargs)


def _save_json(output_path: str, payload: Any) -> None:
    output_path = _ensure_output_path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _print(f"[SUCCESS] result save: {output_path}")


def _looks_like_cn_stock(identifier: str) -> bool:
    'A: - Chinese(company common) - 6 - SH/SZ + 6'
    if not identifier:
        return False
    s = str(identifier).strip().upper()
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    if s.isdigit() and len(s) <= 6:
        return True
    if len(s) == 8 and s[:2] in ("SH", "SZ") and s[2:].isdigit():
        return True
    return False


def _ensure_date_range_for_market_data(start_date: Optional[str], end_date: Optional[str]) -> Dict[str, str]:
    'crawler.run_yfinance/run_stock_data_fetcher start_date/end_date. stable default range, path direct failed.'
    if not start_date:
        start_date = "2020-01-01"
    if not end_date:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
    return {"start_date": start_date, "end_date": end_date}


def _shorten_filename(file_path: str, max_length: int = 200) -> str:
    'file (255).: 1. clean `.tmp_` (call) 2. file (path) max_length, 3. keep 4. path, Args: file_path: complete file path max_length: file maximum (default 200, path) Returns: file path'
    if not file_path:
        return file_path
    
    # directory file
    dir_path = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    # file
    name_without_ext, ext = os.path.splitext(filename)
    
    # clean `.tmp_` (call)
    # :`brk_b_ohlcv.tmp_BRK. B.tmp_BRK. B...` -> `brk_b_ohlcv.tmp_BRK. B`
    import re
    # `.tmp_` mode
    pattern = r'\.tmp_([^\.]+)(\.tmp_\1)+'
    name_without_ext = re.sub(pattern, r'.tmp_\1', name_without_ext)
    
    # file
    filename = name_without_ext + ext
    
    # check file
    if len(filename) <= max_length:
        new_path = os.path.join(dir_path, filename) if dir_path else filename
        # clean path, continue
        if len(new_path) <= 250:
            return new_path
    
    # file, name (keep)
    # calculate available name (empty)
    available_length = max_length - len(ext) - 20  # 20
    if available_length < 10:
        available_length = 10  # keep 10
    
    # name (prefer keeping,)
    truncated_name = name_without_ext[:available_length]
    
    #
    new_filename = truncated_name + ext
    new_path = os.path.join(dir_path, new_filename) if dir_path else new_filename
    
    # path, directory
    if len(new_path) > 250:  # 255
        # keep directory
        parts = new_path.split(os.sep)
        filename_part = parts[-1]
        # keep 2 directory
        if len(parts) > 3:
            parts = parts[-3:]
            parts[-1] = filename_part
            new_path = os.sep.join(parts)
    
    _print(f"[WARN] file,: {file_path[:100]}... -> {new_path[:100]}...")
    return new_path


def _load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_json_file_completeness(file_path: str) -> Tuple[bool, Optional[str]]:
    'JSON file complete Args: file_path: JSON file path Returns: tuple: (is_valid, error_message) - is_valid: True file complete, False file complete invalid - error_message: invalid, return error;, return None'
    if not os.path.exists(file_path):
        return False, f"file does not exist: {file_path}"
    
    # check file size(empty file file complete)
    file_size = os.path.getsize(file_path)
    if file_size < 10:  # 10, complete
        return False, f"file ({file_size}), complete"
    
    try:
        # try to read JSON
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # check file } (complete check)
        content_stripped = content.strip()
        if not content_stripped.endswith('}'):
            return False, 'JSON file (missing)'
        
        # try JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return False, f"JSON failed: {str(e)}"
        
        # data structure(fmp_historical_data format)
        if not isinstance(data, dict):
            return False, 'JSON is notdictionary type'
        
        # check required field
        if "data" not in data:
            return False, "missing required field 'data'"
        
        # data field
        data_field = data.get("data")
        if data_field is None:
            return False, 'data field None'
        
        # data list, check is empty data
        if isinstance(data_field, list):
            if len(data_field) == 0:
                return False, "data fieldis empty list"
            # check list complete(field)
            if len(data_field) > 0:
                first_item = data_field[0]
                if not isinstance(first_item, dict):
                    return False, 'data list is notdictionary type'
                # check date related field()
                has_valid_field = any(
                    key.lower() in ['date', 'open', 'high', 'low', 'close', 'volume', 'close', 'price']
                    for key in first_item.keys()
                )
                if not has_valid_field:
                    return False, 'data list missing field(date field)'
        
        # data dictionary(symbol), check is empty
        elif isinstance(data_field, dict):
            if len(data_field) == 0:
                return False, "data fieldis emptydictionary"
        else:
            return False, f"data field type: {type(data_field)}"
        
        # check
        return True, None
        
    except IOError as e:
        return False, f"file read failed: {str(e)}"
    except Exception as e:
        return False, f": {str(e)}"


def run_historical_data(args) -> None:
    'FMP history market data(OHLCV): Toolkit.get_historical_data() required: symbol(s), output_path optional: api_key, start_date, end_date, limit'
    symbols = _parse_symbols(getattr(args, "symbol", None) or getattr(args, "symbols", None))
    output_path = getattr(args, "output_path", None)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    limit = getattr(args, "limit", None)
    if not symbols:
        raise ValueError('symbol symbols')
    if not output_path:
        raise ValueError('output_path')
    api_key = _get_api_key(args)

    # 1) path: prefer Toolkit(FMP)
    primary_error: Optional[str] = None
    try:
        companies = _toolkit_call(api_key, symbols, start_date=start_date, end_date=end_date)
        getter = getattr(companies, "get_historical_data", None)
        if getter is None:
            raise AttributeError("Toolkit unsupported get_historical_data")

        # compatible parameter
        try:
            df = getter()
        except TypeError:
            df = getter(start_date=start_date, end_date=end_date)

        data_primary = _safe_df_to_records(df, limit=limit)
        # return empty(empty file)
        if data_primary is None or (isinstance(data_primary, list) and len(data_primary) == 0):
            raise ValueError('Toolkit.get_historical_data return empty data,')

        payload = {
            "symbols": symbols,
            "tool": "fmp_historical_data",
            "fallback_used": False,
            "data": data_primary,
        }
        
        # save check output_path()
        try:
            output_path = _shorten_filename(output_path)
        except Exception as e:
            _print(f"[WARN] file: {e}, continueuse path")
        
        _save_json(output_path, payload)
        
        # save JSON file complete
        is_valid, validation_error = _validate_json_file_completeness(output_path)
        if not is_valid:
            _print(f"[WARN] save JSON file complete: {validation_error}")
            # delete complete file
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            # directexception, use crawler
            raise ValueError(f"JSON file failed: {validation_error}")
        
        _print(f"[SUCCESS] JSON file, data complete")
        return
    except Exception as e:
        # path failed, directexception, try crawler
        _print(f"[ERROR] fmp_historical_data path failed: {e}")
        raise


def run_treasury_data(args) -> None:
    'FMP return rate(Treasury): Toolkit.get_treasury_data() required: output_path optional: api_key, start_date, end_date, limit'
    output_path = getattr(args, "output_path", None)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    limit = getattr(args, "limit", None)
    if not output_path:
        raise ValueError('output_path')
    api_key = _get_api_key(args)

    # Toolkit symbols, placeholder; actual treasury data, dependency symbol
    companies = _toolkit_call(api_key, ["AAPL"], start_date=start_date, end_date=end_date)
    getter = getattr(companies, "get_treasury_data", None)
    if getter is None:
        raise AttributeError("Toolkit unsupported get_treasury_data")

    try:
        df = getter()
    except TypeError:
        df = getter(start_date=start_date, end_date=end_date)
    payload = {
        "tool": "fmp_treasury_data",
        "data": _safe_df_to_records(df, limit=limit),
    }
    _save_json(output_path, payload)


def _run_ratios(args, kind: str) -> None:
    'ratios: kind in {profitability, liquidity, solvency, valuation}'
    symbols = _parse_symbols(getattr(args, "symbol", None) or getattr(args, "symbols", None))
    output_path = getattr(args, "output_path", None)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    limit = getattr(args, "limit", None)
    if not symbols:
        raise ValueError('symbol symbols')
    if not output_path:
        raise ValueError('output_path')
    api_key = _get_api_key(args)

    companies = _toolkit_call(api_key, symbols, start_date=start_date, end_date=end_date)
    ratios_obj = getattr(companies, "ratios", None)
    if ratios_obj is None:
        raise AttributeError("Toolkit unsupported ratios module")

    # compatible collect_* get_*
    func = getattr(ratios_obj, f"collect_{kind}_ratios", None) or getattr(ratios_obj, f"get_{kind}_ratios", None)
    if func is None:
        raise AttributeError(f"Toolkit unsupported {kind} ratios")

    df = func()
    payload = {
        "symbols": symbols,
        "tool": f"fmp_{kind}_ratios",
        "data": _safe_df_to_records(df, limit=limit),
    }
    _save_json(output_path, payload)


def run_profitability_ratios(args) -> None:
    'metrics://ROE/ROA'
    _run_ratios(args, "profitability")


def run_liquidity_ratios(args) -> None:
    'metrics:/'
    _run_ratios(args, "liquidity")


def run_solvency_ratios(args) -> None:
    'metrics:/'
    _run_ratios(args, "solvency")


def run_valuation_ratios(args) -> None:
    'metrics: PE/PS/EV/EBITDA (dependency history +)'
    _run_ratios(args, "valuation")


def run_altman_z_score(args) -> None:
    'Altman Z-Score model'
    symbols = _parse_symbols(getattr(args, "symbol", None) or getattr(args, "symbols", None))
    output_path = getattr(args, "output_path", None)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    if not symbols:
        raise ValueError('symbol symbols')
    if not output_path:
        raise ValueError('output_path')
    api_key = _get_api_key(args)
    companies = _toolkit_call(api_key, symbols, start_date=start_date, end_date=end_date)
    models = getattr(companies, "models", None)
    if models is None:
        raise AttributeError("Toolkit unsupported models module")
    func = getattr(models, "get_altman_z_score", None)
    if func is None:
        raise AttributeError("Toolkit unsupported get_altman_z_score")
    df = func()
    payload = {"symbols": symbols, "tool": "fmp_altman_z_score", "data": _safe_df_to_records(df)}
    _save_json(output_path, payload)


def run_piotroski_score(args) -> None:
    'Piotroski F-Score(9 metrics)'
    symbols = _parse_symbols(getattr(args, "symbol", None) or getattr(args, "symbols", None))
    output_path = getattr(args, "output_path", None)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    if not symbols:
        raise ValueError('symbol symbols')
    if not output_path:
        raise ValueError('output_path')
    api_key = _get_api_key(args)
    companies = _toolkit_call(api_key, symbols, start_date=start_date, end_date=end_date)
    models = getattr(companies, "models", None)
    if models is None:
        raise AttributeError("Toolkit unsupported models module")
    func = getattr(models, "get_piotroski_score", None)
    if func is None:
        raise AttributeError("Toolkit unsupported get_piotroski_score")
    df = func()
    payload = {"symbols": symbols, "tool": "fmp_piotroski_score", "data": _safe_df_to_records(df)}
    _save_json(output_path, payload)


def run_wacc(args) -> None:
    'WACC average ()'
    symbols = _parse_symbols(getattr(args, "symbol", None) or getattr(args, "symbols", None))
    output_path = getattr(args, "output_path", None)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    if not symbols:
        raise ValueError('symbol symbols')
    if not output_path:
        raise ValueError('output_path')
    api_key = _get_api_key(args)
    companies = _toolkit_call(api_key, symbols, start_date=start_date, end_date=end_date)
    models = getattr(companies, "models", None)
    if models is None:
        raise AttributeError("Toolkit unsupported models module")
    func = getattr(models, "get_weighted_average_cost_of_capital", None)
    if func is None:
        raise AttributeError("Toolkit unsupported get_weighted_average_cost_of_capital")
    # compatible parameter:risk_free_rate_source, default parameter
    df = func()
    payload = {"symbols": symbols, "tool": "fmp_wacc", "data": _safe_df_to_records(df)}
    _save_json(output_path, payload)


def run_value_at_risk(args) -> None:
    'VaR (based onhistory)'
    symbols = _parse_symbols(getattr(args, "symbol", None) or getattr(args, "symbols", None))
    output_path = getattr(args, "output_path", None)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    confidence_level = getattr(args, "confidence_level", 0.95)
    if not symbols:
        raise ValueError('symbol symbols')
    if not output_path:
        raise ValueError('output_path')
    api_key = _get_api_key(args)
    companies = _toolkit_call(api_key, symbols, start_date=start_date, end_date=end_date)
    risk = getattr(companies, "risk", None)
    if risk is None:
        raise AttributeError("Toolkit unsupported risk module")
    func = getattr(risk, "get_value_at_risk", None)
    if func is None:
        raise AttributeError("Toolkit unsupported get_value_at_risk")
    try:
        df = func(confidence_level=confidence_level)
    except TypeError:
        df = func()
    payload = {
        "symbols": symbols,
        "tool": "fmp_value_at_risk",
        "confidence_level": confidence_level,
        "data": _safe_df_to_records(df),
    }
    _save_json(output_path, payload)


def run_income_statement(args) -> None:
    'get (income-statement) required: symbol, output_path optional: api_key, limit(default40), period(annual/quarter)'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    limit = getattr(args, "limit", 40)
    period = getattr(args, "period", None)

    if not symbol:
        raise ValueError('symbol parameter')
    if not output_path:
        raise ValueError('output_path parameter')

    output_path = _ensure_output_path(output_path)
    api_key = _get_api_key(args)

    _print(f"[INFO] get: {symbol}, period={period}, limit={limit}")
    data = _fetch_statement(api_key, symbol, "income-statement", limit=limit, period=period)
    _save_statement(symbol, data, output_path, "income_statement")


def run_balance_sheet(args) -> None:
    'get (balance-sheet-statement) required: symbol, output_path optional: api_key, limit(default40), period(annual/quarter)'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    limit = getattr(args, "limit", 40)
    period = getattr(args, "period", None)

    if not symbol:
        raise ValueError('symbol parameter')
    if not output_path:
        raise ValueError('output_path parameter')

    output_path = _ensure_output_path(output_path)
    api_key = _get_api_key(args)

    _print(f"[INFO] get: {symbol}, period={period}, limit={limit}")
    data = _fetch_statement(api_key, symbol, "balance-sheet-statement", limit=limit, period=period)
    _save_statement(symbol, data, output_path, "balance_sheet")


def run_cash_flow(args) -> None:
    'get (cash-flow-statement) required: symbol, output_path optional: api_key, limit(default40), period(annual/quarter)'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    limit = getattr(args, "limit", 40)
    period = getattr(args, "period", None)

    if not symbol:
        raise ValueError('symbol parameter')
    if not output_path:
        raise ValueError('output_path parameter')

    output_path = _ensure_output_path(output_path)
    api_key = _get_api_key(args)

    _print(f"[INFO] get: {symbol}, period={period}, limit={limit}")
    data = _fetch_statement(api_key, symbol, "cash-flow-statement", limit=limit, period=period)
    _save_statement(symbol, data, output_path, "cash_flow")


def run_news(args) -> None:
    'get stock news data required: symbol symbols, output_path optional: api_key, limit, from_date, to_date'
    symbol = getattr(args, "symbol", None)
    symbols = getattr(args, "symbols", None)
    output_path = getattr(args, "output_path", None)
    limit = getattr(args, "limit", 50)
    from_date = getattr(args, "from_date", None)
    to_date = getattr(args, "to_date", None)
    
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    
    # symbol symbols
    if symbol:
        symbol_list = [symbol]
    elif symbols:
        symbol_list = _parse_symbols(symbols)
    else:
        raise ValueError('symbol symbols')
    
    _print(f"[INFO] get stock news: {symbol_list}, limit={limit}, from_date={from_date}, to_date={to_date}")
    
    # FMP API:/stock-newssymbols=AAPL, MSFT&limit=50&from=YYYY-MM-DD&to=YYYY-MM-DD
    url = f"{FMP_BASE_URL}/stock-news"
    params = {"apikey": api_key, "limit": limit}
    if len(symbol_list) == 1:
        params["symbols"] = symbol_list[0]
    else:
        params["symbols"] = ",".join(symbol_list)
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbols": symbol_list,
            "tool": "fmp_news",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] getnews failed: {e}")
        raise


def run_crypto_data(args) -> None:
    'get crypto data required: output_path optional: symbol, api_key, limit, start_date, end_date'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    limit = getattr(args, "limit", 100)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    
    _print(f"[INFO] get crypto data: symbol={symbol}, limit={limit}, start_date={start_date}, end_date={end_date}")
    
    if symbol:
        # time range, usehistory endpoint
        if start_date or end_date:
            url = f"{FMP_BASE_URL}/historical-price-full/crypto/{symbol.upper()}"
            params = {"apikey": api_key}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date
        else:
            url = f"{FMP_BASE_URL}/crypto/{symbol.upper()}"
            params = {"apikey": api_key}
    else:
        url = f"{FMP_BASE_URL}/crypto"
        params = {"apikey": api_key, "limit": limit}
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_crypto_data",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get crypto data failed: {e}")
        raise


def run_forex_data(args) -> None:
    'get forex market data required: output_path optional: symbol, api_key, limit, start_date, end_date'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    limit = getattr(args, "limit", 100)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    
    _print(f"[INFO] get forex data: symbol={symbol}, limit={limit}, start_date={start_date}, end_date={end_date}")
    
    if symbol:
        # time range, usehistory endpoint
        if start_date or end_date:
            url = f"{FMP_BASE_URL}/historical-price-full/forex/{symbol.upper()}"
            params = {"apikey": api_key}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date
        else:
            url = f"{FMP_BASE_URL}/forex/{symbol.upper()}"
            params = {"apikey": api_key}
    else:
        url = f"{FMP_BASE_URL}/forex"
        params = {"apikey": api_key, "limit": limit}
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_forex_data",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get forex data failed: {e}")
        raise


def run_analyst_estimates(args) -> None:
    'get analysis data required: symbol, output_path optional: period, limit, api_key'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    period = getattr(args, "period", None)
    limit = getattr(args, "limit", 40)
    
    if not symbol:
        raise ValueError('symbol')
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    symbol = symbol.upper().strip()
    
    _print(f"[INFO] get analysis: {symbol}, period={period}, limit={limit}")
    
    url = f"{FMP_BASE_URL}/analyst-estimates/{symbol}"
    params = {"apikey": api_key, "limit": limit}
    if period:
        params["period"] = period
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_analyst_estimates",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get analysis failed: {e}")
        raise


def run_price_target(args) -> None:
    'get data required: symbol, output_path optional: api_key'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    
    if not symbol:
        raise ValueError('symbol')
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    symbol = symbol.upper().strip()
    
    _print(f"[INFO] get: {symbol}")
    
    url = f"{FMP_BASE_URL}/price-target-summary/{symbol}"
    params = {"apikey": api_key}
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_price_target",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get failed: {e}")
        raise


def run_earnings_transcript(args) -> None:
    'get record required: symbol, output_path optional: quarter, year, api_key'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    quarter = getattr(args, "quarter", None)
    year = getattr(args, "year", None)
    
    if not symbol:
        raise ValueError('symbol')
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    symbol = symbol.upper().strip()
    
    _print(f"[INFO] get record: {symbol}, quarter={quarter}, year={year}")
    
    url = f"{FMP_BASE_URL}/earnings-transcript/{symbol}"
    params = {"apikey": api_key}
    if quarter:
        params["quarter"] = quarter
    if year:
        params["year"] = year
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_earnings_transcript",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get record failed: {e}")
        raise


def run_market_calendar(args) -> None:
    'get market required: output_path optional: from_date, to_date, api_key'
    output_path = getattr(args, "output_path", None)
    from_date = getattr(args, "from_date", None)
    to_date = getattr(args, "to_date", None)
    
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    
    _print(f"[INFO] get market: from_date={from_date}, to_date={to_date}")
    
    url = f"{FMP_BASE_URL}/market-calendar"
    params = {"apikey": api_key}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    
    try:
        data = _http_get(url, params)
        payload = {
            "tool": "fmp_market_calendar",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get market failed: {e}")
        raise


def run_form_13f(args) -> None:
    'get 13F data required: output_path optional: cik, date, from_date, to_date, api_key, limit: date from_date/to_date, prefer using from_date/to_date'
    output_path = getattr(args, "output_path", None)
    cik = getattr(args, "cik", None)
    date = getattr(args, "date", None)
    from_date = getattr(args, "from_date", None)
    to_date = getattr(args, "to_date", None)
    limit = getattr(args, "limit", 100)
    
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    
    _print(f"[INFO] get 13F: cik={cik}, date={date}, from_date={from_date}, to_date={to_date}, limit={limit}")
    
    url = f"{FMP_BASE_URL}/form-13f"
    params = {"apikey": api_key, "limit": limit}
    if cik:
        params["cik"] = cik
    # prefer usingdate range, use date
    if from_date or to_date:
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
    elif date:
        params["date"] = date
    
    try:
        data = _http_get(url, params)
        payload = {
            "tool": "fmp_form_13f",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get 13F failed: {e}")
        raise


def run_key_metrics(args) -> None:
    'get metrics data (TTM) required: symbol, output_path optional: period, limit, api_key'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    period = getattr(args, "period", None)
    limit = getattr(args, "limit", 40)
    
    if not symbol:
        raise ValueError('symbol')
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    symbol = symbol.upper().strip()
    
    _print(f"[INFO] get metrics: {symbol}, period={period}, limit={limit}")
    
    # prefer using key-metrics-ttm, use key-metrics
    url = f"{FMP_BASE_URL}/key-metrics-ttm/{symbol}"
    params = {"apikey": api_key, "limit": limit}
    if period:
        params["period"] = period
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_key_metrics",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[WARN] key-metrics-ttm failed, try key-metrics: {e}")
        url = f"{FMP_BASE_URL}/key-metrics/{symbol}"
        try:
            data = _http_get(url, params)
            payload = {
                "symbol": symbol,
                "tool": "fmp_key_metrics",
                "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
            }
            _save_json(output_path, payload)
        except Exception as e2:
            _print(f"[ERROR] get metrics failed: {e2}")
            raise


def run_enterprise_value(args) -> None:
    'get data required: symbol, output_path optional: period, limit, api_key'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    period = getattr(args, "period", None)
    limit = getattr(args, "limit", 40)
    
    if not symbol:
        raise ValueError('symbol')
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    symbol = symbol.upper().strip()
    
    _print(f"[INFO] get: {symbol}, period={period}, limit={limit}")
    
    url = f"{FMP_BASE_URL}/enterprise-value/{symbol}"
    params = {"apikey": api_key, "limit": limit}
    if period:
        params["period"] = period
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_enterprise_value",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get failed: {e}")
        raise


def run_commodity_data(args) -> None:
    'get market data required: output_path optional: symbol, api_key, limit, start_date, end_date'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    limit = getattr(args, "limit", 100)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    
    _print(f"[INFO] get data: symbol={symbol}, limit={limit}, start_date={start_date}, end_date={end_date}")
    
    if symbol:
        # time range, usehistory endpoint
        if start_date or end_date:
            url = f"{FMP_BASE_URL}/historical-price-full/commodity/{symbol.upper()}"
            params = {"apikey": api_key}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date
        else:
            url = f"{FMP_BASE_URL}/commodity/{symbol.upper()}"
            params = {"apikey": api_key}
    else:
        url = f"{FMP_BASE_URL}/commodity"
        params = {"apikey": api_key, "limit": limit}
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_commodity_data",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get data failed: {e}")
        raise


def run_insider_trading(args) -> None:
    'get data required: symbol, output_path optional: api_key, limit, start_date, end_date'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    limit = getattr(args, "limit", 100)
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    
    if not symbol:
        raise ValueError('symbol')
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    symbol = symbol.upper().strip()
    
    _print(f"[INFO] get: {symbol}, limit={limit}, start_date={start_date}, end_date={end_date}")
    
    url = f"{FMP_BASE_URL}/insider-trading/{symbol}"
    params = {"apikey": api_key, "limit": limit}
    if start_date:
        params["from"] = start_date
    if end_date:
        params["to"] = end_date
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_insider_trading",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get failed: {e}")
        raise


def run_esg_data(args) -> None:
    'get ESG data required: symbol, output_path optional: api_key'
    symbol = getattr(args, "symbol", None)
    output_path = getattr(args, "output_path", None)
    
    if not symbol:
        raise ValueError('symbol')
    if not output_path:
        raise ValueError('output_path')
    
    api_key = _get_api_key(args)
    output_path = _ensure_output_path(output_path)
    symbol = symbol.upper().strip()
    
    _print(f"[INFO] get ESG data: {symbol}")
    
    url = f"{FMP_BASE_URL}/esg-score/{symbol}"
    params = {"apikey": api_key}
    
    try:
        data = _http_get(url, params)
        payload = {
            "symbol": symbol,
            "tool": "fmp_esg_data",
            "data": data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        }
        _save_json(output_path, payload)
    except Exception as e:
        _print(f"[ERROR] get ESG data failed: {e}")
        raise


def run(args) -> None:
    '(FMP tool)'
    run_company_profile(args)
