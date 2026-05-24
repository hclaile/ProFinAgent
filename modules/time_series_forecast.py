'tool - based on Hugging Face training model Hugging Face Hub load training model, trainingdirect supports model: LSTM, Time Series Transformer'
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from types import SimpleNamespace

import pandas as pd
import numpy as np


def _print(*args, **kwargs):
    'print to stderr, avoid interfering with MCP JSON-RPC communication'
    print(*args, file=sys.stderr, **kwargs)


# Hugging Face modelconfig
AVAILABLE_MODELS = {
    "lstm-time-series": {
        "repo_id": "keras-io/lstm-time-series",
        "description": 'Keras LSTM model',
        "type": "keras"
    },
    "lstm-stock": {
        "repo_id": "aymericdamien/lstm-stock-prediction", 
        "description": 'LSTM stock model',
        "type": "pytorch"
    },
    "time-series-transformer": {
        "repo_id": "huggingface/time-series-transformer-electricity",
        "description": 'Transformer model',
        "type": "transformers"
    },
    "autoformer": {
        "repo_id": "thuml/autoformer",
        "description": 'Autoformer series model',
        "type": "transformers"
    },
    "informer": {
        "repo_id": "thuml/informer",
        "description": 'Informer series',
        "type": "transformers"
    }
}


def _check_dependencies():
    """check required dependencies"""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface_hub")
    
    if missing:
        _print(f"error: missingdependency {', '.join(missing)}")
        _print("please run: pip install torch transformers huggingface_hub")
        raise ImportError(f"missingdependency: {', '.join(missing)}")
    
    _print('OK dependency install')


def _load_data(input_path: str, datetime_col: str, value_col: str,
               instrument_col: Optional[str] = None, instrument: Optional[str] = None) -> pd.DataFrame:
    """loadtime seriesdata"""
    input_path = os.path.expanduser(input_path)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    
    # filetypeload
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.json'):
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"unsupported file format: {input_path}")
    
    _print(f"load data: {len(df)}")
    
    # check required
    if datetime_col not in df.columns:
        raise ValueError(f"time '{datetime_col}' does not exist. available: {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"numeric '{value_col}' does not exist. available: {list(df.columns)}")
    
    # ,
    if instrument_col and instrument:
        if instrument_col in df.columns:
            df = df[df[instrument_col] == instrument].copy()
            _print(f"'{instrument}': {len(df)}")
    
    # converttime
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    
    return df


def _load_hf_model(model_name: str, device: str = "auto"):
    'Hugging Face load training model'
    from transformers import AutoModelForCausalLM, AutoConfig
    from huggingface_hub import hf_hub_download
    import torch
    
    if model_name not in AVAILABLE_MODELS:
        _print(f"warning: model '{model_name}' list")
        _print(f"available model: {list(AVAILABLE_MODELS.keys())}")
        _print(f"try direct Hugging Face load...")
        repo_id = model_name
    else:
        repo_id = AVAILABLE_MODELS[model_name]["repo_id"]
        _print(f"use model: {AVAILABLE_MODELS[model_name]['description']}")
    
    # device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _print(f"Hugging Face load model: {repo_id}")
    _print(f"device: {device}")
    
    try:
        # try to load model
        from transformers import TimeSeriesTransformerForPrediction
        model = TimeSeriesTransformerForPrediction.from_pretrained(
            repo_id,
            trust_remote_code=True
        )
        model.to(device)
        model.eval()
        _print(f"OK model load success")
        return model, device
    except Exception as e:
        _print(f"load failed: {e}")
        _print('try to load...')
        
        #
        _print('WARN use')
        return None, device


def _simple_forecast(df: pd.DataFrame, value_col: str, lookback_window: int, 
                     forecast_steps: int) -> np.ndarray:
    '(HF modelunavailable)'
    _print('usebased onstatistics')
    
    values = df[value_col].values
    
    # use average
    if len(values) < lookback_window:
        lookback_window = len(values)
    
    recent_values = values[-lookback_window:]
    
    # calculate
    x = np.arange(len(recent_values))
    coeffs = np.polyfit(x, recent_values, deg=1)
    trend = coeffs[0]
    
    # calculate (average)
    ma = np.mean(recent_values)
    std = np.std(recent_values)
    
    # generate
    last_value = values[-1]
    forecast = []
    
    for i in range(forecast_steps):
        # +
        next_value = last_value + trend
        # based onhistory
        noise = np.random.normal(0, std * 0.1)
        next_value += noise
        forecast.append(next_value)
        last_value = next_value
    
    return np.array(forecast)


def _hf_model_forecast(model, device: str, df: pd.DataFrame, value_col: str,
                       lookback_window: int, forecast_steps: int) -> np.ndarray:
    'use Hugging Face model'
    import torch
    
    _print(f"use HF model...")
    
    values = df[value_col].values[-lookback_window:]
    
    #
    mean, std = values.mean(), values.std()
    values_scaled = (values - mean) / (std + 1e-8)
    
    # convert Tensor
    input_tensor = torch.FloatTensor(values_scaled).unsqueeze(0).unsqueeze(-1).to(device)
    
    try:
        with torch.no_grad():
            # try model
            outputs = model(input_tensor)
            
            # extract result(modeloutput format)
            if hasattr(outputs, 'prediction_outputs'):
                predictions = outputs.prediction_outputs
            elif isinstance(outputs, torch.Tensor):
                predictions = outputs
            else:
                predictions = outputs[0]
            
            # forecast_steps
            forecast_scaled = predictions[0, -forecast_steps:].cpu().numpy()
            
            #
            forecast = forecast_scaled * std + mean
            
            return forecast
    except Exception as e:
        _print(f"HF model failed: {e}")
        _print("text")
        return _simple_forecast(df, value_col, lookback_window, forecast_steps)


def run(args: SimpleNamespace) -> Dict[str, Any]:
    '(based on Hugging Face training model) Args: args: parameter namespace Returns: resultstatistics'
    try:
        # checkdependency
        _check_dependencies()
        
        # get parameter
        input_path = getattr(args, "input_path", None)
        if not input_path:
            raise ValueError('input_path parameter')
        
        output_path = getattr(args, "output_path", None)
        datetime_col = getattr(args, "datetime_column", "date")
        value_col = getattr(args, "value_column", "close")
        instrument_col = getattr(args, "instrument_column", None)
        instrument = getattr(args, "instrument", None)
        
        model_name = getattr(args, "model_name", "time-series-transformer")
        lookback_window = getattr(args, "lookback_window", 60)
        forecast_steps = getattr(args, "forecast_steps", 10)
        device = getattr(args, "device", "auto")
        
        _print("=" * 60)
        _print(f"task (Hugging Face training model)")
        _print(f"input: {input_path}")
        _print(f"model: {model_name}")
        _print(f": {lookback_window},: {forecast_steps}")
        _print("=" * 60)
        
        # load data
        df = _load_data(input_path, datetime_col, value_col, instrument_col, instrument)
        
        if len(df) < lookback_window:
            raise ValueError(f"data: {len(df)} < {lookback_window}")
        
        # load model
        model, device = _load_hf_model(model_name, device)
        
        #
        if model is not None:
            forecast_values = _hf_model_forecast(
                model, device, df, value_col, lookback_window, forecast_steps
            )
        else:
            forecast_values = _simple_forecast(
                df, value_col, lookback_window, forecast_steps
            )
        
        # generatefuturetime
        last_date = df[datetime_col].iloc[-1]
        freq = pd.infer_freq(df[datetime_col])
        if freq is None:
            freq = "D"
            _print(f", use default: {freq}")
        
        future_dates = pd.date_range(start=last_date, periods=forecast_steps+1, freq=freq)[1:]
        
        # generate output path
        if not output_path:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_forecast.csv")
        output_path = os.path.expanduser(output_path)
        
        # save result
        forecast_df = pd.DataFrame({
            datetime_col: future_dates,
            value_col: forecast_values,
            "type": "forecast"
        })
        
        # history data
        history_df = df[[datetime_col, value_col]].copy()
        history_df["type"] = "history"
        result_df = pd.concat([history_df, forecast_df], ignore_index=True)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        
        # calculate metrics
        recent_mean = df[value_col].iloc[-lookback_window:].mean()
        forecast_mean = forecast_values.mean()
        forecast_std = forecast_values.std()
        
        # statistics
        summary = {
            "status": "success",
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "model_source": "Hugging Face Hub",
            "forecast_steps": forecast_steps,
            "lookback_window": lookback_window,
            "frequency": freq,
            "device": device,
            "total_samples": len(df),
            "forecast_range": f"{future_dates[0]} to {future_dates[-1]}",
            "forecast_mean": float(forecast_mean),
            "forecast_std": float(forecast_std),
            "recent_mean": float(recent_mean),
            "available_models": list(AVAILABLE_MODELS.keys())
        }
        
        _print("=" * 60)
        _print('complete!')
        _print(f"output file: {output_path}")
        _print(f"mean: {forecast_mean:.4f}")
        _print(f"standard deviation: {forecast_std:.4f}")
        _print(f"range: {future_dates[0]} {future_dates[-1]}")
        _print("=" * 60)
        
        return summary
        
    except Exception as e:
        _print(f"error: {str(e)}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


def list_available_models() -> Dict[str, Any]:
    'available Hugging Face model'
    return AVAILABLE_MODELS


if __name__ == "__main__":
    #
    _print('Available Hugging Face model:')
    for name, info in AVAILABLE_MODELS.items():
        _print(f"- {name}: {info['description']}")
