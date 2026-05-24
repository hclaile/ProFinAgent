import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import pandas as pd
import numpy as np


def _load_config(path: str) -> Dict[str, Any]:
	path = os.path.expanduser(path)
	with open(path, "r", encoding="utf-8") as f:
		if path.endswith(('. yml', '. yaml')):
			return yaml.safe_load(f)
		return json.load(f)


def _ensure_provider_uri(provider_uri: str, region: str, cfg: Dict[str, Any]) -> str:
	'qlib provider data exists; does not exist auto download.'
	provider_uri = os.path.expanduser(provider_uri)

	try:
		from qlib.tests.data import GetData
		from qlib.utils import exists_qlib_data
	except ImportError as e:
		raise FileNotFoundError(
		 f"provider_uri unavailable, current qlib download/tool: {provider_uri}"
		) from e

	if exists_qlib_data(provider_uri):
		return provider_uri

	interval = str(cfg.get("interval", cfg.get("freq", "1d")))
	data_name = cfg.get("qlib_data_name")
	if not data_name:
		use_simple_data = str(cfg.get("use_simple_data", "")).lower() in {"1", "true", "yes"}
		data_name = "qlib_data_simple" if use_simple_data else "qlib_data"

	os.makedirs(provider_uri, exist_ok=True)
	try:
		GetData().qlib_data(
		 name=data_name,
		 target_dir=provider_uri,
		 interval=interval,
		 region=region,
		 exists_skip=False,
		)
	except Exception as e:
		raise RuntimeError(
		 f"provider_uri unavailable, auto download qlib data failed: {provider_uri}, region={region}, interval={interval}"
		) from e

	if not exists_qlib_data(provider_uri):
		raise FileNotFoundError(f"auto download complete provider_uri unavailable: {provider_uri}")

	return provider_uri


def _init_qlib_from_cfg(cfg: Dict[str, Any]) -> None:
	import qlib
	from qlib.config import REG_CN, REG_US

	provider_uri = os.path.expanduser(cfg.get("provider_uri", '~/.qlib/qlib_data/cn_data'))
	region = cfg.get("region", "cn")
	provider_uri = _ensure_provider_uri(provider_uri, region, cfg)
	region_const = REG_CN if region == "cn" else REG_US
	qlib.init(provider_uri=provider_uri, region=region_const)


def _load_predictions(pred_path: str) -> pd.DataFrame:
	pred_path = os.path.expanduser(pred_path)
	if pred_path.endswith('.parquet'):
		return pd.read_parquet(pred_path)
	return pd.read_csv(pred_path, index_col=0)


def _simple_backtest(pred_df: pd.DataFrame, score_col: str = "score", label_col: str = "label", topk: int = 50) -> Dict[str, Any]:
	'Top-K example, calculate//'
	# pred_df (datetime, instrument), label next_return
	if not isinstance(pred_df.index, pd.MultiIndex):
		raise ValueError('pred_df (datetime, instrument) Multi Index')

 # exists 'score' sort, 'label'
	dates = pred_df.index.get_level_values(0).unique().sort_values()
	cumrets = []
	for dt in dates:
		df_dt = pred_df.xs(dt, level=0)
		df_dt = df_dt.sort_values(score_col, ascending=False).head(topk)
		ret = df_dt[label_col].mean()
		cumrets.append(ret)

	rets = np.array(cumrets)
	cum = float(np.cumprod(1 + rets)[-1] - 1)
	ann_ret = float((1 + cum) ** (252 / max(1, len(rets))) - 1)
	ann_vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else 0.0
	sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
	return {
	 "cum_return": cum,
	 "annual_return": ann_ret,
	 "annual_vol": ann_vol,
	 "sharpe": sharpe,
	 "num_periods": int(len(rets)),
	}


def run(args) -> None:
	cfg = _load_config(args.config)
	if not cfg.get("raw_parquet"):
		_init_qlib_from_cfg(cfg)

	if args.pred_path:
		pred_df = _load_predictions(args.pred_path)
		metrics = _simple_backtest(pred_df, score_col=cfg.get("score_col", "score"), label_col=cfg.get("label_col", "label"), topk=cfg.get("topk", 50))
	else:
	 # raw_parquet mode
		raw_path = cfg.get("raw_parquet")
		if not raw_path:
			raise ValueError('missing --pred_path cfg.raw_parquet')
		raw_path = os.path.expanduser(raw_path)
		pred_df = pd.read_parquet(raw_path)
		metrics = _simple_backtest(pred_df, score_col=cfg.get("score_col", "score"), label_col=cfg.get("label_col", "label"), topk=cfg.get("topk", 50))

	out_dir = os.path.expanduser(cfg.get("output", "./outputs"))
	Path(out_dir).mkdir(parents=True, exist_ok=True)
	out_path = os.path.join(out_dir, 'backtest_metrics.json')
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)

	print(f"complete, metrics save: {out_path}")
