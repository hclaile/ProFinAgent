import asyncio
import json
import warnings
from types import SimpleNamespace

# text pkg_resources textwarning(text)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types

from modules import import_data as mod_import_data
from modules import data_download as mod_data_download
from modules import feature as mod_feature
from modules import train as mod_train
from modules import backtest as mod_backtest
from modules import report as mod_report
from modules import crawler as mod_crawler
from modules import talib_technical_indicators as mod_talib_indicators
from modules import sentiment_analysis_bert as mod_sentiment_analysis
from modules import time_series_forecast as mod_time_series_forecast
from modules import train_qcm_mcp as mod_train_qcm_mcp
from modules import calculate_ic as mod_calculate_ic
from modules import train_AFF as mod_train_AFF
from modules import train_gfn_AlphaSAGE_MCP as mod_train_gfn_AlphaSAGE
from modules import train_GP_AlphaSAGE_MCP as mod_train_GP_AlphaSAGE
from modules import train_PPO_AlphaSAGE_MCP as mod_train_PPO_AlphaSAGE
from modules import qlib_benchmark_runner_fastapi as mod_qlib_benchmark_runner
from modules import mineru_pdf_to_json as mod_mineru_pdf_to_json
from modules import fmp_tools as mod_fmp_tools
from modules import tavily_search as mod_tavily_search


server = Server("ProFinAgent")


def ns(d):
	return SimpleNamespace(**d)


TRAINING_MODULES = {
	"train_qcm": mod_train_qcm_mcp,
	"train_AFF": mod_train_AFF,
	"train_gfn_AlphaSAGE": mod_train_gfn_AlphaSAGE,
	"train_GP_AlphaSAGE": mod_train_GP_AlphaSAGE,
	"train_PPO_AlphaSAGE": mod_train_PPO_AlphaSAGE,
}


def json_text(payload):
	return [types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, default=str))]


def _task_snapshot(module, tool_name, task_id, log_tail=20):
	with module.tasks_lock:
		if task_id not in module.tasks:
			raise KeyError(f"task does not exist: {task_id}")
		task = module.tasks[task_id]
		logs = list(task.get("log", []))
		return {
			"tool": tool_name,
			"task_id": task_id,
			"task_name": task.get("task_name", ""),
			"status": task.get("status"),
			"command": task.get("command"),
			"pid": task.get("pid"),
			"created_at": task.get("created_at"),
			"updated_at": task.get("updated_at"),
			"log_lines": len(logs),
			"recent_logs": logs[-log_tail:] if log_tail else [],
			"error": task.get("error"),
			"repository": str(getattr(module, "REPO_PATH", "")),
			"train_script": getattr(module, "TRAIN_SCRIPT", ""),
		}


def _list_training_tasks(tool_name=None, status=None, limit=50):
	items = []
	modules = {tool_name: TRAINING_MODULES[tool_name]} if tool_name else TRAINING_MODULES
	for current_tool, module in modules.items():
		with module.tasks_lock:
			task_ids = list(module.tasks.keys())
		for task_id in task_ids:
			snapshot = _task_snapshot(module, current_tool, task_id, log_tail=0)
			if status and snapshot.get("status") != status:
				continue
			items.append(snapshot)
	items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
	return {"total": len(items[:limit]), "tasks": items[:limit]}


def _training_logs(tool_name, task_id, lines=100, offset=0):
	module = TRAINING_MODULES[tool_name]
	with module.tasks_lock:
		if task_id not in module.tasks:
			raise KeyError(f"task does not exist: {task_id}")
		task = module.tasks[task_id]
		all_logs = list(task.get("log", []))
	total = len(all_logs)
	start = max(0, total - offset - lines)
	end = total - offset if offset > 0 else total
	return {
		"tool": tool_name,
		"task_id": task_id,
		"status": task.get("status"),
		"total_lines": total,
		"returned_lines": len(all_logs[start:end]),
		"logs": all_logs[start:end],
	}


async def _start_training_tool(tool_name, module, args, run_in_thread):
	with module.tasks_lock:
		before_ids = set(module.tasks.keys())
	await run_in_thread(module.run, args)
	with module.tasks_lock:
		new_ids = [task_id for task_id in module.tasks.keys() if task_id not in before_ids]
	if not new_ids:
		return {"tool": tool_name, "status": "started", "message": "training started, but task_id was not captured"}
	new_ids.sort(key=lambda task_id: module.tasks[task_id].get("created_at") or "", reverse=True)
	task_id = new_ids[0]
	snapshot = _task_snapshot(module, tool_name, task_id, log_tail=10)
	snapshot["message"] = "training task started"
	return snapshot


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
	return [
		types.Tool(name="backtest", description="Backtest tool. Evaluate strategy or model predictions on historical data; computes return, Sharpe, max drawdown, etc.", inputSchema={}),
		types.Tool(name="talib_technical_indicators", description="TA-Lib technical indicator calculator. Based on TA-Lib with 150+ indicators: trend (MA/EMA/MACD), momentum (RSI/MOM/ROC), volatility (ATR/BBANDS), candlestick patterns (60+ CDL), etc. Users can specify indicators or compute all; validation data can be generated automatically.", inputSchema={}),
		types.Tool(name="sentiment_analysis_bert", description="Local BERT/FinBERT sentiment analysis (default local finbert-tone). Outputs positive/negative/neutral with confidence; supports batch processing and custom local model path.", inputSchema={}),
		types.Tool(name="mineru_pdf_to_json", description="Use MinerU API to parse PDF into structured JSON for downstream text mining and extraction.", inputSchema={}),
		types.Tool(name="train_qcm", description="AlphaQCM training tool. Quantile-based distributional RL for Alpha mining. Supports QRDQN, IQN, FQF architectures; learns and optimizes Alpha factors via RL. Runs asynchronously with task/log queries.", inputSchema={}),
		types.Tool(name="train_AFF", description="AlphaForge training tool. GAN-based Alpha factor generation; learns market distributions to generate predictive factors. Runs asynchronously and returns task_id for status/log queries.", inputSchema={}),
		types.Tool(name="train_gfn_AlphaSAGE", description="AlphaSAGE training tool. GFlowNet-based Alpha factor generation with deep RL. Runs asynchronously and returns task_id for status/log queries.", inputSchema={}),
		types.Tool(name="train_GP_AlphaSAGE", description="Genetic Programming-based Alpha factor mining. Runs asynchronously and returns task_id for status/log queries.", inputSchema={}),
		types.Tool(name="train_PPO_AlphaSAGE", description="PPO-based Alpha factor generation. Uses RL (Proximal Policy Optimization) to generate/optimize factors with stable updates. Runs asynchronously with task/log queries.", inputSchema={}),
		types.Tool(name="train_task_status", description="Query status for an inline training task. Required: tool, task_id. tool is one of train_qcm/train_AFF/train_gfn_AlphaSAGE/train_GP_AlphaSAGE/train_PPO_AlphaSAGE.", inputSchema={}),
		types.Tool(name="train_task_logs", description="Query logs for an inline training task. Required: tool, task_id. Optional: lines, offset.", inputSchema={}),
		types.Tool(name="train_task_list", description="List inline training tasks across all training tools, or filter by tool/status.", inputSchema={}),
		types.Tool(name="qlib_benchmark_runner", description="Qlib Benchmark runner. Execute standard Qlib benchmark models via YAML configs (XGBoost/LightGBM/GRU/LSTM/Transformer, etc.). Scans benchmarks directory, supports updating provider_uri, and runs via qrun. Asynchronous with task/log queries.", inputSchema={}),
		types.Tool(name="qlib_benchmark_list_models", description="List all available Qlib benchmark models. Scans benchmarks directory and returns model types (XGBoost/LightGBM/GRU/LSTM, etc.) with YAML paths for selection.", inputSchema={}),
		types.Tool(name="calculate_ic", description="Compute IC (Information Coefficient) metrics for factors. Returns IC mean/std, IR, Rank IC mean/std, Rank IR, IC win rate, etc., to assess predictive power and stability.", inputSchema={}),
		types.Tool(name="fmp_company_profile", description="Fetch company profile information from Financial Modeling Prep (FMP) by company name or symbol.", inputSchema={}),
		types.Tool(name="fmp_income_statement", description="Get income statement from Financial Modeling Prep (FMP) stable API (income-statement).", inputSchema={}),
		types.Tool(name="fmp_balance_sheet", description="Get balance sheet from Financial Modeling Prep (FMP) stable API (balance-sheet-statement).", inputSchema={}),
		types.Tool(name="fmp_cash_flow", description="Get cash flow statement from Financial Modeling Prep (FMP) stable API (cash-flow-statement).", inputSchema={}),
		types.Tool(name="fmp_historical_data", description=f"Get historical OHLCV data via FinancialModelingPrep using FinanceToolkit Toolkit (endpoint: /historical-price-full/{{symbol}}).", inputSchema={}),
		types.Tool(name="fmp_treasury_data", description="Get US Treasury yield data via FinanceToolkit Toolkit (treasury endpoint).", inputSchema={}),
		types.Tool(name="time_series_forecast", description="Time-series forecasting tool (Hugging Face pre-trained models).", inputSchema={}),
		types.Tool(name="fmp_profitability_ratios", description="Collect profitability ratios via FinanceToolkit (gross margin, net margin, ROE/ROA, etc.).", inputSchema={}),
		types.Tool(name="fmp_liquidity_ratios", description="Collect liquidity ratios via FinanceToolkit (current ratio, quick ratio, etc.). Requires explicit output_path.", inputSchema={}),
		types.Tool(name="fmp_solvency_ratios", description="Collect solvency ratios via FinanceToolkit (debt-to-equity, interest coverage, etc.). Requires explicit output_path.", inputSchema={}),
		types.Tool(name="fmp_valuation_ratios", description="Collect valuation ratios via FinanceToolkit (PE/PS, EV/EBITDA, etc.) using financial statements + historical price.", inputSchema={}),
		types.Tool(name="fmp_altman_z_score", description="Compute Altman Z-Score via FinanceToolkit models (bankruptcy risk proxy). Requires explicit output_path.", inputSchema={}),
		types.Tool(name="fmp_piotroski_score", description="Compute Piotroski F-Score via FinanceToolkit models (9-signal fundamental score).", inputSchema={}),
		types.Tool(name="fmp_wacc", description="Compute WACC (Weighted Average Cost of Capital) via FinanceToolkit models. Uses financial statements + treasury + beta internally where supported. Requires explicit output_path.", inputSchema={}),
		types.Tool(name="fmp_value_at_risk", description="Compute Value-at-Risk (VaR) via FinanceToolkit risk module using historical price volatility.", inputSchema={}),
		types.Tool(name="fmp_news", description="Get financial market news for stock symbols from FMP API.", inputSchema={}),
		types.Tool(name="fmp_crypto_data", description="Get cryptocurrency market data from FMP API.", inputSchema={}),
		types.Tool(name="fmp_forex_data", description="Get forex (foreign exchange) market data from FMP API.", inputSchema={}),
		types.Tool(name="fmp_analyst_estimates", description="Get analyst estimates (EPS, revenue, etc.) for a stock symbol from FMP API.", inputSchema={}),
		types.Tool(name="fmp_price_target", description="Get analyst price target summary for a stock symbol from FMP API.", inputSchema={}),
		types.Tool(name="fmp_earnings_transcript", description="Get earnings call transcripts for a stock symbol from FMP API.", inputSchema={}),
		types.Tool(name="fmp_market_calendar", description="Get market calendar (earnings dates, economic events, etc.) from FMP API.", inputSchema={}),
		types.Tool(name="fmp_form_13f", description="Get Form 13F filings data from FMP API.", inputSchema={}),
		types.Tool(name="fmp_key_metrics", description="Get key financial metrics (TTM - Trailing Twelve Months) for a stock symbol from FMP API.", inputSchema={}),
		types.Tool(name="fmp_enterprise_value", description="Get enterprise value data for a stock symbol from FMP API.", inputSchema={}),
		types.Tool(name="fmp_commodity_data", description="Get commodity market data from FMP API. Returns prices for commodities like gold, oil, etc. Can fetch specific commodity by symbol or list all available.", inputSchema={}),
		types.Tool(name="fmp_insider_trading", description="Get insider trading data for a stock symbol from FMP API. Returns transactions by company insiders (officers, directors, etc.).", inputSchema={}),
		types.Tool(name="fmp_esg_data", description="Get ESG (Environmental, Social, Governance) score data for a stock symbol from FMP API.", inputSchema={}),
		types.Tool(name="convert_to_qlib", description="Convert stock market data that is not in qlib format to qlib format.", inputSchema={}),
		
		# texttool
		types.Tool(name="crawler_fred_search", description="FRED economic data search. Query macro indicators in the Federal Reserve FRED database (GDP, inflation, unemployment, etc.).", inputSchema={}),
		types.Tool(name="crawler_finnhub_news", description="Finnhub equity news fetcher. Get news for a given ticker (title, summary, time, source).", inputSchema={}),
		types.Tool(name="crawler_finhun_scraper", description="Finhun web content scraper. Batch crawl a list of URLs and extract structured info such as title and body.", inputSchema={}),
		types.Tool(name="crawler_stock_data_fetcher", description="A-share equity data fetcher (akShare). Supports fetching China A-share historical prices by ticker or company name.", inputSchema={}),
		types.Tool(name="crawler_google_search", description="Use this for normal/free-form queries and questions; returns structured results (snippets, links, etc.).", inputSchema={}),
		types.Tool(name="crawler_reddit_extractor", description="Reddit posts & comments extractor. Pull posts and top comments by keyword.", inputSchema={}),
		types.Tool(name="crawler_reddit_news_scraper", description="Reddit news scraper. Search related company news and discussions across subreddits, then crawl linked content.", inputSchema={}),
		types.Tool(name="crawler_sec_filings", description="SEC filing downloader and XBRL-to-JSON converter. Automatically fetches 10-K/10-Q (and related) filings for US-listed companies, then converts the downloaded primary HTM/iXBRL into JSON.", inputSchema={}),
		types.Tool(name="crawler_wiki_search", description="The entered query must be an exact name, not a query that requires a search engine to look up.", inputSchema={}),
		types.Tool(name="crawler_yfinance", description="yfinance equity data downloader. Fetch global equities (US/HK, etc.) historical prices such as open/close/volume.", inputSchema={}),
		types.Tool(name="tavily_search", description="Tavily search results with answer and cleaned text snippets. Returns structured answers and cleaned text snippets optimized for AI agents.", inputSchema={}),
		types.Tool(name="report_generate", description="Generate professional PDF reports by combining LLM analysis with data visualizations. ", inputSchema={}),
	]


@server.call_tool(validate_input=False)
async def handle_call_tool(name: str, arguments: dict | None):
	args = ns(arguments or {})

	# textcalltext,textdirecttext
	async def run_in_thread(fn, *f_args):
		return await asyncio.to_thread(fn, *f_args)
	if name == "import_data":
		await run_in_thread(mod_import_data.run, args)
		return [types.TextContent(type="text", text="import_data done")]
	elif name == "data_download":
		# textget market parameter,textdefaulttext provider_uri text region
		market = getattr(args, "market", "cn")
		# text market textdefaulttext provider_uri text region
		if market == "us":
			default_provider_uri = "./ProFinAgent/.qlib/qlib_data/us_data"
			default_region = "us"
		else:  # defaulttext cn
			default_provider_uri = "./ProFinAgent/.qlib/qlib_data/cn_data"
			default_region = "cn"
		
		# textusertext provider_uri,usetext market textdefaulttext
		provider_uri = getattr(args, "provider_uri", None)
		if not provider_uri:
			provider_uri = default_provider_uri
		
		# textusertext region,usetext market textdefaulttext(text market text)
		region = getattr(args, "region", None)
		if not region:
			region = default_region
		else:
			# textusertext region,text market text,textwarningtextuse market text region
			if (market == "us" and region != "us") or (market == "cn" and region != "cn"):
				import sys
				print(f"warning: market={market} text region={region} text,textuse region={default_region}", file=sys.stderr)
				region = default_region
		
		p = ns({
			"market": market,
			"provider_uri": provider_uri,
			"region": region,
			"force": bool(getattr(args, "force", False)),
			"company": getattr(args, "company", None),
			"start_date": getattr(args, "start_date", None),
			"end_date": getattr(args, "end_date", None),
			"output_path": getattr(args, "output_path", None),
		})
		await run_in_thread(mod_data_download.run, p)
		return [types.TextContent(type="text", text="data_download done")]
	elif name == "feature":
		await run_in_thread(mod_feature.run, args)
		return [types.TextContent(type="text", text="feature done")]
	elif name == "train":
		await run_in_thread(mod_train.run, args)
		return [types.TextContent(type="text", text="train done")]
	elif name == "backtest":
		await run_in_thread(mod_backtest.run, args)
		return [types.TextContent(type="text", text="backtest done")]
	elif name == "report":
		await run_in_thread(mod_report.run, args)
		return [types.TextContent(type="text", text="report done")]
	elif name == "talib_technical_indicators":
		await run_in_thread(mod_talib_indicators.run, args)
		return [types.TextContent(type="text", text="talib_technical_indicators done")]
	elif name == "time_series_forecast":
		result = await run_in_thread(mod_time_series_forecast.run, args)
		return [types.TextContent(type="text", text=str(result))]
	elif name == "sentiment_analysis_bert":
		await run_in_thread(mod_sentiment_analysis.run, args)
		return [types.TextContent(type="text", text="sentiment_analysis_bert done")]
	elif name == "mineru_pdf_to_json":
		result = await run_in_thread(mod_mineru_pdf_to_json.run, args)
		return [types.TextContent(type="text", text=str(result))]
	elif name == "train_qcm":
		return json_text(await _start_training_tool("train_qcm", mod_train_qcm_mcp, args, run_in_thread))
	elif name == "train_AFF":
		return json_text(await _start_training_tool("train_AFF", mod_train_AFF, args, run_in_thread))
	elif name == "train_gfn_AlphaSAGE":
		return json_text(await _start_training_tool("train_gfn_AlphaSAGE", mod_train_gfn_AlphaSAGE, args, run_in_thread))
	elif name == "train_GP_AlphaSAGE":
		return json_text(await _start_training_tool("train_GP_AlphaSAGE", mod_train_GP_AlphaSAGE, args, run_in_thread))
	elif name == "train_PPO_AlphaSAGE":
		return json_text(await _start_training_tool("train_PPO_AlphaSAGE", mod_train_PPO_AlphaSAGE, args, run_in_thread))
	elif name == "train_task_status":
		tool_name = getattr(args, "tool", None)
		task_id = getattr(args, "task_id", None)
		if tool_name not in TRAINING_MODULES:
			raise ValueError(f"unknown training tool: {tool_name}")
		return json_text(_task_snapshot(TRAINING_MODULES[tool_name], tool_name, task_id, int(getattr(args, "log_tail", 20))))
	elif name == "train_task_logs":
		tool_name = getattr(args, "tool", None)
		task_id = getattr(args, "task_id", None)
		if tool_name not in TRAINING_MODULES:
			raise ValueError(f"unknown training tool: {tool_name}")
		return json_text(_training_logs(tool_name, task_id, int(getattr(args, "lines", 100)), int(getattr(args, "offset", 0))))
	elif name == "train_task_list":
		tool_name = getattr(args, "tool", None)
		status = getattr(args, "status", None)
		limit = int(getattr(args, "limit", 50))
		if tool_name is not None and tool_name not in TRAINING_MODULES:
			raise ValueError(f"unknown training tool: {tool_name}")
		return json_text(_list_training_tasks(tool_name, status, limit))
	elif name == "qlib_benchmark_runner":
		await run_in_thread(mod_qlib_benchmark_runner.run, args)
		return [types.TextContent(type="text", text="qlib_benchmark_runner done")]
	elif name == "qlib_benchmark_list_models":
		await run_in_thread(mod_qlib_benchmark_runner.list_models, args)
		return [types.TextContent(type="text", text="qlib_benchmark_list_models done")]
	elif name == "calculate_ic":
		await run_in_thread(mod_calculate_ic.run, args)
		return [types.TextContent(type="text", text="calculate_ic done")]
	# texttooltext
	elif name == "crawler_fred_search":
		crawler_args = ns({
			"crawler_type": "fred_search",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_fred_search done")]
	elif name == "crawler_finnhub_news":
		crawler_args = ns({
			"crawler_type": "finnhub_news",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_finnhub_news done")]
	elif name == "crawler_finhun_scraper":
		crawler_args = ns({
			"crawler_type": "finhun_scraper",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_finhun_scraper done")]
	elif name == "crawler_google_kg":
		crawler_args = ns({
			"crawler_type": "google_kg",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_google_kg done")]
	elif name == "crawler_stock_data_fetcher":
		crawler_args = ns({
			"crawler_type": "stock_data_fetcher",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_stock_data_fetcher done")]
	elif name == "crawler_google_search":
		crawler_args = ns({
			"crawler_type": "google_search",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_google_search done")]
	elif name == "crawler_reddit_extractor":
		crawler_args = ns({
			"crawler_type": "reddit_extractor",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_reddit_extractor done")]
	elif name == "crawler_reddit_news_scraper":
		crawler_args = ns({
			"crawler_type": "reddit_news_scraper",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_reddit_news_scraper done")]
	elif name == "crawler_sec_filings":
		crawler_args = ns({
			"crawler_type": "sec_filings",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_sec_filings done")]
	elif name == "crawler_wiki_search":
		crawler_args = ns({
			"crawler_type": "wiki_search",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_wiki_search done")]
	elif name == "crawler_yfinance":
		crawler_args = ns({
			"crawler_type": "yfinance",
			**(arguments or {})
		})
		await run_in_thread(mod_crawler.run, crawler_args)
		return [types.TextContent(type="text", text="crawler_yfinance done")]
	elif name == "fmp_company_profile":
		await run_in_thread(mod_fmp_tools.run_company_profile, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_company_profile done")]
	elif name == "tavily_search":
		await run_in_thread(mod_tavily_search.run, ns(arguments or {}))
		return [types.TextContent(type="text", text="tavily_search done")]
	elif name == "report_generate":
		await run_in_thread(mod_report.run_generate_report, ns(arguments or {}))
		return [types.TextContent(type="text", text="report_generate done")]
	elif name == "fmp_income_statement":
		await run_in_thread(mod_fmp_tools.run_income_statement, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_income_statement done")]
	elif name == "fmp_balance_sheet":
		await run_in_thread(mod_fmp_tools.run_balance_sheet, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_balance_sheet done")]
	elif name == "fmp_cash_flow":
		await run_in_thread(mod_fmp_tools.run_cash_flow, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_cash_flow done")]
	elif name == "fmp_historical_data":
		await run_in_thread(mod_fmp_tools.run_historical_data, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_historical_data done")]
	elif name == "fmp_treasury_data":
		await run_in_thread(mod_fmp_tools.run_treasury_data, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_treasury_data done")]
	elif name == "fmp_profitability_ratios":
		await run_in_thread(mod_fmp_tools.run_profitability_ratios, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_profitability_ratios done")]
	elif name == "fmp_liquidity_ratios":
		await run_in_thread(mod_fmp_tools.run_liquidity_ratios, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_liquidity_ratios done")]
	elif name == "fmp_solvency_ratios":
		await run_in_thread(mod_fmp_tools.run_solvency_ratios, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_solvency_ratios done")]
	elif name == "fmp_valuation_ratios":
		await run_in_thread(mod_fmp_tools.run_valuation_ratios, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_valuation_ratios done")]
	elif name == "fmp_altman_z_score":
		await run_in_thread(mod_fmp_tools.run_altman_z_score, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_altman_z_score done")]
	elif name == "fmp_piotroski_score":
		await run_in_thread(mod_fmp_tools.run_piotroski_score, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_piotroski_score done")]
	elif name == "fmp_wacc":
		await run_in_thread(mod_fmp_tools.run_wacc, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_wacc done")]
	elif name == "fmp_value_at_risk":
		await run_in_thread(mod_fmp_tools.run_value_at_risk, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_value_at_risk done")]
	elif name == "fmp_news":
		await run_in_thread(mod_fmp_tools.run_news, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_news done")]
	elif name == "fmp_crypto_data":
		await run_in_thread(mod_fmp_tools.run_crypto_data, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_crypto_data done")]
	elif name == "fmp_forex_data":
		await run_in_thread(mod_fmp_tools.run_forex_data, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_forex_data done")]
	elif name == "fmp_analyst_estimates":
		await run_in_thread(mod_fmp_tools.run_analyst_estimates, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_analyst_estimates done")]
	elif name == "fmp_price_target":
		await run_in_thread(mod_fmp_tools.run_price_target, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_price_target done")]
	elif name == "fmp_earnings_transcript":
		await run_in_thread(mod_fmp_tools.run_earnings_transcript, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_earnings_transcript done")]
	elif name == "fmp_market_calendar":
		await run_in_thread(mod_fmp_tools.run_market_calendar, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_market_calendar done")]
	elif name == "fmp_form_13f":
		await run_in_thread(mod_fmp_tools.run_form_13f, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_form_13f done")]
	elif name == "fmp_key_metrics":
		await run_in_thread(mod_fmp_tools.run_key_metrics, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_key_metrics done")]
	elif name == "fmp_enterprise_value":
		await run_in_thread(mod_fmp_tools.run_enterprise_value, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_enterprise_value done")]
	elif name == "fmp_commodity_data":
		await run_in_thread(mod_fmp_tools.run_commodity_data, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_commodity_data done")]
	elif name == "fmp_insider_trading":
		await run_in_thread(mod_fmp_tools.run_insider_trading, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_insider_trading done")]
	elif name == "fmp_esg_data":
		await run_in_thread(mod_fmp_tools.run_esg_data, ns(arguments or {}))
		return [types.TextContent(type="text", text="fmp_esg_data done")]
	else:
		return [types.TextContent(type="text", text=f"unknown tool: {name}")]


async def main():
	async with stdio_server() as streams:
		await server.run(
			streams[0],
			streams[1],
			server.create_initialization_options(notification_options=NotificationOptions()),
		)


if __name__ == "__main__":
	asyncio.run(main())
