import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List

from agent.tmdb_agent import TMDBAgent


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATASET_PATH = os.path.join(ROOT_DIR, "RestBench", 'tmdb.json')
DEFAULT_TOOL_CARD_PATH = os.path.join(ROOT_DIR, "RestBench", 'tmdb_tool_card.json')
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "Result", "tmdb_agent")


def normalize_tool_name(tool_name: str) -> str:
    s = (tool_name or "").strip()
    s = re.sub(r"\s+", '', s)
    if not s:
        return s
    parts = s.split(None, 1)
    if len(parts) == 1:
        return parts[0].upper()
    method = parts[0].upper()
    path = parts[1].strip()
    return f"{method} {path}"


def extract_predicted_tools(select_result: Dict[str, Any]) -> List[str]:
    calls = select_result.get("toolchain_calls", [])
    if not isinstance(calls, list):
        return []
    tools: List[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        t = call.get("tool") or call.get("tool_name")
        if t:
            tools.append(normalize_tool_name(str(t)))
    return tools


def evaluate_prediction(pred: List[str], gold: List[str]) -> Dict[str, Any]:
    pred_norm = [normalize_tool_name(x) for x in pred]
    gold_norm = [normalize_tool_name(x) for x in gold]

    exact_match = pred_norm == gold_norm

    pred_set = set(pred_norm)
    gold_set = set(gold_norm)
    inter = pred_set.intersection(gold_set)

    precision = len(inter) / len(pred_set) if pred_set else 0.0
    recall = len(inter) / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "exact_match": exact_match,
        "precision_set": precision,
        "recall_set": recall,
        "f1_set": f1,
        "predicted_len": len(pred_norm),
        "gold_len": len(gold_norm),
    }


def build_output_path(output_path: str) -> str:
    if output_path:
        return output_path
    return os.path.join(DEFAULT_OUTPUT_DIR, 'tmdb_agent_benchmark_gpt4omini.json')


def load_existing_output(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_output(path: str, payload: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return [x for x in data if isinstance(x, dict)]


def run_single_case(
    idx: int,
    item: Dict[str, Any],
    tool_card_path: str,
    agent: TMDBAgent,
    max_retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    query = str(item.get("query", "")).strip()
    solution_raw = item.get("solution", [])
    solution = [normalize_tool_name(str(x)) for x in solution_raw] if isinstance(solution_raw, list) else []

    result: Dict[str, Any] = {
        "index": idx,
        "query": query,
        "reference_solution": solution,
        "status": "success",
    }

    if not query:
        result["status"] = "error"
        result["error"] = "empty query"
        result["llm_toolchain_calls"] = []
        result["llm_dag_result"] = []
        return result

    start = time.time()
    try:
        select_result = agent.select_tools(
            question=query,
            tools=tool_card_path,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        calls = select_result.get("toolchain_calls", [])
        if not isinstance(calls, list):
            calls = []

        dag_result = agent.generate_tool_dependencies(
            toolchain_calls=calls,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )

        reference_answer = json.dumps({"solution": solution}, ensure_ascii=False)
        reflection = agent.self_reflect_and_save(
            task=query,
            toolchain_calls=calls,
            reference_answer=reference_answer,
            dag_results={"llm_dag_result": dag_result},
            max_retries=3,
            retry_sleep=retry_sleep,
        )

        predicted_tools = extract_predicted_tools(select_result)
        metrics = evaluate_prediction(predicted_tools, solution)

        result["llm_toolchain_calls"] = calls
        result["llm_dag_result"] = dag_result
        result["predicted_tools"] = predicted_tools
        result["metrics"] = metrics
        result["reflection"] = reflection

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result.setdefault("llm_toolchain_calls", [])
        result.setdefault("llm_dag_result", [])
        result.setdefault("predicted_tools", [])
        result.setdefault("metrics", evaluate_prediction([], solution))
        result.setdefault("reflection", "")

    result["elapsed_sec"] = round(time.time() - start, 3)
    return result


def summarize_results(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not case_results:
        return {
            "total_cases": 0,
            "success_cases": 0,
            "error_cases": 0,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "avg_precision_set": 0.0,
            "avg_recall_set": 0.0,
            "avg_f1_set": 0.0,
        }

    total = len(case_results)
    success = sum(1 for r in case_results if r.get("status") == "success")
    error = total - success

    exact = sum(1 for r in case_results if bool((r.get("metrics") or {}).get("exact_match", False)))

    precision_vals = [float((r.get("metrics") or {}).get("precision_set", 0.0)) for r in case_results]
    recall_vals = [float((r.get("metrics") or {}).get("recall_set", 0.0)) for r in case_results]
    f1_vals = [float((r.get("metrics") or {}).get("f1_set", 0.0)) for r in case_results]

    return {
        "total_cases": total,
        "success_cases": success,
        "error_cases": error,
        "exact_match_count": exact,
        "exact_match_rate": exact / total if total else 0.0,
        "avg_precision_set": sum(precision_vals) / total if total else 0.0,
        "avg_recall_set": sum(recall_vals) / total if total else 0.0,
        "avg_f1_set": sum(f1_vals) / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TMDBAgent on TMDB benchmark set.")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--tool_card_path", default=DEFAULT_TOOL_CARD_PATH)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--max_cases", type=int, default=0, help="0 means all")
    parser.add_argument("--start_index", type=int, default=1, help="1-based start index")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=1.0)
    parser.add_argument("--tool_threshold", type=float, default=0.1)
    parser.add_argument("--notebook_threshold", type=float, default=0.9)
    parser.add_argument(
        "--resume_mode",
        default="skip_done",
        choices=["skip_done", "rerun_all"],
        help='skip_done: skip result file complete;rerun_all:.',
    )
    args = parser.parse_args()

    dataset = load_json_list(args.dataset_path)
    total_all = len(dataset)

    start_idx = max(1, args.start_index)
    if start_idx > total_all:
        raise ValueError(f"start_index {start_idx} is larger than dataset size {total_all}")

    subset = dataset[start_idx - 1 :]
    if args.max_cases and args.max_cases > 0:
        subset = subset[: args.max_cases]

    agent_init_config = {
        "tool_threshold": args.tool_threshold,
        "notebook_threshold": args.notebook_threshold,
    }

    output_path = build_output_path(args.output_path)
    existing_output = load_existing_output(output_path)

    existing_results: List[Dict[str, Any]] = []
    if isinstance(existing_output.get("results"), list):
        existing_results = [x for x in existing_output["results"] if isinstance(x, dict)]

    existing_map: Dict[int, Dict[str, Any]] = {}
    for r in existing_results:
        idx = r.get("index")
        if isinstance(idx, int):
            existing_map[idx] = r

    case_results_map: Dict[int, Dict[str, Any]] = dict(existing_map) if args.resume_mode == "skip_done" else {}

    # initialize agent, load embedding/model
    agent = TMDBAgent(config=agent_init_config)

    bench_start = time.time()
    for i, item in enumerate(subset, start=start_idx):
        if args.resume_mode == "skip_done" and i in case_results_map:
            print(f"[INFO] Skipping done case {i}/{total_all}")
            continue

        print(f"[INFO] Running case {i}/{total_all}: {item.get('query', '')[:120]}")
        case_result = run_single_case(
            idx=i,
            item=item,
            tool_card_path=args.tool_card_path,
            agent=agent,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        case_results_map[i] = case_result

        # save, supports resume
        ordered_results = [case_results_map[k] for k in sorted(case_results_map.keys())]
        summary = summarize_results(ordered_results)
        output = {
            "benchmark_name": "tmdb_tool_routing",
            "dataset_path": os.path.abspath(args.dataset_path),
            "tool_card_path": os.path.abspath(args.tool_card_path),
            "run_started_at": existing_output.get("run_started_at") or datetime.now().isoformat(),
            "last_updated_at": datetime.now().isoformat(),
            "elapsed_sec": round(time.time() - bench_start, 3),
            "agent_config": agent_init_config,
            "summary": summary,
            "results": ordered_results,
        }
        save_output(output_path, output)

    ordered_results = [case_results_map[k] for k in sorted(case_results_map.keys())]
    summary = summarize_results(ordered_results)
    output = {
        "benchmark_name": "tmdb_tool_routing",
        "dataset_path": os.path.abspath(args.dataset_path),
        "tool_card_path": os.path.abspath(args.tool_card_path),
        "run_started_at": existing_output.get("run_started_at") or datetime.now().isoformat(),
        "last_updated_at": datetime.now().isoformat(),
        "elapsed_sec": round(time.time() - bench_start, 3),
        "agent_config": agent_init_config,
        "summary": summary,
        "results": ordered_results,
    }
    save_output(output_path, output)

    print(f"[DONE] Saved benchmark results to: {output_path}")
    print(f"[DONE] Summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
