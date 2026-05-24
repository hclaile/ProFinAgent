import sys
import os
import json
import traceback
import argparse
from datetime import datetime
from typing import Any, Dict, List

# agent path sys. path
_agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "agent"))
if _agent_path not in sys.path:
    sys.path.insert(0, _agent_path)

from agent.base_agent import BaseAgent  # type: ignore
import agent.agent_config as agent_config  # type: ignore
from utils.benchmark_progress import BenchmarkProgress  # type: ignore


def test_no_tool_query(
    agent: BaseAgent,
    query: str,
    query_index: int,
    total_queries: int,
) -> Dict[str, Any]:
    print('' + "=" * 80)
    print(f"[INFO] query {query_index}/{total_queries}")
    print(f"[INFO] Query: {query}")
    print("=" * 80)

    result: Dict[str, Any] = {
        "query": query,
        "query_index": query_index,
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "answer": "",
        "error": None,
    }

    try:
        print('[INFO] call Base Agent.no_tool_answer generate answer...')
        answer = agent.no_tool_answer(
            task=query,
            max_retries=3,
        )

        if answer:
            result["status"] = "success"
            result["answer"] = answer
            print(f"[SUCCESS] success generate answer(: {len(answer)})")
            print(f"[INFO] answer: {answer[:200]}...")
        else:
            result["status"] = "error"
            result["error"] = 'Agent return empty answer'
            print(f"[WARN] Agent return empty answer")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"[ERROR] failed: {e}")
        traceback.print_exc()

    return result


def main() -> None:
    'Base Agent.no_tool_answer.'
    # save dag_memory.json
    BaseAgent.SAVE_TO_DAG_MEMORY = False
    print(f"[INFO] SAVE_TO_DAG_MEMORY False, save dag_memory.json")
    
    parser = argparse.ArgumentParser(description='Resumable benchmark runner for Base Agent.no_tool_answer')
    parser.add_argument("--benchmark_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/benchmark.json')))
    parser.add_argument("--results_path", default="", help='result file path(.json.jsonl). exists resume file append or update.')
    parser.add_argument("--resume_mode", default="skip_success_and_error", choices=["skip_success", "skip_success_and_error", "skip_done", "retry_errors"])
    parser.add_argument("--start", type=int, default=1, help="1-based start index")
    parser.add_argument("--limit", type=int, default=0, help="max cases to run (0 means all)")
    args = parser.parse_args()

    # read benchmark.json
    benchmark_path = os.path.abspath(args.benchmark_path)

    if not os.path.exists(benchmark_path):
        print(f"[ERROR] benchmark.json file does not exist: {benchmark_path}")
        sys.exit(1)

    print(f"[INFO] read benchmark.json: {benchmark_path}")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    total_queries = len(benchmark_data)
    print(f"[INFO] {total_queries} query")

    # record start time
    test_start_dt = datetime.now()
    test_start_time = test_start_dt.isoformat()

    # Base Agent
    print('[INFO] initialize Base Agent...')
    agent = BaseAgent(
        base_url=agent_config.BASE_URL,
        model_name=agent_config.MODEL_NAME,
        model_path=agent_config.MODEL_PATH,
        api_key=agent_config.API_KEY,
        temperature=agent_config.TEMPERATURE,
        max_tokens=agent_config.MAX_TOKENS,
        timeout=agent_config.TIMEOUT,
        auto_load=agent_config.AUTO_LOAD,
        system_prompt=agent_config.SYSTEM_PROMPT,
        embedding_model_path=agent_config.EMBEDDING_MODEL_PATH,
        chroma_db_path=agent_config.CHROMA_DB_PATH,
        chroma_collection_name=agent_config.CHROMA_COLLECTION_NAME,
        matryoshka_dim=agent_config.MATRYOSHKA_DIM,
    )
    print('[SUCCESS] Base Agent initialize complete')

    # resume result file(JSON: notebook file update; JSONL: append)
    output_dir = os.path.join(os.path.dirname(__file__), "Result", "NoTool")
    os.makedirs(output_dir, exist_ok=True)
    if args.results_path:
        results_path = os.path.abspath(args.results_path)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"no_tool_benchmark_results_{ts}.json")
    progress = BenchmarkProgress(results_path=results_path, resume_mode=args.resume_mode, run_name="myagent_no_tool")
    existing = progress.stats()
    if existing.total_cases_seen:
        print(f"[INFO] Resume enabled: already have {existing.total_cases_seen} cases"
              f"(success={existing.success_cases}, error={existing.error_cases}) in {progress.results_path}")
    progress.start_run({"script": "myagent_no_tool", "benchmark_path": benchmark_path})

    # ()
    evaluated_queries = 0
    start_idx = max(1, int(args.start))
    end_idx = len(benchmark_data)
    if args.limit and args.limit > 0:
        end_idx = min(end_idx, start_idx + args.limit - 1)

    for idx, benchmark_item in enumerate(benchmark_data, 1):
        if idx < start_idx or idx > end_idx:
            continue
        query = benchmark_item.get("query", "")
        if not query:
            print(f"[WARN] skip {idx} query(query is empty)")
            continue

        if progress.should_skip(query):
            # complete
            evaluated_queries += 1
            continue

        try:
            result = test_no_tool_query(
                agent,
                query,
                idx,
                total_queries,
            )
            evaluated_queries += 1
            record = {
                "query": query,
                "query_index": idx,
                "answer": result.get("answer", ""),
                "status": result.get("status", "error"),
                "error": result.get("error"),
            }
            progress.append(record)

        except Exception as e:  # , exception
            print(f"[ERROR] {idx} query exception: {e}")
            traceback.print_exc()
            evaluated_queries += 1
            progress.append({
                "query": query,
                "query_index": idx,
                "answer": "",
                "status": "exception",
                "error": str(e),
            })

    # record time
    test_end_dt = datetime.now()
    test_end_time = test_end_dt.isoformat()
    duration_seconds = (test_end_dt - test_start_dt).total_seconds()

    progress.finish_run(
        {
            "last_test_start_time": test_start_time,
            "last_test_end_time": test_end_time,
            "last_duration_seconds": duration_seconds,
        }
    )
    all_latest = progress.iter_latest_records()
    success_count = sum(1 for r in all_latest if str(r.get("status", "")).lower() == "success")
    error_count = len(all_latest) - success_count

    print('' + "=" * 80)
    print('[SUCCESS] Base Agent.no_tool_answer Benchmark complete')
    print(f"[INFO] query(): {len(all_latest)}")
    print(f"[INFO] success: {success_count}")
    print(f"[INFO] failed: {error_count}")
    print(f"[INFO]: {duration_seconds:.2f}")
    print(f"[INFO] result save: {progress.results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

