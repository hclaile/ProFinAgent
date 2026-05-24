import sys
import os
import json
import traceback
import asyncio
import argparse
from datetime import datetime
from typing import Any, Dict, List

# agent path sys. path
_agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "agent"))
if _agent_path not in sys.path:
    sys.path.insert(0, _agent_path)

from agent.base_agent import BaseAgent  # type: ignore
import agent.agent_config as agent_config  # type: ignore
from mcp_server_inline import handle_call_tool  # type: ignore
from utils.benchmark_progress import BenchmarkProgress  # type: ignore


def collect_output_paths(dag_results: Dict[str, Any]) -> List[str]:
    output_paths = []
    
    for task_id, task_result in dag_results.items():
        if not isinstance(task_result, dict):
            continue
        
        # task_result args extract output_path
        args = task_result.get("args", {})
        if isinstance(args, dict):
            output_path = args.get("output_path")
            if output_path and isinstance(output_path, str) and os.path.exists(output_path):
                output_paths.append(output_path)
        
        # task_result output extract output_path(tool output return path)
        output = task_result.get("output", [])
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and "text" in item:
                    # try extract path
                    text = item.get("text", "")
                    if "output_path" in text or '.csv' in text or '.json' in text:
                        # extract: pathmode
                        import re
                        paths = re.findall(r'[^\\s<>"]+\\.(: csv|json|parquet)', text)
                        output_paths.extend([p for p in paths if os.path.exists(p)])
    
    #
    return list(set(output_paths))


def extract_json_from_final_answer(final_answer: str) -> Dict[str, Any]:
    'final_answer extract JSON content. Args: final_answer: final answer Returns: Dict[str, Any]: extract JSON dictionary, extract failed; return final_answer dictionary'
    if not final_answer:
        return {}
    
    # try direct JSON
    try:
        return json.loads(final_answer)
    except json.JSONDecodeError:
        pass
    
    # try extract JSON
    import re
    json_match = re.search(r'\{[^{}]*(:\{[^{}]*\}[^{}]*)*\}', final_answer, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # failed, return original dictionary
    return {"final_answer": final_answer}


async def execute_tool_sequentially(
    toolchain_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    dag_results: Dict[str, Any] = {}
    
    for idx, call in enumerate(toolchain_calls, 1):
        if not isinstance(call, dict):
            print(f"[WARN] toolchain_calls[{idx-1}] is not, skip")
            continue
        
        tool_name = call.get("tool", "")
        arguments = call.get("arguments", {})
        
        if not tool_name:
            print(f"[WARN] toolchain_calls[{idx-1}] missing 'tool' field, skip")
            continue
        
        task_id = str(idx)
        print(f"\n[INFO] tool {idx}/{len(toolchain_calls)}: {tool_name}")
        print(f"[INFO] tool parameter: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
        
        try:
            output = await handle_call_tool(tool_name, arguments)
            dag_results[task_id] = {
                "status": "success",
                "tool": tool_name,
                "args": arguments,
                "task_id": task_id,
                "output": output,
            }
            print(f"[SUCCESS] tool {tool_name} success")
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] tool {tool_name} failed: {error_msg}")
            dag_results[task_id] = {
                "status": "error",
                "tool": tool_name,
                "args": arguments,
                "task_id": task_id,
                "output": error_msg,
            }
    
    return dag_results


def test_straightforward_query(
    agent: BaseAgent,
    query: str,
    query_index: int,
    total_queries: int,
    tools_config: Dict[str, Any],
) -> Dict[str, Any]:
    'use Base Agent.test_straightforward benchmark query. Args: agent: Base Agent query: user query query_index: query index starts at 1 total_queries: query tools_config: tools.json config Returns: dict: query, tool, result final answer'
    print('' + "=" * 80)
    print(f"[INFO] query {query_index}/{total_queries}")
    print(f"[INFO] Query: {query}")
    print("=" * 80)

    result: Dict[str, Any] = {
        "query": query,
        "query_index": query_index,
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "final_answer": "",
        "error": None,
    }

    try:
        # 1: use test_straightforward get tool
        print('[INFO] 1: call Base Agent.test_straightforward generate tool...')
        llm_output = agent.test_straightforward(
            question=query,
            tools=tools_config,
            max_retries=5,
            retry_sleep=1.0,
        )
        
        if isinstance(llm_output, dict) and llm_output.get("error"):
            raise ValueError(f"Base Agent.test_straightforward return error: {llm_output.get('error')}")

        toolchain_calls = llm_output.get("toolchain_calls", []) if isinstance(llm_output, dict) else []
        if not isinstance(toolchain_calls, list):
            raise ValueError("Base Agent.test_straightforward return 'toolchain_calls' is not a list.")
        toolchain_calls = [call for call in toolchain_calls if isinstance(call, dict)]

        # 2: tool is empty, direct generate final answer
        if not toolchain_calls or len(toolchain_calls) == 0:
            print(f"[INFO] tool is empty, direct generate final answer(use tool)")
            try:
                final_answer = agent.generate_final_answer(
                    task=query,
                    dag_results={},
                    file_contents=[],
                )
                result["status"] = "success"
                result["final_answer"] = final_answer
                print(f"[SUCCESS] final answer generate complete(: {len(final_answer)})")
                
            except Exception as e:
                result["status"] = "error"
                result["error"] = f"generate final answer failed: {str(e)}"
                print(f"[ERROR] generate final answer failed: {e}")
            return result

        # 3: execute planned tools
        print(f"\n[INFO] 2: toolchain ({len(toolchain_calls)} tool)...")
        dag_results = asyncio.run(execute_tool_sequentially(toolchain_calls))
        
        print(f"\n[SUCCESS] tool complete")
        print(f"[INFO] result: {len(dag_results)} task")

        # 4: tool output file content(use)
        print(f"\n[INFO] 3: tool output file content...")
        file_contents = agent.collect_existing_file_contents(
            dag_results,
            # task=query, # querytask related
            max_chars=10000,
            # use_llamaindex=True, # use Llama Index extract
        )
        print(f"[INFO] {len(file_contents)} file")

        # 5: generate final answer
        print(f"\n[INFO] 4: generate final answer...")
        try:
            final_answer = agent.generate_final_answer(
                task=query,
                dag_results=dag_results,
                file_contents=file_contents,
            )
            result["status"] = "success"
            result["final_answer"] = final_answer
            print(f"[SUCCESS] final answer generate complete(: {len(final_answer)})")
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = f"generate final answer failed: {str(e)}"
            print(f"[ERROR] generate final answer failed: {e}")
            traceback.print_exc()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"[ERROR] failed: {e}")
        traceback.print_exc()

    return result


def main() -> None:
    'Base Agent.test_straightforward.'
    # save dag_memory.json
    BaseAgent.SAVE_TO_DAG_MEMORY = False
    print(f"[INFO] SAVE_TO_DAG_MEMORY False, save dag_memory.json")
    
    # read benchmark.json
    benchmark_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'dataset/benchmark.json')
    )

    if not os.path.exists(benchmark_path):
        print(f"[ERROR] benchmark.json file does not exist: {benchmark_path}")
        sys.exit(1)

    print(f"[INFO] read benchmark.json: {benchmark_path}")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    total_queries = len(benchmark_data)
    print(f"[INFO] {total_queries} query")

    # load tools.json
    tools_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'configs/tools.json')
    )
    if not os.path.exists(tools_config_path):
        print(f"[ERROR] tools.json file does not exist: {tools_config_path}")
        sys.exit(1)
    
    print(f"[INFO] read tools.json: {tools_config_path}")
    with open(tools_config_path, "r", encoding="utf-8") as f:
        tools_config = json.load(f)

    # record start time
    test_start_dt = datetime.now()
    test_start_time = test_start_dt.isoformat()

    # Base Agent(initialize RAG, use embedding)
    print('[INFO] initialize Base Agent(use RAG embedding)...')
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
        embedding_model_path=None,  # use embedding
        chroma_db_path=None,  # use RAG
        chroma_collection_name=None,
        matryoshka_dim=None,
    )
    print('[SUCCESS] Base Agent initialize complete')

    parser = argparse.ArgumentParser(description='Resumable benchmark runner for Base Agent.test_straightforward')
    parser.add_argument("--benchmark_path", default=benchmark_path)
    parser.add_argument("--results_path", default="", help='result file path(.json.jsonl). exists resume file append or update.')
    parser.add_argument("--resume_mode", default="skip_success_and_error", choices=["skip_success", "skip_success_and_error", "skip_done", "retry_errors"])
    parser.add_argument("--start", type=int, default=1, help="1-based start index")
    parser.add_argument("--limit", type=int, default=0, help="max cases to run (0 means all)")
    args = parser.parse_args()

    # resume result file(JSON: notebook file update; JSONL: append)
    output_dir = os.path.join(os.path.dirname(__file__), "Result", "Straightforward")
    os.makedirs(output_dir, exist_ok=True)
    if args.results_path:
        results_path = os.path.abspath(args.results_path)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"straightforward_benchmark_results_{ts}.json")
    progress = BenchmarkProgress(results_path=results_path, resume_mode=args.resume_mode, run_name="myagent_tool_straight")
    existing = progress.stats()
    if existing.total_cases_seen:
        print(f"[INFO] Resume enabled: already have {existing.total_cases_seen} cases"
              f"(success={existing.success_cases}, error={existing.error_cases}) in {progress.results_path}")
    progress.start_run({"script": "myagent_tool_straight", "benchmark_path": benchmark_path})

    evaluated_queries = 0
    start_idx = max(1, int(args.start))
    end_idx = total_queries
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
            evaluated_queries += 1
            continue

        try:
            result = test_straightforward_query(
                agent,
                query,
                idx,
                total_queries,
                tools_config,
            )
            evaluated_queries += 1

            progress.append({
                "query": query,
                "query_index": idx,
                "final_answer": result.get("final_answer", ""),
                "status": result.get("status", "error"),
                "error": result.get("error"),
            })

        except Exception as e:  # , exception
            print(f"[ERROR] {idx} query exception: {e}")
            traceback.print_exc()
            evaluated_queries += 1
            progress.append({
                "query": query,
                "query_index": idx,
                "final_answer": "",
                "status": "exception",
                "error": str(e),
            })

    # record time
    test_end_dt = datetime.now()
    test_end_time = test_end_dt.isoformat()
    duration_seconds = (test_end_dt - test_start_dt).total_seconds()

    all_latest = progress.iter_latest_records()
    success_count = sum(1 for r in all_latest if str(r.get("status", "")).lower() == "success")
    error_count = sum(1 for r in all_latest if str(r.get("status", "")).lower() != "success")
    progress.finish_run(
        {
            "last_test_start_time": test_start_time,
            "last_test_end_time": test_end_time,
            "last_duration_seconds": duration_seconds,
        }
    )

    print('' + "=" * 80)
    print('[SUCCESS] Base Agent.test_straightforward Benchmark complete')
    print(f"[INFO] query(): {len(all_latest)}")
    print(f"[INFO] success: {success_count}")
    print(f"[INFO] failed: {error_count}")
    print(f"[INFO]: {duration_seconds:.2f}")
    print(f"[INFO] result save: {progress.results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
