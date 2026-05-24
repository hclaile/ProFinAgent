import sys
import os
import json
import traceback
import asyncio
import argparse
from datetime import datetime
from typing import Any, Dict, List, Union

# agent path sys. path
_agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "agent"))
if _agent_path not in sys.path:
    sys.path.insert(0, _agent_path)

from agent.test_agent import TestAgent  # type: ignore
import agent.agent_config as agent_config  # type: ignore
from agent.dag import DAGExecutor  # type: ignore
from mcp_server_inline import handle_call_tool  # type: ignore
import re
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
                        paths = re.findall(r'[^\\s<>"]+\\.(: csv|json|parquet)', text)
                        output_paths.extend([p for p in paths if os.path.exists(p)])
    
    #
    return list(set(output_paths))


def extract_json_from_final_answer(final_answer: Union[str, Dict[str, Any], Any]) -> Dict[str, Any]:
    if not final_answer:
        return {}
    
    # dictionary type, direct return
    if isinstance(final_answer, dict):
        return final_answer
    
    # list type, dictionary
    if isinstance(final_answer, list):
        return {"data": final_answer}
    
    # is not, convert
    if not isinstance(final_answer, str):
        final_answer = str(final_answer)
    
    # try direct JSON
    try:
        parsed = json.loads(final_answer)
        # result dictionary, direct return
        if isinstance(parsed, dict):
            return parsed
        # result list, dictionary
        elif isinstance(parsed, list):
            return {"data": parsed}
        # type, dictionary
        else:
            return {"final_answer": parsed}
    except (json.JSONDecodeError, TypeError):
        pass
    
    # try extract JSON
    json_match = re.search(r'\{[^{}]*(:\{[^{}]*\}[^{}]*)*\}', final_answer, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                return parsed
            elif isinstance(parsed, list):
                return {"data": parsed}
            else:
                return {"final_answer": parsed}
        except (json.JSONDecodeError, TypeError):
            pass
    
    # failed, return original dictionary
    return {"final_answer": final_answer}


def test_agent_query(
    agent: TestAgent,
    query: str,
    query_index: int,
    total_queries: int,
    tools_config: Dict[str, Any],
    reference_tools: List[str],
    reference_answer: str,
) -> Dict[str, Any]:
    print('' + "=" * 80)
    print(f"[INFO] (Test Agent) query {query_index}/{total_queries}")
    print(f"[INFO] Query: {query}")
    print("=" * 80)

    result: Dict[str, Any] = {
        "query": query,
        "query_index": query_index,
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "result": None,
        "error": None,
    }

    try:
        print('[INFO] Call Test Agent. Test generate toolchain_calls...')
        llm_output = agent.test(
            question=query,
            tools=tools_config,
            max_retries=5,
            retry_sleep=1.0,
        )
        result["llm_output"] = llm_output

        if isinstance(llm_output, dict) and llm_output.get("error"):
            raise ValueError(f"Test Agent. test return error: {llm_output.get('error')}")

        toolchain_calls = llm_output.get("toolchain_calls", [])
        if not isinstance(toolchain_calls, list):
            raise ValueError("Test Agent. Test return 'toolchain_calls' is not a list.")
        toolchain_calls = [call for call in toolchain_calls if isinstance(call, dict)]

        # tool is empty
        if not toolchain_calls:
            print(f"[INFO] tool is empty, Agent tool available tool")
            result["status"] = "success"
            result["result"] = {"query": query, "tools": [], "arguments": []}
            result["dag_status"] = "no_tools"
            
            print(f"\n" + "=" * 80)
            print(f"[INFO] tool is empty, direct generate final answer")
            print("=" * 80)
            try:
                final_answer = agent.generate_final_answer(
                    task=query,
                    dag_results={},
                    file_contents=[],
                )
                result["final_answer"] = final_answer
                print(f"[SUCCESS] final answer generate complete(: {len(final_answer)})")
                
                reflection = agent.self_reflect_and_save(
                    task=query,
                    toolchain_calls=toolchain_calls,  # use tool
                    final_answer=final_answer,
                    reference_tools=reference_tools,
                    reference_answer=reference_answer,
                    dag_results={},
                    dag_status=result.get("dag_status", "no_tools"),
                    dag_error=None,
                )
                # reflection = ""
                result["self_reflection"] = reflection
            except Exception as e:
                print(f"[ERROR] generate final answer failed: {e}")
                result["final_answer"] = ""
            
            return result

        tools_list = [str(call.get("tool")) for call in toolchain_calls if call.get("tool")]
        arguments_list = [call.get("arguments", {}) for call in toolchain_calls]

        result["status"] = "success"
        print(f"[SUCCESS] Test Agent generate complete, {len(tools_list)} tool call")

        result["result"] = {
            "query": query,
            "tools": tools_list,
            "arguments": arguments_list,
        }

        # analysis dependency DAG
        print(f"\n" + "=" * 80)
        print(f"[INFO] start analyze tool dependency...")
        print("=" * 80)
        
        try:
            # dependency analysis, useempty list
            dependency_result = agent.generate_tool_dependencies(
                toolchain_calls=toolchain_calls,  # use tool
                max_retries=5
            )
            # dependency_result = []
            print(f"[SUCCESS] dependency analysis complete")
            
            # output: dependency analysis result
            print(f"[DEBUG] dependency_result type: {type(dependency_result)}")
            print(f"[DEBUG] dependency_result content: {json.dumps(dependency_result, ensure_ascii=False, indent=2)[:500]}")
            
            dag_task_list = []
            
            # dependency analysis returned result
            if isinstance(dependency_result, list) and len(dependency_result) > 0:
                print(f"[INFO] dependency analysis returned {len(dependency_result)} task")
                for task in dependency_result:
                    if not isinstance(task, dict):
                        print(f"[WARN] skip dictionarytask: {task}")
                        continue
                    
                    # : model return tool_name tool
                    tool_name = task.get("tool_name") or task.get("tool") or ""
                    task_id = task.get("id")
                    
                    # missing field, skip
                    if not tool_name:
                        print(f"[WARN] task missing tool_name/tool field, skip: {task}")
                        continue
                    if task_id is None:
                        print(f"[WARN] task missing id field, skip: {task}")
                        continue
                    
                    dag_task_list.append({
                        "id": task_id,
                        "tool_name": str(tool_name),
                        "arguments": task.get("arguments") if isinstance(task.get("arguments"), dict) else {},
                        "dependencies": task.get("dependencies") if isinstance(task.get("dependencies"), list) else [],
                    })
            elif isinstance(dependency_result, list) and len(dependency_result) == 0:
                print(f"[WARN] dependency analysis returned an empty list, use: dependency")
                # : dependency analysis returned an empty list, use dependency
                # use tool
                for idx, call in enumerate(toolchain_calls, 1):
                    if not isinstance(call, dict):
                        continue
                    tool_name = call.get("tool")
                    call_args = call.get("arguments", {})
                    if not tool_name:
                        continue
                    
                    dag_task_list.append({
                        "id": idx,
                        "tool_name": str(tool_name),
                        "arguments": call_args if isinstance(call_args, dict) else {},
                        "dependencies": [idx - 1] if idx > 1 else [],
                    })
            else:
                print(f"[WARN] dependency_result is not list, type: {type(dependency_result)}")
                # : directly use toolchain_calls dependency
                for idx, call in enumerate(toolchain_calls, 1):
                    if not isinstance(call, dict):
                        continue
                    tool_name = call.get("tool")
                    call_args = call.get("arguments", {})
                    if not tool_name:
                        continue
                    
                    dag_task_list.append({
                        "id": idx,
                        "tool_name": str(tool_name),
                        "arguments": call_args if isinstance(call_args, dict) else {},
                        "dependencies": [idx - 1] if idx > 1 else [],
                    })
            
            print(f"[INFO] {len(dag_task_list)} DAG task")
            
            if dag_task_list:
                print(f"[INFO] DAG task list:")
                for task in dag_task_list:
                    print(f"- Task {task['id']}: {task['tool_name']} (dependency: {task['dependencies']})")

                async def run_tool(task: dict):
                    tool = task.get("tool_name")
                    args = task.get("arguments", {}) or {}
                    print(f"[INFO] tool: {tool} (Task ID: {task['id']})")
                    try:
                        output = await handle_call_tool(tool, args)
                        return {
                            "status": "success",
                            "tool": tool,
                            "args": args,
                            "task_id": task["id"],
                            "output": output,
                        }
                    except Exception as e:
                        print(f"[ERROR] tool {tool} exception: {e}")
                        return {
                            "status": "error",
                            "tool": tool,
                            "args": args,
                            "task_id": task["id"],
                            "output": str(e),
                        }
                
                executor = DAGExecutor(
                    task_list=dag_task_list,
                    run_tool=run_tool,
                    agent_instance=agent
                )
                
                # DAG
                print(f"[INFO] start DAG...")
                dag_results = asyncio.run(executor.run())
                result["dag_results"] = dag_results
                result["dag_status"] = "success"
                print(f"[SUCCESS] DAG complete")

                # read file content generate answer(use)
                file_contents = agent.collect_existing_file_contents(
                    dag_results,
                    task=query,  # querytask related
                    max_chars=10000,  # file
                    use_llamaindex=True,  # use Llama Index extract
                )
                if file_contents:
                    print(f"[INFO] extract {len(file_contents)} related file content")

                # generate final answer
                print(f"[INFO] generate final answer...")
                final_answer = agent.generate_final_answer(
                    task=query,
                    dag_results=dag_results,
                    file_contents=file_contents,
                )
                result["final_answer"] = final_answer
                print(f"[SUCCESS] final answer generate complete(: {len(final_answer)})")
                
                # reflection
                reflection = agent.self_reflect_and_save(
                    task=query,
                    toolchain_calls=toolchain_calls,  # use tool
                    final_answer=final_answer,
                    reference_tools=reference_tools,
                    reference_answer=reference_answer,
                    dag_results=dag_results,
                    dag_status=result.get("dag_status", "success"),
                    dag_error=result.get("dag_error"),
                )
                # reflection = ""
                result["self_reflection"] = reflection
            else:
                print(f"[WARN] DAG task, try direct generate answer")
                result["dag_status"] = "no_tasks"
                try:
                    final_answer = agent.generate_final_answer(query, {}, [])
                    result["final_answer"] = final_answer
                    print(f"[SUCCESS] final answer generate complete(: {len(final_answer)})")
                    
                    # tool, reflection
                    reflection = agent.self_reflect_and_save(
                        task=query,
                        toolchain_calls=toolchain_calls,  # use tool
                        final_answer=final_answer,
                        reference_tools=reference_tools,
                        reference_answer=reference_answer,
                        dag_results={},
                        dag_status=result.get("dag_status", "no_tasks"),
                        dag_error=None,
                    )
                    # reflection = ""
                    result["self_reflection"] = reflection
                except Exception as e:
                    print(f"[ERROR] generate final answerreflection failed: {e}")
                    result["final_answer"] = ""

                
        except Exception as dag_exc:
            print(f"[ERROR] DAG failed: {dag_exc}")
            print(f"[ERROR] errortype: {type(dag_exc).__name__}")
            traceback.print_exc()
            result["dag_status"] = "error"
            result["dag_error"] = str(dag_exc)
            try:
                # DAG failed, try generate final answerreflection
                final_answer = agent.generate_final_answer(query, {}, [])
                result["final_answer"] = final_answer
                print(f"[SUCCESS] final answer generate complete(: {len(final_answer)})")
                
                # DAG failed, reflection
                reflection = agent.self_reflect_and_save(
                    task=query,
                    toolchain_calls=toolchain_calls,  # use tool
                    final_answer=final_answer,
                    reference_tools=reference_tools,
                    reference_answer=reference_answer,
                    dag_results=result.get("dag_results", {}),
                    dag_status=result.get("dag_status", "error"),
                    dag_error=result.get("dag_error"),
                )
                # reflection = ""
                result["self_reflection"] = reflection
            except Exception as e:
                print(f"[ERROR] generate final answerreflection failed: {e}")
                result["final_answer"] = ""

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if "final_answer" not in result:
            result["final_answer"] = ""
        print(f"[ERROR] failed: {e}")
        traceback.print_exc()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Resumable benchmark runner for Test Agent (with tools + report + notebook)')
    parser.add_argument("--benchmark_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/benchmark.json')))
    parser.add_argument("--benchmark_true_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/benchmark.json')))
    parser.add_argument("--tools_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs/tools.json')))
    parser.add_argument("--results_path", default="", help='result file path(.json.jsonl). exists resume file append or update.')
    parser.add_argument("--resume_mode", default="skip_success_and_error", choices=["skip_success", "skip_success_and_error", "skip_done", "retry_errors"])
    parser.add_argument("--start", type=int, default=1, help="1-based start index")
    parser.add_argument("--limit", type=int, default=0, help="max cases to run (0 means all)")
    args = parser.parse_args()

    # load data
    benchmark_path = os.path.abspath(args.benchmark_path)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    benchmark_true_path = os.path.abspath(args.benchmark_true_path)
    benchmark_true_map = {}
    if os.path.exists(benchmark_true_path):
        with open(benchmark_true_path, "r", encoding="utf-8") as f:
            true_list = json.load(f)
            for item in true_list:
                if item.get("query"):
                    benchmark_true_map[item["query"]] = item

    tools_path = os.path.abspath(args.tools_path)
    with open(tools_path, "r", encoding="utf-8") as f:
        tools_config = json.load(f)

    # initialize Test Agent
    agent = TestAgent()
    print('[SUCCESS] Test Agent initialize complete')

    # Resumable results (JSON: notebook file update; JSONL: append)
    output_dir = os.path.join(os.path.dirname(__file__), "Result", "TestAgent")
    os.makedirs(output_dir, exist_ok=True)
    if args.results_path:
        results_path = os.path.abspath(args.results_path)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"deepseek_tool0.2_results_{ts}.json")
    progress = BenchmarkProgress(results_path=results_path, resume_mode=args.resume_mode, run_name="test_agent_tool")
    existing = progress.stats()
    if existing.total_cases_seen:
        print(f"[INFO] Resume enabled: already have {existing.total_cases_seen} cases"
              f"(success={existing.success_cases}, error={existing.error_cases}) in {progress.results_path}")
    progress.start_run({"script": "test_agent_tool", "benchmark_path": benchmark_path})

    total_expected_slots = 0
    total_hit_slots = 0
    evaluated_queries = 0

    start_idx = max(1, int(args.start))
    end_idx = len(benchmark_data)
    if args.limit and args.limit > 0:
        end_idx = min(end_idx, start_idx + args.limit - 1)

    test_start_dt = datetime.now()

    for idx, benchmark_item in enumerate(benchmark_data, 1):
        if idx < start_idx or idx > end_idx:
            continue
        query = benchmark_item.get("query", "")
        if not query: continue

        if progress.should_skip(query):
            # success
            # overall_recall statistics, expected slots
            raw_expected = benchmark_item.get("expected_tool_types", [])
            expected_groups = []
            for entry in raw_expected:
                if isinstance(entry, list): expected_groups.append(entry)
                elif entry: expected_groups.append([entry])
            total_expected_slots += len(expected_groups)
            evaluated_queries += 1
            continue

        raw_expected = benchmark_item.get("expected_tool_types", [])
        expected_groups = []
        for entry in raw_expected:
            if isinstance(entry, list): expected_groups.append(entry)
            elif entry: expected_groups.append([entry])
        
        expected_count = len(expected_groups)
        total_expected_slots += expected_count

        ref_entry = benchmark_true_map.get(query, {})
        reference_tools = ref_entry.get("expected_tool_types", []) or []
        reference_answer = ref_entry.get("execution_result", "") or ""

        try:
            result = test_agent_query(
                agent, query, idx, len(benchmark_data),
                tools_config, reference_tools, reference_answer
            )

            status = result.get("status", "error")
            final_result_item = result.get("result", {"query": query, "tools": [], "arguments": []})
            predicted_tools = final_result_item.get("tools", []) or []
            final_answer = result.get("final_answer", "")

            hit_count = 0
            if expected_count > 0:
                predicted_set = set(predicted_tools)
                for group in expected_groups:
                    if any(t in predicted_set for t in group): hit_count += 1
                recall = hit_count / expected_count
            else:
                recall = 1.0 if status == "success" else 0.0

            total_hit_slots += hit_count
            evaluated_queries += 1
            success_flag = (hit_count == expected_count) if expected_count > 0 else (status == "success")
            progress.append({
                "query": query,
                "query_index": idx,
                "status": status,
                "tools": predicted_tools,
                "arguments": final_result_item.get("arguments", []),
                "final_answer": final_answer,
                "error": result.get("error"),
                "recall": recall,
                "success": success_flag,
                "expected_slots": expected_count,
                "hit_slots": hit_count,
            })

        except Exception as e:
            print(f"[ERROR] {e}")
            evaluated_queries += 1
            progress.append({
                "query": query,
                "query_index": idx,
                "status": "exception",
                "error": str(e),
                "expected_slots": expected_count,
                "hit_slots": 0,
                "recall": 0.0,
                "success": False,
            })

    # finish + duration result file
    test_end_dt = datetime.now()
    duration_seconds = (test_end_dt - test_start_dt).total_seconds()
    progress.finish_run({"last_duration_seconds": duration_seconds})

    all_latest = progress.iter_latest_records()
    success_count = sum(1 for r in all_latest if bool(r.get("success")) or str(r.get("status", "")).lower() == "success")
    print(f"\n[SUCCESS] complete, result save: {progress.results_path}")


if __name__ == "__main__":
    main()
