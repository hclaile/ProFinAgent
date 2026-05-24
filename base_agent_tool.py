import sys
import os
import json
import traceback
import asyncio
import argparse
from datetime import datetime
from typing import Any, Dict, List, Union

# agent path sys. path(test_benchmark.py)
_agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "agent"))
if _agent_path not in sys.path:
    sys.path.insert(0, _agent_path)

from agent.base_agent import BaseAgent  # type: ignore
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
    agent: BaseAgent,
    query: str,
    query_index: int,
    total_queries: int,
    tools_config: Dict[str, Any],
    reference_tools: List[str],
    reference_answer: str,
) -> Dict[str, Any]:
    'use Base Agent test benchmark query. Args: agent: Base Agent query: user query query_index: query index starts at 1 total_queries: query tools_config: tools.json config(complete) Returns: dict: query, LLM output tool list'
    print('' + "=" * 80)
    print(f"[INFO] (Base Agent test) query {query_index}/{total_queries}")
    print(f"[INFO] Query: {query}")
    print("=" * 80)

    result: Dict[str, Any] = {
        "query": query,
        "query_index": query_index,
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "result": None,  # final result: query, use tool, tool parameter
        "error": None,
    }

    try:
        print('[INFO] call Base Agent test generate toolchain_calls...')
        llm_output = agent.test(
            question=query,
            tools=tools_config,
            max_retries=5,
            retry_sleep=1.0,
        )
        result["llm_output"] = llm_output

        # Base Agent test return error field, failed
        if isinstance(llm_output, dict) and llm_output.get("error"):
            raise ValueError(f"Base Agent test return error: {llm_output.get('error')}")

        toolchain_calls = llm_output.get("toolchain_calls", []) if isinstance(
            llm_output, dict
        ) else []
        if not isinstance(toolchain_calls, list):
            raise ValueError("Base Agent test return 'toolchain_calls' is not a list.")
        toolchain_calls = [call for call in toolchain_calls if isinstance(call, dict)]

        # tool is empty: is empty, description agent tool available tool
        if not toolchain_calls or len(toolchain_calls) == 0:
            print(f"[INFO] tool is empty, Agent tool available tool")
            result["status"] = "success"
            tools_list = [str(call.get("tool")) for call in toolchain_calls if call.get("tool")]
            arguments_list = [call.get("arguments", {}) for call in toolchain_calls]
            result["result"] = {
                "query": query,
                "tools": tools_list,
                "arguments": arguments_list,
            }
            result["dag_status"] = "no_tools"
            
            # directcall generate_final_answer, empty dag_results file_contents
            print(f"\n" + "=" * 80)
            print(f"[INFO] tool is empty, direct generate final answer(use tool)")
            print("=" * 80)
            try:
                final_answer = agent.generate_final_answer(
                    task=query,
                    dag_results={},  # emptydictionary
                    file_contents=[],  # empty list
                )
                result["final_answer"] = final_answer
                print(f"[SUCCESS] final answer generate complete(: {len(final_answer)})")
                
                # reflection notebook(tool call)
                # reflection = agent.self_reflect_and_save(
                # task=query,
                # toolchain_calls=toolchain_calls, # use tool
                # final_answer=final_answer,
                # reference_tools=reference_tools,
                # reference_answer=reference_answer,
                # dag_results={},
                # dag_status=result. get("dag_status", "no_tools"),
                # dag_error=None,
                # )
                reflection = ""
                result["self_reflection"] = reflection
            except Exception as final_answer_exc:
                print(f"[ERROR] generate final answer failed: {final_answer_exc}")
                result["final_answer"] = ""
                result["final_answer_error"] = str(final_answer_exc)
            
            return result

        # tool is empty, continue tool call
        tools_list: List[str] = []
        arguments_list: List[Dict[str, Any]] = []

        for idx, call in enumerate(toolchain_calls):
            if not isinstance(call, dict):
                print(
                    f"[WARN] toolchain_calls[{idx}] is not, skip: {repr(call)}"
                )
                continue
            tool_name = call.get("tool")
            call_args = call.get("arguments", {})
            if not tool_name:
                print(
                    f"[WARN] toolchain_calls[{idx}] missing 'tool' field, skip: {repr(call)}"
                )
                continue
            tools_list.append(str(tool_name))
            if isinstance(call_args, dict):
                arguments_list.append(call_args)
            else:
                # dict keep originalstructure,
                arguments_list.append({"_raw": call_args})

        result["status"] = "success"
        print(f"[SUCCESS] Base Agent test generate complete, {len(tools_list)} tool call")

        # finalresult format: query, use tool, tool parameter
        result["result"] = {
            "query": query,
            "tools": tools_list,
            "arguments": arguments_list,
        }

        # success generate tool, use DAG tool
        if tools_list and len(tools_list) > 0:
            print(f"\n" + "=" * 80)
            print(f"[INFO] start analyze tool dependency...")
            print("=" * 80)
            
            try:
                # use Base Agent.generate_tool_dependencies analyze tool dependency
                print(f"[INFO] call Base Agent.generate_tool_dependencies analysis dependency...")
                dependency_result = agent.generate_tool_dependencies(
                    toolchain_calls=toolchain_calls,  # use tool
                    max_retries=5
                )
                print(f"[SUCCESS] dependency analysis complete, {len(dependency_result)} task")
                
                # output dependency summary
                print(f"[INFO] dependency summary:")
                for task in dependency_result:
                    deps = task.get("dependencies", [])
                    tool_name = task.get("tool_name", "Unknown")
                    task_id = task.get("id", "N/A")
                    if deps:
                        print(f"- Task {task_id} ({tool_name}) dependency: {deps}")
                    else:
                        print(f"- Task {task_id} ({tool_name}) dependency(parallel)")
                
                # analysis result convert DAGExecutor format
                dag_task_list: List[Dict[str, Any]] = []
                for task in dependency_result:
                    task_id = task.get("id")
                    tool_name = task.get("tool_name", "")
                    arguments = task.get("arguments", {})
                    dependencies = task.get("dependencies", [])
                    
                    if not tool_name:
                        print(f"[WARN] task {task_id} missing tool_name, skip")
                        continue
                    
                    dag_task = {
                        "id": task_id,
                        "tool_name": str(tool_name),
                        "arguments": arguments if isinstance(arguments, dict) else {},
                        "dependencies": dependencies if isinstance(dependencies, list) else [],
                    }
                    dag_task_list.append(dag_task)
                
                if dag_task_list:
                    print(f"\n" + "=" * 80)
                    print(f"[INFO] startuse DAGExecutor tool...")
                    print("=" * 80)
                    print(f"[INFO] {len(dag_task_list)} DAG task")
                    print(f"[INFO] DAG task list:")
                    for task in dag_task_list:
                        print(f"- Task {task['id']}: {task['tool_name']} (dependencies: {task['dependencies']})")
                    
                    # tool
                    async def run_tool(task: dict):
                        tool = task.get("tool_name")
                        args = task.get("arguments", {}) or {}
                        print(f"[INFO] tool: {tool} (Task ID: {task['id']})")
                        print(f"[INFO] tool parameter: {json.dumps(args, ensure_ascii=False, indent=2)}")
                        # direct mcp_server_inline handle_call_tool, MCP tool
                        output = await handle_call_tool(tool, args)
                        return {
                            "status": "success",
                            "tool": tool,
                            "args": args,
                            "task_id": task["id"],
                            "output": output,
                        }
                    
                    # use DAGExecutor tool
                    executor = DAGExecutor(
                        task_list=dag_task_list,
                        run_tool=run_tool,
                        agent_instance=agent  # agent, save full_task dag_memory.json
                    )
                    dag_results = asyncio.run(executor.run())
                    
                    print(f"\n[SUCCESS] DAG complete")
                    print(f"[INFO] DAG result:")
                    print(json.dumps(dag_results, ensure_ascii=False, indent=2, default=str))
                    
                    # DAG result result
                    result["dag_results"] = dag_results
                    result["dag_status"] = "success"

                    # read DAG output file content(use)
                    file_contents = agent.collect_existing_file_contents(
                        dag_results,
                        task=query,  # querytask related
                        max_chars=10000,  # file
                        use_llamaindex=True,  # use Llama Index extract
                    )

                    # generate final answer
                    final_answer = agent.generate_final_answer(
                        task=query,
                        dag_results=dag_results,
                        file_contents=file_contents,
                    )
                    result["final_answer"] = final_answer
                    
                    # reflection notebook
                    # reflection = agent.self_reflect_and_save(
                    # task=query,
                    # toolchain_calls=toolchain_calls, # use tool
                    # final_answer=final_answer,
                    # reference_tools=reference_tools,
                    # reference_answer=reference_answer,
                    # dag_results=dag_results,
                    # dag_status=result. get("dag_status", "success"),
                    # dag_error=result. get("dag_error"),
                    # )
                    reflection = ""
                    result["self_reflection"] = reflection
                else:
                    print(f"[WARN] DAG task")
                    result["dag_status"] = "no_tasks"
                    # DAG task, try generate final answer(useempty dag_results)
                    try:
                        final_answer = agent.generate_final_answer(
                            task=query,
                            dag_results={},  # emptydictionary
                            file_contents=[],  # empty list
                        )
                        result["final_answer"] = final_answer
                        
                    except Exception as final_answer_exc:
                        print(f"[ERROR] generate final answer failed: {final_answer_exc}")
                        result["final_answer"] = ""
                        result["final_answer_error"] = str(final_answer_exc)
                    
            except Exception as dag_exc:
                print(f"[ERROR] dependency analysis DAG failed: {dag_exc}")
                print(f"[ERROR] errortype: {type(dag_exc).__name__}")
                print(f"[ERROR] error:")
                traceback.print_exc()
                
                # : dependency analysis failed, use dependency
                print(f"\n[WARN] dependency analysis failed, mode...")
                try:
                    dag_task_list: List[Dict[str, Any]] = []
                    for idx, call in enumerate(toolchain_calls):
                        if not isinstance(call, dict):
                            continue
                        tool_name = call.get("tool")
                        call_args = call.get("arguments", {})
                        if not tool_name:
                            continue
                        
                        task_id = idx + 1
                        dependencies = [task_id - 1] if task_id > 1 else []
                        
                        dag_task = {
                            "id": task_id,
                            "tool_name": str(tool_name),
                            "arguments": call_args if isinstance(call_args, dict) else {},
                            "dependencies": dependencies,
                        }
                        dag_task_list.append(dag_task)
                    
                    if dag_task_list:
                        print(f"[INFO] use dependencymode, {len(dag_task_list)} task")
                        async def run_tool(task: dict):
                            tool = task.get("tool_name")
                            args = task.get("arguments", {}) or {}
                            print(f"[INFO] tool: {tool} (Task ID: {task['id']})")
                            output = await handle_call_tool(tool, args)
                            return {
                                "status": "success",
                                "tool": tool,
                                "args": args,
                                "task_id": task["id"],
                                "output": output,
                            }
                        
                        executor = DAGExecutor(
                            task_list=dag_task_list,
                            run_tool=run_tool,
                            agent_instance=agent
                        )
                        dag_results = asyncio.run(executor.run())
                        result["dag_results"] = dag_results
                        result["dag_status"] = "success_fallback"
                        result["dag_fallback_reason"] = str(dag_exc)

                        file_contents = agent.collect_existing_file_contents(
                            dag_results,
                            task=query,  # querytask related
                            max_chars=10000,
                            use_llamaindex=True,
                        )
                        final_answer = agent.generate_final_answer(
                            task=query,
                            dag_results=dag_results,
                            file_contents=file_contents,
                        )
                        result["final_answer"] = final_answer
                        
                        # reflection = agent.self_reflect_and_save(
                        # task=query,
                        # toolchain_calls=toolchain_calls, # use tool
                        # final_answer=final_answer,
                        # reference_tools=reference_tools,
                        # reference_answer=reference_answer,
                        # dag_results=dag_results,
                        # dag_status=result. get("dag_status", "success_fallback"),
                        # dag_error=result. get("dag_fallback_reason"),
                        # )
                        reflection = ""
                        result["self_reflection"] = reflection
                    else:
                        result["dag_status"] = "error"
                        result["dag_error"] = str(dag_exc)
                        result["dag_error_type"] = type(dag_exc).__name__
                        # failed, try generate final answer(useempty dag_results)
                        try:
                            final_answer = agent.generate_final_answer(
                                task=query,
                                dag_results={},  # emptydictionary
                                file_contents=[],  # empty list
                            )
                            result["final_answer"] = final_answer
                            
                        except Exception as final_answer_exc:
                            print(f"[ERROR] generate final answer failed: {final_answer_exc}")
                            result["final_answer"] = ""
                            result["final_answer_error"] = str(final_answer_exc)
                except Exception as fallback_exc:
                    print(f"[ERROR] failed: {fallback_exc}")
                    result["dag_status"] = "error"
                    result["dag_error"] = f"dependency analysis failed: {dag_exc}; failed: {fallback_exc}"
                    result["dag_error_type"] = type(dag_exc).__name__
                    # exception, try generate final answer(useempty dag_results)
                    try:
                        final_answer = agent.generate_final_answer(
                            task=query,
                            dag_results={},  # emptydictionary
                            file_contents=[],  # empty list
                        )
                        result["final_answer"] = final_answer
                    except Exception as final_answer_exc:
                        print(f"[ERROR] generate final answer failed: {final_answer_exc}")
                        result["final_answer"] = ""
                        result["final_answer_error"] = str(final_answer_exc)
        else:
            # , empty tool
            # , (tools_list is empty toolchain_calls is empty)
            print(f"[WARN] tool listis empty, skip DAG")
            result["dag_status"] = "no_tools"
            # tool listis empty, try generate final answer
            try:
                final_answer = agent.generate_final_answer(
                    task=query,
                    dag_results={},  # emptydictionary
                    file_contents=[],  # empty list
                )
                result["final_answer"] = final_answer
            except Exception as final_answer_exc:
                print(f"[ERROR] generate final answer failed: {final_answer_exc}")
                result["final_answer"] = ""
                result["final_answer_error"] = str(final_answer_exc)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        # exception, final_answer field exists(is empty)
        if "final_answer" not in result:
            result["final_answer"] = ""
        print(f"[ERROR] (Base Agent test) failed: {e}")
        traceback.print_exc()

    return result


def main() -> None:
    'Base Agent test.'
    # save dag_memory.json
    BaseAgent.SAVE_TO_DAG_MEMORY = False
    print(f"[INFO] SAVE_TO_DAG_MEMORY False, save dag_memory.json")
    
    parser = argparse.ArgumentParser(description='Resumable benchmark runner for Base Agent test (with tools/report/notebook)')
    parser.add_argument("--benchmark_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/benchmark.json')))
    parser.add_argument("--benchmark_true_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset/benchmark.json')))
    parser.add_argument("--tools_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs/tools.json')))
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

    # read benchmark_true.json get tool type answer
    benchmark_true_path = os.path.abspath(args.benchmark_true_path)
    benchmark_true_map = {}
    if os.path.exists(benchmark_true_path):
        with open(benchmark_true_path, "r", encoding="utf-8") as f:
            try:
                benchmark_true_list = json.load(f)
                if isinstance(benchmark_true_list, list):
                    for item in benchmark_true_list:
                        q = item.get("query", "")
                        if q:
                            benchmark_true_map[q] = item
            except Exception as e:
                print(f"[WARN] benchmark_true.json failed: {e}")

    total_queries = len(benchmark_data)
    print(f"[INFO] {total_queries} query")

    # read tools.json(Base Agent test tools input)
    tools_path = os.path.abspath(args.tools_path)
    if not os.path.exists(tools_path):
        print(f"[ERROR] tools.json file does not exist: {tools_path}")
        sys.exit(1)

    print(f"[INFO] read tools.json Base Agent test tool: {tools_path}")
    with open(tools_path, "r", encoding="utf-8") as f:
        tools_config = json.load(f)

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
    output_dir = os.path.join(os.path.dirname(__file__), "Result", "Benchmark")
    os.makedirs(output_dir, exist_ok=True)
    if args.results_path:
        results_path = os.path.abspath(args.results_path)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"normal_qwen_benchmark_results_{ts}.json")
    progress = BenchmarkProgress(results_path=results_path, resume_mode=args.resume_mode, run_name="test_normal_qwen")
    existing = progress.stats()
    if existing.total_cases_seen:
        print(
            f"[INFO] Resume enabled: already have {existing.total_cases_seen} cases"
            f"(success={existing.success_cases}, error={existing.error_cases}) in {progress.results_path}"
        )
    progress.start_run({"script": "test_normal_qwen", "benchmark_path": benchmark_path, "metric": "tool_recall"})

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
            continue

        # readcurrent query tool(candidate list)
        raw_expected_tools = benchmark_item.get("expected_tool_types", [])
        expected_groups: List[List[str]] = []
        if isinstance(raw_expected_tools, list):
            for entry in raw_expected_tools:
                if isinstance(entry, list):
                    expected_groups.append(entry)
                elif entry is not None:
                    expected_groups.append([entry])

        expected_count = len(expected_groups)
        predicted_tools: List[str] = []
        status = "error"
        error_msg: Any = None

        ref_entry = benchmark_true_map.get(query, {})
        reference_tools = ref_entry.get("expected_tool_types", []) or []
        reference_answer = ref_entry.get("execution_result", "") or ""

        try:
            result = test_agent_query(
                agent,
                query,
                idx,
                total_queries,
                tools_config,
                reference_tools,
                reference_answer,
            )

            status = result.get("status", "error")
            error_msg = result.get("error")

            # result field Predicted tools(status error)
            final_result_item = result.get(
                "result",
                {
                    "query": query,
                    "tools": [],
                    "arguments": [],
                },
            )
            predicted_tools = final_result_item.get("tools", []) or []
            
            # extract final_answer(agent.generate_final_answer() return)
            # test_agent_query agent.generate_final_answer() call
            final_answer = result.get("final_answer", "")
            
            # : final_answer extract
            if final_answer:
                print(f"[INFO] extract final_answer (: {len(final_answer)})")
            else:
                print(f"[WARN] final_answer is empty or does not exist")

            # calculate: Agent,
            hit_count = 0
            if expected_count > 0:
                predicted_set = set(predicted_tools)
                for group in expected_groups:
                    if any(t in predicted_set for t in group):
                        hit_count += 1
                recall = hit_count / expected_count if expected_count > 0 else 0.0
            else:
                # tool: success, 1.0, 0.0
                hit_count = 0
                recall = 1.0 if status == "success" else 0.0

            # success: candidate tool (Recall=1.0), extra tool
            if expected_count == 0:
                success_flag = status == "success"
            else:
                success_flag = hit_count == expected_count

            progress.append(
                {
                    "query": query,
                    "query_index": idx,
                    "status": status,
                    "expected_tools": expected_groups,
                    "predicted_tools": predicted_tools,
                    "tools": predicted_tools,
                    "arguments": final_result_item.get("arguments", []),
                    "final_answer": final_answer if final_answer else "",
                    "hit_slots": hit_count,
                    "expected_slots": expected_count,
                    "recall": recall,
                    "success": success_flag,
                    "error": error_msg,
                }
            )

        except Exception as e:  # , exception
            print(f"[ERROR] (Base Agent test) {idx} query exception: {e}")
            traceback.print_exc()
            # try exception extract final_answer(result generate)
            final_answer = ""
            try:
                if 'result' in locals():
                    final_answer = result.get("final_answer", "")
            except:
                pass
            progress.append(
                {
                    "query": query,
                    "query_index": idx,
                    "status": "exception",
                    "expected_tools": expected_groups,
                    "predicted_tools": [],
                    "tools": [],
                    "arguments": [],
                    "final_answer": final_answer if final_answer else "",
                    "hit_slots": 0,
                    "expected_slots": expected_count,
                    "recall": 0.0,
                    "success": False,
                    "error": str(e),
                }
            )

    all_latest = progress.iter_latest_records()
    success_count = sum(1 for r in all_latest if bool(r.get("success")))
    error_count = len(all_latest) - success_count
    total_expected_slots = 0
    total_hit_slots = 0
    for r in all_latest:
        try:
            total_expected_slots += int(r.get("expected_slots", 0))
        except Exception:
            pass
        try:
            total_hit_slots += int(r.get("hit_slots", 0))
        except Exception:
            pass
    overall_recall = total_hit_slots / total_expected_slots if total_expected_slots > 0 else 0.0

    # record time (result file)
    test_end_dt = datetime.now()
    test_end_time = test_end_dt.isoformat()
    duration_seconds = (test_end_dt - test_start_dt).total_seconds()
    progress.finish_run(
        {
            "last_test_start_time": test_start_time,
            "last_test_end_time": test_end_time,
            "last_duration_seconds": duration_seconds,
            "overall_recall": overall_recall,
            "total_expected_slots": total_expected_slots,
            "total_hit_slots": total_hit_slots,
        }
    )

    print('' + "=" * 80)
    print('[SUCCESS] Base Agent test Benchmark complete')
    print(f"[INFO] query(): {len(all_latest)}")
    print(f"[INFO] success(Recall=1.0): {success_count}")
    print(f"[INFO] failed/: {error_count}")
    print(f"[INFO] tool: {total_expected_slots}")
    print(f"[INFO]: {total_hit_slots}")
    print(f"[INFO] Overall Recall: {overall_recall:.4f}")
    print(f"[INFO] result save: {progress.results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
