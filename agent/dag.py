import asyncio
import json
import os
import time
import traceback
from typing import List, Dict, Any, Callable, Awaitable, Optional

try:
    from agent.base_agent import BaseAgent  # type: ignore
except ImportError:
    BaseAgent = None  # type: ignore


class DAGExecutor:
    

    def __init__(
        self,
        task_list: List[Dict[str, Any]],
        run_tool: Optional[Callable[[Dict[str, Any]], Awaitable[Any]]] = None,
        agent_instance: Optional[Any] = None,
        dag_memory_path: Optional[str] = None,
    ):
        self.tasks = {t["id"]: t for t in task_list}
        self.graph = {t["id"]: [] for t in task_list}  
        self.in_degree = {t["id"]: 0 for t in task_list}  
        self.run_tool = run_tool or self._run_tool_mock
        self.agent_instance = agent_instance  
        
        if dag_memory_path is None:
            dag_memory_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../rag/data/dag_memory.json')
            )
        self.dag_memory_path = dag_memory_path
        self._build_graph(task_list)

    def _build_graph(self, task_list):
        """text structure"""
        for task in task_list:
            u = task["id"]
            for v in task.get("dependencies", []):
                if v not in self.tasks:
                    raise ValueError(f"Task {u} depends on unknown task {v}")
                self.graph[v].append(u)
                self.in_degree[u] += 1

    async def _run_tool_mock(self, task: Dict[str, Any]):
        'default mock; actually use run_tool overwrite'
        tool_name = task["tool_name"]
        args = task.get("arguments", {})
        print(f"[START] ID:{task['id']} | Tool: {tool_name}")
        await asyncio.sleep(0.5)
        print(
            f"OK [FINISH] ID:{task['id']} | Args: {json.dumps(args, ensure_ascii=False)[:80]}..."
        )
        return {"status": "success", "tool": tool_name, "args": args}

    async def run(self):
        ': dependency parallel, dependency'
        start_time = time.time()

        # (0)
        queue = [tid for tid, deg in self.in_degree.items() if deg == 0]

        running_futures = {}  # {future: task_id}
        results = {}
        failed_tasks = {}  # {task_id: error_message} record failed task

        for tid in queue:
            task_data = self.tasks[tid]
            future = asyncio.create_task(self.run_tool(task_data))
            running_futures[future] = tid

        while running_futures:
            done, _ = await asyncio.wait(
                running_futures.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for future in done:
                tid = running_futures.pop(future)
                task_success = False
                try:
                    res = future.result()
                    results[tid] = res
                    task_success = True
                except Exception as e:
                    error_msg = str(e)
                    print(f"ERR Task {tid} failed: {error_msg}")
                    failed_tasks[tid] = error_msg
                    # failedcontinue task
                
                # task success failed, dependency
                # dependency parallel, task
                for successor_id in self.graph[tid]:
                    self.in_degree[successor_id] -= 1
                    if self.in_degree[successor_id] == 0:
                        next_task = self.tasks[successor_id]
                        # dependency task failed, task record warning
                        if not task_success:
                            task_info = self.tasks.get(tid, {})
                            dep_tool_name = task_info.get('tool_name', f'Task {tid}')
                            print(f"WARN Task {successor_id} dependency task {tid} ({dep_tool_name}) failed, continue")
                        new_future = asyncio.create_task(self.run_tool(next_task))
                        running_futures[new_future] = successor_id

        total_time = time.time() - start_time
        print(f"\n All tasks completed in {total_time:.2f}s")
        
        # save result dag_memory.json
        try:
            self._save_to_dag_memory(results, failed_tasks)
        except Exception as e:
            print(f"[WARN] save dag_memory.json failed: {e}")
            traceback.print_exc()
        
        return results
    
    def _save_to_dag_memory(self, results: Dict[int, Any], failed_tasks: Dict[int, str]):
        'save result to dag_memory.json. Args: results: successful task result {task_id: result} failed_tasks: failed task {task_id: error_message}'
        # checkglobal, False skip save
        if BaseAgent is None:
            print('[WARN] Base Agent, skip save dag_memory.json')
            return
        if not BaseAgent.SAVE_TO_DAG_MEMORY:
            print('[INFO] SAVE_TO_DAG_MEMORY=False, skip save dag_memory.json')
            return
        
        if self.agent_instance is None or self.agent_instance.full_task is None:
            print('[WARN] agent_instance full_task is empty, skip save dag_memory.json')
            return
        
        # success: task success success
        all_success = len(failed_tasks) == 0
        
        # error
        error_messages = []
        if not all_success:
            for tid, error_msg in failed_tasks.items():
                task_info = self.tasks.get(tid, {})
                tool_name = task_info.get('tool_name', 'Unknown')
                error_messages.append(f"Task {tid} ({tool_name}): {error_msg}")
            error_message = ';'.join(error_messages)
        else:
            error_message = ""
        
        # update full_task result
        self.agent_instance._update_full_task_result(all_success, error_message)
        
        # read dag_memory.json
        dag_memory_data = []
        if os.path.exists(self.dag_memory_path):
            try:
                with open(self.dag_memory_path, 'r', encoding='utf-8') as f:
                    dag_memory_data = json.load(f)
                    if not isinstance(dag_memory_data, list):
                        dag_memory_data = []
            except Exception as e:
                print(f"[WARN] read dag_memory.json failed: {e}, file")
                dag_memory_data = []
        
        # task record
        task_record = self.agent_instance.full_task.copy()
        dag_memory_data.append(task_record)
        
        # save file
        try:
            # directory exists
            os.makedirs(os.path.dirname(self.dag_memory_path), exist_ok=True)
            
            with open(self.dag_memory_path, 'w', encoding='utf-8') as f:
                json.dump(dag_memory_data, f, ensure_ascii=False, indent=2)
            
            print(f"[SUCCESS] save task record {self.dag_memory_path}")
            print(f"[INFO] task status: {'success' if all_success else 'failed'}")
            if error_message:
                print(f"[INFO] failed: {error_message[:200]}...")
        except Exception as e:
            print(f"[ERROR] save dag_memory.json failed: {e}")
            traceback.print_exc()
            raise