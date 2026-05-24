import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agent.judge_agent import JudgeAgent


DEFAULT_BENCHMARK_PATH = ""
DEFAULT_INPUT_PATH = ""
DEFAULT_OUTPUT_DIR = ""


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_benchmark_map(benchmark_items: List[Dict[str, Any]]) -> Dict[str, str]:
    """query -> execution_result"""
    m: Dict[str, str] = {}
    for item in benchmark_items:
        q = (item.get("query") or "").strip()
        ans = item.get("execution_result")
        if not q:
            continue
        
        if ans is None:
            ans_s = ""
        elif isinstance(ans, str):
            ans_s = ans
        else:
            ans_s = json.dumps(ans, ensure_ascii=False)
        m[q] = ans_s
    return m


def _extract_results_list(obj: Any) -> List[Dict[str, Any]]:
    """
    兼容：
    - {"results":[...]}
    - 直接是 list
    """
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        results = obj.get("results")
        if isinstance(results, list):
            return [x for x in results if isinstance(x, dict)]
    return []


def _pick_agent_answer(item: Dict[str, Any]) -> str:
    
    for k in ("final_answer", "answer", "output", "result"):
        if k in item and item.get(k) is not None:
            v = item.get(k)
            if isinstance(v, str):
                return v
            return json.dumps(v, ensure_ascii=False)
    return ""


def judge_one(
    judge_agent: JudgeAgent,
    query: str,
    agent_answer: str,
    benchmark_answer: str,
    max_retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    return judge_agent.judge(
        query=query,
        agent_execution=agent_answer,
        true_answer=benchmark_answer,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )


def evaluate_file(
    judge_agent: JudgeAgent,
    source_name: str,
    result_path: str,
    benchmark_map: Dict[str, str],
    output_dir: str,
    max_items: Optional[int],
    max_retries: int,
    retry_sleep: float,
) -> Tuple[Dict[str, Any], str]:
    raw = _load_json(result_path)
    items = _extract_results_list(raw)

    evaluated: List[Dict[str, Any]] = []
    pass_cnt = 0
    fail_cnt = 0
    missing_truth = 0

    total = len(items)
    if max_items is not None and max_items > 0:
        items = items[:max_items]

    for idx, item in enumerate(items, 1):
        query = (item.get("query") or "").strip()
        if not query:
            continue

        benchmark_answer = benchmark_map.get(query, "")
        if not benchmark_answer:
            missing_truth += 1

        agent_answer = _pick_agent_answer(item)
        judge_output = judge_one(
            judge_agent=judge_agent,
            query=query,
            agent_answer=agent_answer,
            benchmark_answer=benchmark_answer,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )

        result_flag = (judge_output.get("result") or "").upper().strip()
        if result_flag == "PASS":
            pass_cnt += 1
        else:
            fail_cnt += 1

        evaluated.append(
            {
                "query_index": idx,
                "query": query,
                "agent_answer": agent_answer,
                "benchmark_answer": benchmark_answer,
                "judge_output": judge_output,
            }
        )
       
        if idx == 1 or idx % 20 == 0 or idx == len(items):
            print(
                f"[{source_name}] {idx}/{len(items)} | PASS={pass_cnt} FAIL={fail_cnt} missing_truth={missing_truth}"
            )

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "source_name": source_name,
        "source_path": result_path,
        "total_items_in_file": total,
        "evaluated_count": len(evaluated),
        "pass": pass_cnt,
        "fail": fail_cnt,
        "missing_truth": missing_truth,
        "details": evaluated,
    }

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"judge_{source_name}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="",
    )
    parser.add_argument(
        "--source_name",
        default="",
        help="",
    )
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_items", type=int, default=0, help="")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=1.0)
    args = parser.parse_args()

    benchmark_data = _load_json(args.benchmark)
    if not isinstance(benchmark_data, list):
        raise ValueError("")
    benchmark_map = _build_benchmark_map(benchmark_data)
    print(f"[INFO] benchmark loading successful")

    judge_agent = JudgeAgent(config={})

    max_items = args.max_items if args.max_items and args.max_items > 0 else None

   
    input_path = args.input
    if not input_path:
        raise ValueError("")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"--input not existed: {input_path}")

    source_name = (args.source_name or "").strip()
    if not source_name:
        base = os.path.basename(input_path)
        source_name = os.path.splitext(base)[0]

    summary, out_path = evaluate_file(
        judge_agent=judge_agent,
        source_name=source_name,
        result_path=input_path,
        benchmark_map=benchmark_map,
        output_dir=args.out_dir,
        max_items=max_items,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )

    print(
        f"[SUCCESS]: source={source_name} PASS={summary.get('pass', 0)} FAIL={summary.get('fail', 0)} "
        f"evaluated={summary.get('evaluated_count', 0)} -> {out_path}"
    )


if __name__ == "__main__":
    main()


