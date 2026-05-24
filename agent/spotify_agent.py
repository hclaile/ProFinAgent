import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

try:
    import requests  # type: ignore
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None  # type: ignore
    _REQUESTS_AVAILABLE = False

import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import prompt.spotify_prompt as PROMPT  # type: ignore

try:
    import agent.agent_config as agent_cfg  # type: ignore
    _AGENT_CONFIG_AVAILABLE = True
except ImportError:
    agent_cfg = None  # type: ignore
    _AGENT_CONFIG_AVAILABLE = False

CustomEmbeddingFunction = None  # type: ignore
_RAG_AVAILABLE = False
if _AGENT_CONFIG_AVAILABLE and getattr(agent_cfg, "EMBEDDING_MODEL_PATH", ""):
    try:
        from agent.rag import CustomEmbeddingFunction  # type: ignore
        _RAG_AVAILABLE = True
    except ImportError:
        CustomEmbeddingFunction = None  # type: ignore
        _RAG_AVAILABLE = False


class SpotifyAgent:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        if not _OPENAI_AVAILABLE and not _REQUESTS_AVAILABLE:
            raise ImportError('not installed openai/requests, initialize Spotify Agent.')
        if not _AGENT_CONFIG_AVAILABLE:
            raise ImportError('agent_config, check/agent/agent_config.py.')

        self.api_key = (
            self.config.get("api_key")
            or os.getenv("SPOTIFY_AGENT_API_KEY")
            or getattr(agent_cfg, "spotify_agent_apikey", None)
            or getattr(agent_cfg, "test_agent_apikey", None)
            or os.getenv("OPENAI_API_KEY", "")
        )
        self.base_url = str(
            self.config.get("base_url")
            or os.getenv("SPOTIFY_AGENT_BASE_URL")
            or getattr(agent_cfg, "spotify_agent_base_url", None)
            or getattr(agent_cfg, "test_agent_base_url", "")
        ).rstrip("/")
        self.model = (
            self.config.get("model")
            or os.getenv("SPOTIFY_AGENT_MODEL")
            or getattr(agent_cfg, "spotify_agent_model", None)
            or getattr(agent_cfg, "test_agent_model", "")
        )

        if not self.api_key:
            raise ValueError('Spotify Agent initialize failed: missing API Key.')
        if not self.base_url:
            raise ValueError('Spotify Agent initialize failed: missing base_url.')
        if not self.model:
            raise ValueError('Spotify Agent initialize failed: missing model.')

        self.temperature = float(self.config.get("temperature", getattr(agent_cfg, "TEMPERATURE", 0.0)))
        self.max_tokens = int(self.config.get("max_tokens", getattr(agent_cfg, "MAX_TOKENS", 6144)))

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if _OPENAI_AVAILABLE else None

        self.tool_threshold = float(self.config.get("tool_threshold", 0.1))
        self.notebook_threshold = float(self.config.get("notebook_threshold", 0.9))

        # notebook data (similar)+ file(reflection)
        self.seed_notebook_path = self.config.get(
            "seed_notebook_path",
            os.path.join(ROOT_DIR, "notebook", 'spotify_notebook.json'),
        )
        self.notebook_path = self.config.get(
            "notebook_path",
            os.path.join(ROOT_DIR, "notebook", 'spotify_notebook.json'),
        )

        self._embedding_function = None
        # record LLM tool, DAG dependency
        self._last_tool_meta: Dict[str, Dict[str, Any]] = {}
        if _RAG_AVAILABLE and hasattr(agent_cfg, "EMBEDDING_MODEL_PATH"):
            try:
                self._embedding_function = CustomEmbeddingFunction(
                    model_path=agent_cfg.EMBEDDING_MODEL_PATH,
                    matryoshka_dim=getattr(agent_cfg, "MATRYOSHKA_DIM", None),
                )
            except Exception:
                self._embedding_function = None

    # ---------- basic tool ----------
    def _chat_json_object(self, prompt: str) -> str:
        if self.client is not None:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Please output JSON only."},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""

        if requests is None:
            raise ImportError('requests not installed, call LLM.')

        url = self.base_url
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Please output JSON only."},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        r = requests.post(url, headers=headers, json=payload, timeout=600)
        if r.status_code >= 400:
            payload.pop("response_format", None)
            r = requests.post(url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""

    def _strip_tags_and_fences(self, text: str) -> str:
        s = (text or "").strip()
        s = s.replace("<PLAN>", "").replace("<END_OF_PLAN>", "").strip()
        if s.startswith("```"):
            first_newline = s.find('')
            if first_newline != -1:
                s = s[first_newline + 1 :]
            if s.endswith("```"):
                s = s[:-3]
        return s.strip()

    def _safe_parse_json(self, text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        s = self._strip_tags_and_fences(text)
        try:
            return json.loads(s)
        except Exception:
            pass
        fb, lb = s.find("{"), s.rfind("}")
        if fb != -1 and lb > fb:
            try:
                return json.loads(s[fb : lb + 1])
            except Exception:
                pass
        fbr, lbr = s.find("["), s.rfind("]")
        if fbr != -1 and lbr > fbr:
            try:
                return json.loads(s[fbr : lbr + 1])
            except Exception:
                pass
        return None

    def _compute_similarity(self, q_vec: List[float], d_vec: List[float]) -> float:
        q = np.array(q_vec, dtype=np.float32)
        d = np.array(d_vec, dtype=np.float32)
        denom = (np.linalg.norm(q) * np.linalg.norm(d)) + 1e-12
        return float(np.dot(q, d) / denom)

    def _normalize_tool_key(self, tool_name: str) -> str:
        s = (tool_name or "").strip()
        if not s:
            return ""
        parts = s.split(None, 1)
        if len(parts) == 1:
            return parts[0].upper()
        return f"{parts[0].upper()} {parts[1].strip()}"

    def _load_json_if_path(self, obj: Union[str, Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any], str]:
        if isinstance(obj, str):
            path = os.path.expanduser(obj)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return obj

    # ---------- tool ----------
    def _normalize_tools(self, tools_input: Union[str, Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
        tools_obj = self._load_json_if_path(tools_input)

        # OpenAPI format(spotify_oas.json)
        if isinstance(tools_obj, dict) and isinstance(tools_obj.get("paths"), dict):
            normalized: List[Dict[str, Any]] = []
            for path, methods in tools_obj["paths"].items():
                if not isinstance(methods, dict):
                    continue
                for method, spec in methods.items():
                    if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                        continue
                    if not isinstance(spec, dict):
                        continue
                    name = f"{method.upper()} {path}"
                    summary = spec.get("summary", "")
                    desc = spec.get("description", "")
                    normalized.append(
                        {
                            "name": name,
                            "description": f"{summary}. {desc}".strip(),
                        }
                    )
            return normalized

        # list tool
        if isinstance(tools_obj, list):
            normalized = []
            for item in tools_obj:
                if isinstance(item, str):
                    normalized.append(
                        {
                            "tool": item,
                            "tool_description": "",
                            "name": item,
                            "description": "",
                        }
                    )
                elif isinstance(item, dict):
                    # prefer using benchmark tool field: tool/tool_description
                    tool_name = item.get("tool") or item.get("name") or item.get("operationId") or ""
                    if not tool_name and item.get("method") and item.get("path"):
                        tool_name = f"{str(item['method']).upper()} {item['path']}"
                    if tool_name:
                        tool_desc = item.get("tool_description") or item.get("description") or item.get("summary") or ""
                        # keep original tool key/value, zero-shot field
                        enriched = dict(item)
                        enriched["tool"] = str(tool_name)
                        enriched["tool_description"] = str(tool_desc)
                        # compatible field
                        enriched.setdefault("name", str(tool_name))
                        enriched.setdefault("description", str(tool_desc))
                        normalized.append(enriched)
            return normalized

        return []

    def _fallback_toolchain_from_tools(
        self,
        question: str,
        tools: List[Dict[str, Any]],
        max_calls: int = 4,
    ) -> List[Dict[str, Any]]:
        q_tokens = set((question or "").lower().split())
        scored: List[tuple[float, str]] = []
        for t in tools:
            tool_name = str(t.get("tool") or t.get("name") or "").strip()
            desc = str(t.get("tool_description") or t.get("description") or "").strip()
            if not tool_name:
                continue
            text = f"{tool_name} {desc}".lower()
            overlap = sum(1 for tok in q_tokens if tok and tok in text)
            # search userrelated tool, empty
            bias = 0.0
            if "search" in tool_name.lower():
                bias += 0.3
            if "/me" in tool_name.lower():
                bias += 0.1
            scored.append((float(overlap) + bias, tool_name))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked_names: List[str] = []
        for _, name in scored:
            if name not in picked_names:
                picked_names.append(name)
            if len(picked_names) >= max_calls:
                break
        return [{"tool": n} for n in picked_names]

    def _tool_embedding_filter(self, question: str, tools_input: Union[str, Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
        tools = self._normalize_tools(tools_input)
        if not tools:
            return []
        if self._embedding_function is None:
            return tools

        q_vec = self._embedding_function.embed_query(question)
        scored: List[Dict[str, Any]] = []
        for tool in tools:
            tool_desc = str(tool.get("tool_description") or tool.get("description") or "")
            # use tool_description embedding, field
            d_vec = self._embedding_function.embed_documents([tool_desc])[0]
            sim = self._compute_similarity(q_vec, d_vec)
            item = dict(tool)
            item["_similarity"] = sim
            scored.append(item)

        picked = [t for t in scored if float(t.get("_similarity", 0.0)) >= self.tool_threshold]
        if picked:
            picked.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)
            return picked

        scored.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)
        # threshold, candidate, LLM tool output empty
        fallback_k = int(self.config.get("tool_fallback_top_k", 24))
        fallback_k = max(8, fallback_k)
        return scored[: min(fallback_k, len(scored))]

    def _to_llm_tool_card(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        'LLM tool_card: keep tool directrelated field.'
        llm_cards: List[Dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            llm_cards.append(
                {
                    "tool_description": t.get("tool_description", ""),
                    "tool": t.get("tool", t.get("name", "")),
                    "input_parameters": t.get("input_parameters", {}),
                    "output_parameters": t.get("output_parameters", {}),
                }
            )
        return llm_cards

    def _build_tool_meta_map(self, llm_cards: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        meta: Dict[str, Dict[str, Any]] = {}
        for card in llm_cards:
            tool_name = str(card.get("tool", "")).strip()
            if not tool_name:
                continue
            key = self._normalize_tool_key(tool_name)
            meta[key] = {
                "tool_description": str(card.get("tool_description", "")),
                "input_parameters": card.get("input_parameters", {}),
                "output_parameters": card.get("output_parameters", {}),
            }
        return meta

    # ---------- notebook ----------
    def _load_notebook_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for path in [self.seed_notebook_path, self.notebook_path]:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    entries.extend([x for x in data if isinstance(x, dict)])
            except Exception:
                continue
        return entries

    def _find_similar_notebooks(self, question: str, top_k: int = 1) -> List[Dict[str, Any]]:
        entries = self._load_notebook_entries()
        if not entries:
            return []
        if self._embedding_function is None:
            return entries[:top_k]

        q_vec = self._embedding_function.embed_query(question)
        scored: List[Dict[str, Any]] = []
        for entry in entries:
            task_text = entry.get("task") or entry.get("query") or ""
            if not task_text:
                continue
            d_vec = self._embedding_function.embed_documents([str(task_text)])[0]
            sim = self._compute_similarity(q_vec, d_vec)
            if sim >= self.notebook_threshold:
                item = dict(entry)
                item["_similarity"] = sim
                scored.append(item)

        scored.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)
        return scored[:top_k]

    # ---------- 3 ----------
    def select_tools(
        self,
        question: str,
        tools: Union[str, Dict[str, Any], List[Any]],
        max_retries: int = 5,
        retry_sleep: float = 1.0,
    ) -> Dict[str, Any]:
        '1: embedding tools notebook(threshold 0.7), call LLM plan tool.'
        if not isinstance(question, str) or not question.strip():
            raise ValueError('question empty.')

        filtered_tools = self._tool_embedding_filter(question, tools)
        llm_tool_card = self._to_llm_tool_card(filtered_tools)
        self._last_tool_meta = self._build_tool_meta_map(llm_tool_card)
        similar_notebooks = self._find_similar_notebooks(question, top_k=1)

        prompt = PROMPT.select_tools.format(
            question=question,
            tools=json.dumps(llm_tool_card, ensure_ascii=False, indent=2),
            notebook=json.dumps(similar_notebooks, ensure_ascii=False, indent=2),
        )

        last_error: Optional[str] = None
        for attempt in range(max(1, max_retries)):
            try:
                content = self._chat_json_object(prompt)
                if not content:
                    raise ValueError("LLM returnis empty")
                data = self._safe_parse_json(content)
                if not isinstance(data, dict):
                    raise ValueError('return result is not JSON')
                calls = data.get("toolchain_calls", [])
                if not isinstance(calls, list):
                    raise ValueError('toolchain_calls is not a list')
                normalized_calls = []
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    tool_name = call.get("tool") or call.get("tool_name")
                    if tool_name:
                        tool_name_str = str(tool_name)
                        key = self._normalize_tool_key(tool_name_str)
                        meta = self._last_tool_meta.get(key, {})
                        normalized_calls.append(
                            {
                                "tool": tool_name_str,
                                "tool_description": meta.get("tool_description", ""),
                                "input_parameters": meta.get("input_parameters", {}),
                                "output_parameters": meta.get("output_parameters", {}),
                            }
                        )
                if not normalized_calls:
                    # LLM return empty, benchmark result []
                    normalized_calls = self._fallback_toolchain_from_tools(
                        question=question,
                        tools=filtered_tools,
                        max_calls=int(self.config.get("fallback_max_calls", 4)),
                    )
                    # tool tool
                    enriched_calls = []
                    for c in normalized_calls:
                        tool_name_str = str(c.get("tool", ""))
                        key = self._normalize_tool_key(tool_name_str)
                        meta = self._last_tool_meta.get(key, {})
                        enriched_calls.append(
                            {
                                "tool": tool_name_str,
                                "tool_description": meta.get("tool_description", ""),
                                "input_parameters": meta.get("input_parameters", {}),
                                "output_parameters": meta.get("output_parameters", {}),
                            }
                        )
                    normalized_calls = enriched_calls
                return {"toolchain_calls": normalized_calls}
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1 and retry_sleep > 0:
                    time.sleep(retry_sleep)

        return {
            "toolchain_calls": [
                {
                    "tool": str(c.get("tool", "")),
                    "tool_description": self._last_tool_meta.get(
                        self._normalize_tool_key(str(c.get("tool", ""))), {}
                    ).get("tool_description", ""),
                    "input_parameters": self._last_tool_meta.get(
                        self._normalize_tool_key(str(c.get("tool", ""))), {}
                    ).get("input_parameters", {}),
                    "output_parameters": self._last_tool_meta.get(
                        self._normalize_tool_key(str(c.get("tool", ""))), {}
                    ).get("output_parameters", {}),
                }
                for c in self._fallback_toolchain_from_tools(
                    question=question,
                    tools=filtered_tools,
                    max_calls=int(self.config.get("fallback_max_calls", 4)),
                )
            ],
            "error": last_error or "select_tools failed",
        }

    def zero_shot_spotify(
        self,
        question: str,
        tools: Union[str, Dict[str, Any], List[Any]],
        max_retries: int = 5,
        retry_sleep: float = 1.0,
    ) -> Dict[str, Any]:
        'Zero-shot Spotify tool plan: - use prompt.zero_spotify(dependency notebook) - output format select_tools:{"toolchain_calls": [...]}'
        if not isinstance(question, str) or not question.strip():
            raise ValueError('question empty.')

        # zero-shot mode: LLM original tools(original OAS JSON),
        raw_tools_obj = self._load_json_if_path(tools)
        llm_tool_card = raw_tools_obj

        # tool meta, fallback result field
        normalized_tools = self._normalize_tools(raw_tools_obj)
        self._last_tool_meta = self._build_tool_meta_map(self._to_llm_tool_card(normalized_tools))

        prompt = PROMPT.zero_spotify.format(
            question=question,
            tools=json.dumps(llm_tool_card, ensure_ascii=False, indent=2),
        )

        last_error: Optional[str] = None
        for attempt in range(max(1, max_retries)):
            try:
                content = self._chat_json_object(prompt)
                if not content:
                    raise ValueError("LLM returnis empty")

                data = self._safe_parse_json(content)
                if not isinstance(data, dict):
                    raise ValueError('return result is not JSON')

                calls = data.get("toolchain_calls", [])
                if not isinstance(calls, list):
                    raise ValueError('toolchain_calls is not a list')

                normalized_calls: List[Dict[str, Any]] = []
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    tool_name = call.get("tool") or call.get("tool_name")
                    if not tool_name:
                        continue
                    tool_name_str = str(tool_name)
                    key = self._normalize_tool_key(tool_name_str)
                    meta = self._last_tool_meta.get(key, {})
                    normalized_calls.append(
                        {
                            "tool": tool_name_str,
                            "tool_description": meta.get("tool_description", ""),
                            "input_parameters": meta.get("input_parameters", {}),
                            "output_parameters": meta.get("output_parameters", {}),
                        }
                    )

                if not normalized_calls:
                    normalized_calls = self._fallback_toolchain_from_tools(
                        question=question,
                        tools=normalized_tools,
                        max_calls=int(self.config.get("fallback_max_calls", 4)),
                    )
                    enriched_calls: List[Dict[str, Any]] = []
                    for c in normalized_calls:
                        tool_name_str = str(c.get("tool", ""))
                        key = self._normalize_tool_key(tool_name_str)
                        meta = self._last_tool_meta.get(key, {})
                        enriched_calls.append(
                            {
                                "tool": tool_name_str,
                                "tool_description": meta.get("tool_description", ""),
                                "input_parameters": meta.get("input_parameters", {}),
                                "output_parameters": meta.get("output_parameters", {}),
                            }
                        )
                    normalized_calls = enriched_calls

                return {"toolchain_calls": normalized_calls}
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1 and retry_sleep > 0:
                    time.sleep(retry_sleep)

        return {
            "toolchain_calls": [
                {
                    "tool": str(c.get("tool", "")),
                    "tool_description": self._last_tool_meta.get(
                        self._normalize_tool_key(str(c.get("tool", ""))), {}
                    ).get("tool_description", ""),
                    "input_parameters": self._last_tool_meta.get(
                        self._normalize_tool_key(str(c.get("tool", ""))), {}
                    ).get("input_parameters", {}),
                    "output_parameters": self._last_tool_meta.get(
                        self._normalize_tool_key(str(c.get("tool", ""))), {}
                    ).get("output_parameters", {}),
                }
                for c in self._fallback_toolchain_from_tools(
                    question=question,
                    tools=normalized_tools,
                    max_calls=int(self.config.get("fallback_max_calls", 4)),
                )
            ],
            "error": last_error or "zero_shot_spotify failed",
        }

    def generate_tool_dependencies(
        self,
        toolchain_calls: List[Dict[str, Any]],
        max_retries: int = 5,
        retry_sleep: float = 1.0,
    ) -> List[Dict[str, Any]]:
        '2: based on tool generate DAG dependency.'
        if not toolchain_calls:
            return []

        input_data = []
        for idx, call in enumerate(toolchain_calls, 1):
            tool_name = str(call.get("tool") or call.get("tool_name") or "")
            key = self._normalize_tool_key(tool_name)
            meta = self._last_tool_meta.get(key, {})
            input_data.append(
                {
                    "id": idx,
                    "tool_name": tool_name,
                    "tool_description": call.get("tool_description") or meta.get("tool_description", ""),
                    "input_parameters": call.get("input_parameters") or meta.get("input_parameters", {}),
                    "output_parameters": call.get("output_parameters") or meta.get("output_parameters", {}),
                }
            )
        prompt = PROMPT.Generate_tool_dependencies.format(
            result=json.dumps(input_data, ensure_ascii=False, indent=2)
        )

        last_error: Optional[str] = None
        for attempt in range(max(1, max_retries)):
            try:
                content = self._chat_json_object(prompt)
                if not content:
                    raise ValueError("LLM returnis empty")
                data = self._safe_parse_json(content)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for key in ["tasks", "dependencies", "plan", "result"]:
                        if isinstance(data.get(key), list):
                            return data[key]
                raise ValueError("not found available dependency list")
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1 and retry_sleep > 0:
                    time.sleep(retry_sleep)

        # : dependency
        fallback: List[Dict[str, Any]] = []
        for idx, call in enumerate(toolchain_calls, 1):
            fallback.append(
                {
                    "id": idx,
                    "tool": call.get("tool") or call.get("tool_name"),
                    "dependencies": [idx - 1] if idx > 1 else [],
                }
            )
        return fallback

    def self_reflect_and_save(
        self,
        task: str,
        toolchain_calls: List[Dict[str, Any]],
        reference_answer: str,
        dag_results: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_sleep: float = 1.0,
    ) -> str:
        '3: generatereflection write notebook.'
        dag_results_str = json.dumps(dag_results or {}, ensure_ascii=False, indent=2)
        prompt = PROMPT.self_reflection.format(
            task=task,
            result=json.dumps(toolchain_calls, ensure_ascii=False, indent=2),
            dag_results=dag_results_str,
            reference_answer=reference_answer,
        )

        reflection_text = ""
        last_error: Optional[str] = None
        for attempt in range(max(1, max_retries)):
            try:
                content = self._chat_json_object(prompt)
                if not content:
                    raise ValueError("LLM returnis empty")
                data = self._safe_parse_json(content)
                if not isinstance(data, dict):
                    raise ValueError('reflection result is not JSON')
                reflection_text = str(data.get("self_reflection", "")).strip()
                if reflection_text:
                    break
                raise ValueError("self_reflection is empty")
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1 and retry_sleep > 0:
                    time.sleep(retry_sleep)

        if not reflection_text:
            reflection_text = f"Reflection generation failed: {last_error or 'unknown error'}"

        # save notebook()
        entries: List[Dict[str, Any]] = []
        if os.path.exists(self.notebook_path):
            try:
                with open(self.notebook_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    entries = data
            except Exception:
                entries = []

        entries.append(
            {
                "task": task,
                "query": task,
                "toolchain_calls": toolchain_calls,
                "reference_answer": reference_answer,
                "dag_results": dag_results or {},
                "self_reflection": reflection_text,
            }
        )

        os.makedirs(os.path.dirname(self.notebook_path), exist_ok=True)
        with open(self.notebook_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        return reflection_text


__all__ = ["SpotifyAgent"]
