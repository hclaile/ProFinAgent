import json
import os
import time
import traceback
import re
import numpy as np
import threading
import builtins
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Union, Optional

# optional dependency: complete/normalize JSON(JSONDecode Error: Unterminated string)
try:
    from json_repair import repair_json  # type: ignore
    _JSON_REPAIR_AVAILABLE = True
except Exception:
    repair_json = None  # type: ignore
    _JSON_REPAIR_AVAILABLE = False

# optional dependency: token (unavailable)
try:
    import tiktoken  # type: ignore
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False

try:
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

# root directory sys. path, prompt module
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import prompt.test_agent as PROMPT  # type: ignore

try:
    # config module, judge_agent/base_agent
    import agent.agent_config as agent_cfg  # type: ignore
    _AGENT_CONFIG_AVAILABLE = True
except ImportError:
    agent_cfg = None  # type: ignore
    _AGENT_CONFIG_AVAILABLE = False

try:
    from agent.rag import CustomEmbeddingFunction # type: ignore
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

# (base_agent.py: log _debug_print,)
# - default (benchmark, I/O)
# - environment variable:ProFinAgent_DEBUG_LOG=1(compatible DEBUG_LOG=1)
_DEBUG_LOG = False

def _debug_print(*args, **kwargs) -> None:
    ', _DEBUG_LOG output.'
    if _DEBUG_LOG:
        builtins.print(*args, **kwargs)

class TestAgent:

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # must-have/quota config: supports configs/must_have_rules.json update
        self._tool_selection_rules_cache: Dict[str, Any] = {}
        # embedding/similarity LRU cache(embedding similarity call)
        self._embedding_cache: 'OrderedDict[tuple, List[float]]' = OrderedDict()
        self._similarity_cache: 'OrderedDict[tuple, float]' = OrderedDict()
        self._embedding_cache_max = int(os.getenv("ProFinAgent_EMBED_CACHE_MAX", "4096"))
        self._similarity_cache_max = int(os.getenv("ProFinAgent_SIM_CACHE_MAX", "16384"))
        self._cache_lock = threading.Lock()

        if not _OPENAI_AVAILABLE:
            raise ImportError(
                'not installed openai, initialize Test Agent.'
                "'pip install openai' retry."
            )

        if not _AGENT_CONFIG_AVAILABLE:
            raise ImportError(
                'agent_config,/agent/agent_config.py exists available.'
            )

        # agent_config read Test Agent config
        apikey = getattr(agent_cfg, "test_agent_apikey", None) or os.getenv(
            "OPENAI_API_KEY", ""
        )
        if not apikey:
            raise ValueError(
                'Test Agent initialize failed: agent_config.test_agent_apikey environment variable'
                'OPENAI_API_KEY API Key.'
            )

        base_url = getattr(
            agent_cfg,
            "test_agent_base_url",
        )

        self.model = getattr(
            agent_cfg,
            "test_agent_model",
        )

        self.client = OpenAI(api_key=apikey, base_url=base_url)
        
        self.temperature = getattr(agent_cfg, "TEMPERATURE", 0.0)
        self.max_tokens = getattr(agent_cfg, "MAX_TOKENS", 6144)
        # manual embedding/notebook parameter:
        # 1) tool embedding threshold, LLM
        # 2) notebook return top-k similar, LLM
        self.tool_embedding_threshold = float(
            self.config.get(
                "tool_embedding_threshold",
                os.getenv(
                    "ProFinAgent_TOOL_EMBED_THRESHOLD",
                    getattr(agent_cfg, "TEST_AGENT_TOOL_EMBEDDING_THRESHOLD", 0.2),
                ),
            )
        )
        self.notebook_match_threshold = float(
            self.config.get(
                "notebook_match_threshold",
                os.getenv(
                    "ProFinAgent_NOTEBOOK_MATCH_THRESHOLD",
                    getattr(agent_cfg, "TEST_AGENT_NOTEBOOK_MATCH_THRESHOLD", 0.70),
                ),
            )
        )
        self.notebook_top_k = max(
            1,
            int(
                self.config.get(
                    "notebook_top_k",
                    os.getenv(
                        "ProFinAgent_NOTEBOOK_TOP_K",
                        getattr(agent_cfg, "TEST_AGENT_NOTEBOOK_TOP_K", 1),
                    ),
                )
            ),
        )
        
        # notebook path, default notebook_api.json
        # self.notebook_path = os. path. abs path(
        # os. path. join(os. path. dirname(__file__), "./ProFinAgent/notebook/notebook_api.json")
        # )
        self.notebook_path = './ProFinAgent/notebook/notebook_api.json'
        
        # initialize embedding notebook
        self._embedding_function = None
        if _RAG_AVAILABLE and hasattr(agent_cfg, "EMBEDDING_MODEL_PATH"):
            try:
                self._embedding_function = CustomEmbeddingFunction(
                    model_path=agent_cfg.EMBEDDING_MODEL_PATH,
                    matryoshka_dim=getattr(agent_cfg, "MATRYOSHKA_DIM", None)
                )
            except Exception as e:
                _debug_print(f"[WARN] Test Agent initialize embedding failed: {e}")

    # === LRU helpers ===
    def _lru_get(self, cache: 'OrderedDict[tuple, Any]', key: tuple) -> Any:
        try:
            with self._cache_lock:
                val = cache.get(key, None)
                if val is not None:
                    cache.move_to_end(key)
                return val
        except Exception:
            return None

    def _lru_set(self, cache: 'OrderedDict[tuple, Any]', key: tuple, value: Any, maxsize: int) -> None:
        try:
            with self._cache_lock:
                cache[key] = value
                cache.move_to_end(key)
                while len(cache) > maxsize:
                    cache.popitem(last=False)
        except Exception:
            return

    def _cache_key_pair(self, query_text: str, doc_text: str) -> tuple:
        return (hash(query_text), len(query_text), hash(doc_text), len(doc_text))

    # === Embedding cache wrappers ===
    def _embed_query_cached(self, text: str) -> List[float]:
        if self._embedding_function is None:
            raise RuntimeError('Embedding initialize')
        key = ("q", text)
        cached = self._lru_get(self._embedding_cache, key)
        if cached is not None:
            return cached
        vec = self._embedding_function.embed_query(text)
        self._lru_set(self._embedding_cache, key, vec, self._embedding_cache_max)
        return vec

    def _embed_documents_cached(self, texts: List[str]) -> List[List[float]]:
        if self._embedding_function is None:
            raise RuntimeError('Embedding initialize')
        if not texts:
            return []
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_texts: List[str] = []
        miss_idx: List[int] = []
        for i, t in enumerate(texts):
            key = ("d", t)
            cached = self._lru_get(self._embedding_cache, key)
            if cached is not None:
                results[i] = cached
            else:
                miss_texts.append(t)
                miss_idx.append(i)
        if miss_texts:
            bs = int(os.getenv("ProFinAgent_EMBED_BATCH_SIZE", "32"))
            bs = max(1, bs)
            total = len(miss_texts)
            for start in range(0, total, bs):
                end = min(start + bs, total)
                batch = miss_texts[start:end]
                # log: default batchsummary, " "
                if _DEBUG_LOG or total > bs * 4:
                    _debug_print(f"[INFO] embed_documents batch {start//bs + 1}/{(total + bs - 1)//bs} (size={len(batch)})")
                vecs = self._embedding_function.embed_documents(batch)
                for j, vec in enumerate(vecs):
                    idx = miss_idx[start + j]
                    t = miss_texts[start + j]
                    results[idx] = vec
                    self._lru_set(self._embedding_cache, ("d", t), vec, self._embedding_cache_max)
        return [r for r in results if r is not None]  # type: ignore

    # === Two-stage rough filter ===
    def _extract_query_keywords(self, query: str, max_keywords: int = 24) -> List[str]:
        q = (query or "").lower()
        if not q:
            return []
        tokens = re.findall(r"[a-z][a-z0-9_+\-]{1,}", q)
        zh = re.findall(r"[\u4e00-\u9fff]{2,}", query or "")
        tokens.extend([z.strip() for z in zh if z.strip()])
        stop = {
            "the", "and", "with", "from", "for", "to", "of", "in", "on", "a", "an", "as", "by",
            "calculate", "compute", "using", "factor", "return", "returns", "average",
        }
        uniq: List[str] = []
        for t in tokens:
            t = t.strip()
            if not t or t in stop:
                continue
            if t not in uniq:
                uniq.append(t)
            if len(uniq) >= max_keywords:
                break
        return uniq

    def _rough_score_text(self, text: str, keywords: List[str]) -> float:
        if not text or not keywords:
            return 0.0
        tl = text.lower()
        score = 0.0
        for i, kw in enumerate(keywords):
            if kw in tl:
                score += 1.0
                if i < 8:
                    c = tl.count(kw)
                    score += min(2.0, 0.2 * c)
        return score

    def _rough_select_indices(self, texts: List[str], query: str, max_candidates: int) -> List[int]:
        n = len(texts)
        if n <= max_candidates:
            return list(range(n))
        kws = self._extract_query_keywords(query)
        if not kws:
            step = max(1, n // max_candidates)
            idxs = list(range(0, n, step))[:max_candidates]
            idxs.append(0)
            idxs.append(n - 1)
            return sorted(set(idxs))
        scored = [(i, self._rough_score_text(texts[i], kws)) for i in range(n)]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [i for i, _ in scored[:max_candidates]]
        top.append(0)
        top.append(n - 1)
        return sorted(set(top))

    # === Batch similarity ===
    def _compute_similarity_batch(self, query_vec: List[float], doc_vecs: List[List[float]]) -> List[float]:
        if not doc_vecs:
            return []
        if self._is_qwen3_model():
            try:
                model = self._embedding_function.model  # type: ignore[attr-defined]
                q = np.array([query_vec], dtype=np.float32)
                d = np.array(doc_vecs, dtype=np.float32)
                sims = model.similarity(q, d)
                sims_np = np.array(sims, dtype=np.float32).reshape(-1)
                _debug_print(f"[INFO] Qwen3 batch similarity computed for {len(doc_vecs)} docs")
                return [float(x) for x in sims_np.tolist()]
            except Exception as e:
                _debug_print(f"[WARN] Qwen3 batch similarity failed,: {e}")
        qv = np.array(query_vec, dtype=np.float32)
        dv = np.array(doc_vecs, dtype=np.float32)
        q_norm = float(np.linalg.norm(qv) + 1e-12)
        d_norm = np.linalg.norm(dv, axis=1) + 1e-12
        sims = (dv @ qv) / (d_norm * q_norm)
        return [float(x) for x in sims.tolist()]

    def _format_tools(self, tools: Union[str, Dict[str, Any], List[Any]]) -> str:
        'tools parameter convert prompt.'
        if isinstance(tools, str):
            path = os.path.expanduser(tools)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                return json.dumps(obj, ensure_ascii=False, indent=2)
            return tools

        if isinstance(tools, (dict, list)):
            return json.dumps(tools, ensure_ascii=False, indent=2)

        return str(tools)

    def _load_notebook_entries(self) -> List[Dict[str, Any]]:
        'read notebook_api.json, return list.'
        if not self.notebook_path or not os.path.exists(self.notebook_path):
            return []
        try:
            with open(self.notebook_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # print("load success")
            if isinstance(data, list):
                return data
        except Exception as e:
            _debug_print(f"[WARN] read notebook failed: {e}")
        return []

    def _save_notebook_entries(self, entries: List[Dict[str, Any]]) -> None:
        'save notebook list.'
        if not self.notebook_path:
            return
        try:
            os.makedirs(os.path.dirname(self.notebook_path), exist_ok=True)
            with open(self.notebook_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _debug_print(f"[WARN] save notebook failed: {e}")

    def _is_qwen3_model(self) -> bool:
        'check current use embedding model Qwen3 model Returns: True Qwen3 model, False'
        if self._embedding_function is None:
            return False
        model_type = getattr(self._embedding_function, 'model_type', None)
        return model_type == 'qwen3'
    
    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        'calculate similar use Qwen3 model, use model similarity(); use similar calculate. Args: vec1: 1(query) vec2: 2() Returns: similar (range:-1 1, similar)'
        # use Qwen3 model, use model similarity()
        if self._is_qwen3_model():
            try:
                # get model
                model = self._embedding_function.model
                
                # check model similarity
                if not hasattr(model, 'similarity'):
                    _debug_print(f"[WARN] Qwen3 model similarity(), similar")
                    return self._compute_cosine_similarity(vec1, vec2)
                
                # convert numpy (Qwen3 similarity numpy)
                # , similarity(query_embeddings, document_embeddings)
                # query_embeddings document_embeddings list
                # , list [1, dim]
                vec1_array = np.array([vec1], dtype=np.float32)
                vec2_array = np.array([vec2], dtype=np.float32)
                
                # call model similarity
                # return tensor, shape [1, 1](query,)
                similarity_result = model.similarity(vec1_array, vec2_array)
                
                # extractsimilar
                # similarity_result tensor, numpy
                if hasattr(similarity_result, 'item'):
                    # Py Torch tensor, use item() extract
                    score = float(similarity_result.item())
                elif hasattr(similarity_result, '__getitem__'):
                    # numpy structure, extract [0, 0] position
                    try:
                        score = float(similarity_result[0, 0])
                    except (IndexError, TypeError):
                        # failed, try direct convert
                        score = float(similarity_result)
                else:
                    #
                    score = float(similarity_result)
                
                _debug_print(f"[INFO] use Qwen3 model. similarity() calculate similarity: {score:.4f}")
                return score
            except Exception as e:
                _debug_print(f"[WARN] Qwen3 model. similarity() call failed, similar: {e}")
                traceback.print_exc()
                # similar
                return self._compute_cosine_similarity(vec1, vec2)
        else:
            # Qwen3 model, use similar
            return self._compute_cosine_similarity(vec1, vec2)
    
    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        'calculate similar (Qwen3 model) Args: vec1: 1 vec2: 2 Returns: similar (range:-1 1, similar)'
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    # === Token budget helpers (prompt max_seq_len) ===
    def _estimate_tokens(self, text: str) -> int:
        'token count(prefer tiktoken;),.: model tokenizer,.'
        if not text:
            return 0

        base_estimate = 0
        if _TIKTOKEN_AVAILABLE and tiktoken is not None:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                base_estimate = len(enc.encode(text))
            except Exception:
                base_estimate = 0

        if base_estimate == 0:
            # : Chinese 1.5/token, English 4/token
            chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
            total_chars = len(text)
            base_estimate = int(chinese_chars / 1.5 + (total_chars - chinese_chars) / 4)
            base_estimate = max(base_estimate, total_chars // 4)

        return int(base_estimate * 1.2)  # 20%

    def _truncate_text(self, text: str, max_tokens: int, preserve_end: bool = False) -> str:
        'token (keepstructure).'
        if not text:
            return text

        if self._estimate_tokens(text) <= max_tokens:
            return text

        if _TIKTOKEN_AVAILABLE and tiktoken is not None:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                toks = enc.encode(text)
                toks = toks[-max_tokens:] if preserve_end else toks[:max_tokens]
                return enc.decode(toks)
            except Exception:
                pass

        # :
        max_chars = int(max_tokens * 2.5)
        return text[-max_chars:] if preserve_end else text[:max_chars]

    def _find_similar_notebook_entries(
        self,
        task: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        'use embedding notebook, return threshold top-k result.'
        entries = self._load_notebook_entries()
        if not entries or self._embedding_function is None:
            return []

        threshold = self.notebook_match_threshold if threshold is None else threshold
        top_k = self.notebook_top_k if top_k is None else max(1, int(top_k))

        try:
            query_vec = self._embed_query_cached(task)
        except Exception as e:
            _debug_print(f"[WARN] notebook generate query failed: {e}")
            return []

        valid_entries: List[Dict[str, Any]] = []
        task_texts: List[str] = []
        for entry in entries:
            task_text = entry.get("task", "")
            if not task_text:
                continue
            valid_entries.append(entry)
            task_texts.append(task_text)

        if not task_texts:
            return []

        try:
            doc_vecs = self._embed_documents_cached(task_texts)
            scores = self._compute_similarity_batch(query_vec, doc_vecs)
        except Exception as e:
            _debug_print(f"[WARN] notebook similar calculate failed: {e}")
            return []

        matched: List[Dict[str, Any]] = []
        for entry, score in zip(valid_entries, scores):
            score = float(score)
            if score >= threshold:
                enriched = dict(entry)
                enriched["_similarity"] = score
                matched.append(enriched)

        matched.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)
        return matched[:top_k]
    
    def _load_tool_type_config(self) -> Dict[str, Any]:
        'load tool_type.json config.'
        # tool_type_path = os. path. abs path(
        # os. path. join(os. path. dirname(__file__), '../configs/tool_type.json')
        # )
        tool_type_path = './ProFinAgent/configs/tool_type.json'
        if not os.path.exists(tool_type_path):
            raise FileNotFoundError(f"tool_type.json file does not exist: {tool_type_path}")
        
        with open(tool_type_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def choose_tool_types(
        self,
        query: str,
        max_retries: int = 10,
        retry_sleep: float = 1.0,
    ) -> List[str]:
        _debug_print(f"[INFO] start tool type: {query[:100]}...")
        
        # load tool_type.json
        tool_type_config = self._load_tool_type_config()
        
        # tool type list(format:tool_type: description)
        tool_types_list = []
        valid_tool_types = set()
        for item in tool_type_config:
            tool_type = item.get('tool_type', '')
            description = item.get('description', '')
            if tool_type:
                valid_tool_types.add(tool_type)
                tool_types_list.append(f"{tool_type}: {description}")
        
        tool_types_str = ''.join(tool_types_list)
        
        # prompt
        prompt = PROMPT.choose_types.format(
            context=PROMPT.context,
            query=query,
            tool_types=tool_types_str,
        )
        
        total_attempts = max(1, max_retries)
        last_error: Any = None
        
        for attempt in range(total_attempts):
            try:
                # call LLM
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Please output JSON only."}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("LLM return contentis empty")

                # JSON()
                data = self._safe_parse_json(content)
                if data is None:
                    raise ValueError('return content JSON')
                
                # format
                if not isinstance(data, dict):
                    raise ValueError(f"return result is not JSON: {type(data)}")
                
                tool_types = data.get("tool_types", [])
                if not isinstance(tool_types, list):
                    raise ValueError("return JSON 'tool_types' is not a list")
                
                # tool type exists tool_type.json
                invalid_types = []
                for tool_type in tool_types:
                    if not isinstance(tool_type, str):
                        invalid_types.append(f"{tool_type} (is not)")
                    elif tool_type not in valid_tool_types:
                        invalid_types.append(tool_type)
                
                if invalid_types:
                    raise ValueError(
                        f"tool type does not exist tool_type.json: {invalid_types}"
                    )
                
                _debug_print(f"[SUCCESS] successful tool type: {tool_types}")
                return tool_types
                
            except Exception as e:
                last_error = str(e)
                _debug_print(f"[WARN] Test Agent {attempt + 1}/{total_attempts} tool type failed: {e}")
                if attempt < total_attempts - 1 and retry_sleep > 0:
                    time.sleep(retry_sleep)
        
        # retry failed, return empty list exception
        raise RuntimeError(
            f"maximum retry ({max_retries}), tool type."
            f"error: {last_error}"
        )
    
    def _filter_tools_by_embedding(
        self,
        query: str,
        tool_types: List[str],
        tools_config: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        'query tool_types using embedding filter top k tool. Args: query: user query tool_types: tool type list tools_config: tools.json complete config (format) Returns: List[Dict[str, Any]]: filter tool list (format), name, description, inputSchema, outputSchema, example'
        _debug_print(f"[INFO] start embedding filter tool...")
        _debug_print(f"[INFO] query: {query[:100]}...")
        _debug_print(f"[INFO] tool type: {tool_types}")
        
        # check embedding available
        if self._embedding_function is None:
            raise RuntimeError(
                'Embedding initialize, embedding.'
            )
        
        # tools_config format: dict, format, convert; list, format
        if isinstance(tools_config, dict):
            # format, convert format
            all_tools = []
            if 'categories' in tools_config:
                for category in tools_config['categories']:
                    tools = category.get('tools', [])
                    all_tools.extend(tools)
            tools_list = all_tools
        elif isinstance(tools_config, list):
            # format, directly use
            tools_list = tools_config
        else:
            raise ValueError(f"tools_config format unsupported: {type(tools_config)}")
        
        # tool_type.json tool name and tool type
        tool_name_to_type = {}
        try:
            # tool_type_path = os. path. abs path(
            # os. path. join(os. path. dirname(__file__), '../configs/tool_type.json')
            # )
            tool_type_path = './ProFinAgent/configs/tool_type.json'
            if os.path.exists(tool_type_path):
                with open(tool_type_path, 'r', encoding='utf-8') as f:
                    tool_type_config = json.load(f)
                # tool_type.json format, tool_type, tools (tool name list)
                if isinstance(tool_type_config, list):
                    for category in tool_type_config:
                        cat_tool_type = category.get('tool_type', '')
                        tools_names = category.get('tools', [])
                        for tool_name in tools_names:
                            tool_name_to_type[tool_name] = cat_tool_type
        except Exception as e:
            _debug_print(f"[WARN] load tool_type.json failed: {e}")
        
        # tools_list extract tool_types tool
        candidate_tools = []  # [(tool_dict, tool_type),...]
        
        for tool in tools_list:
            if not isinstance(tool, dict):
                continue
            tool_name = tool.get('name', '')
            if not tool_name:
                continue
            # tool type
            tool_type = tool_name_to_type.get(tool_name, '')
            if tool_type in tool_types:
                candidate_tools.append((tool, tool_type))
        
        _debug_print(f"[INFO] candidate tool: {len(candidate_tools)}")
        
        if len(candidate_tools) == 0:
            _debug_print(f"[WARN] tool, return empty list")
            return []
        
        # ===: must-have + per-type quota + global top (, tool top-k) ===
        # configs/must_have_rules.json read(supports update/), config use default,.
        from agent.tool_selection_rules import load_tool_selection_strategy  # type: ignore

        env = (
            self.config.get("must_have_rules_env")
            or os.getenv("ProFinAgent_RULES_ENV", None)
        )
        strategy = load_tool_selection_strategy(
            env=env,
            cache=self._tool_selection_rules_cache,
            debug_print=_debug_print,
        )
        PER_TYPE_QUOTA = int(strategy.get("per_type_quota", 1))
        MIN_TOTAL_TOOLS = int(strategy.get("min_total_tools", 1))
        MAX_TOTAL_TOOLS = int(strategy.get("max_total_tools", 16))
        MUST_HAVE_RULES: List[tuple[List[str], List[str]]] = strategy.get("must_have_rules", [])  # type: ignore

        def _get_must_have_tool_names(q: str) -> List[str]:
            import re
            qn = (q or "").lower()
            names: List[str] = []
            for patterns, tool_names in MUST_HAVE_RULES:
                for pat in patterns:
                    try:
                        if re.search(pat, qn, flags=re.IGNORECASE):
                            for n in tool_names:
                                if n not in names:
                                    names.append(n)
                            break
                    except re.error:
                        if pat.lower() in qn:
                            for n in tool_names:
                                if n not in names:
                                    names.append(n)
                            break
            return names

        def _find_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
            for t in tools_list:
                if isinstance(t, dict) and t.get("name") == name:
                    return t
            return None

        # must-have tool current candidate(tool_type), tools_list candidate
        must_have_names = _get_must_have_tool_names(query)
        if must_have_names:
            for n in must_have_names:
                tool_obj = _find_tool_by_name(n)
                if not tool_obj:
                    continue
                actual_type = tool_name_to_type.get(n, "")
                if not any(isinstance(td, dict) and td.get("name") == n for td, _ in candidate_tools):
                    candidate_tools.append((tool_obj, actual_type))

        _debug_print(f"[INFO] candidate tool (must-have): {len(candidate_tools)}; must-have: {must_have_names}")

        # --- use embedding (candidate) ---
        tool_scores: List[Dict[str, Any]] = []
        enhanced_query = f"Query: {query}."
        try:
            query_vec = self._embedding_function.embed_query(enhanced_query)
        except Exception as e:
            _debug_print(f"[WARN] generate query failed: {e}")
            raise

        for tool_dict, tool_type in candidate_tools:
            tool_name = tool_dict.get("name", "")
            tool_desc = tool_dict.get("description", "")
            if not tool_name or not tool_desc:
                continue
            try:
                tool_doc = f"Tool: {tool_name}. Description: {tool_desc}."
                tool_vec = self._embedding_function.embed_documents([tool_doc])[0]
                score = float(self._compute_similarity(query_vec, tool_vec))
                tool_scores.append({"tool": tool_dict, "tool_type": tool_type, "score": score})
            except Exception as e:
                _debug_print(f"[WARN] tool {tool_name} failed: {e}")
                continue

        if not tool_scores:
            _debug_print('[WARN] tool success, return empty list')
            return []

        # global sort(prefer), keep original sort.
        tool_scores.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        ranked_tool_scores = list(tool_scores)

        thresholded_tool_scores = [
            item for item in tool_scores
            if float(item.get("score", 0.0)) > self.tool_embedding_threshold
        ]
        _debug_print(
            f"[INFO] embedding threshold {len(thresholded_tool_scores)}/{len(tool_scores)} tool"
            f"(threshold>{self.tool_embedding_threshold})"
        )
        if not thresholded_tool_scores:
            _debug_print('[WARN] tool embedding threshold, use embedding sort 5 candidate')
            tool_scores = ranked_tool_scores[:5]
        else:
            tool_scores = thresholded_tool_scores

        # threshold prefer.
        tool_scores.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # --- per-type quota: tool_type keep PER_TYPE_QUOTA ---
        from collections import defaultdict
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in tool_scores:
            by_type[item.get("tool_type", "")].append(item)

        selected_by_name: Dict[str, Dict[str, Any]] = {}

        # must-have()
        for n in must_have_names:
            for item in tool_scores:
                if item.get("tool", {}).get("name") == n:
                    selected_by_name[n] = item
                    break

        # per-type quota(top-N)
        for t, items in by_type.items():
            if not items:
                continue
            for item in items[:PER_TYPE_QUOTA]:
                name = item.get("tool", {}).get("name", "")
                if name:
                    selected_by_name.setdefault(name, item)

        # global top MIN_TOTAL_TOOLS(MAX_TOTAL_TOOLS)
        for item in tool_scores:
            if len(selected_by_name) >= MAX_TOTAL_TOOLS:
                break
            name = item.get("tool", {}).get("name", "")
            if not name:
                continue
            if len(selected_by_name) < MIN_TOTAL_TOOLS or name in must_have_names:
                selected_by_name.setdefault(name, item)
            elif len(selected_by_name) < MAX_TOTAL_TOOLS:
                selected_by_name.setdefault(name, item)

        selected = list(selected_by_name.values())
        selected.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        if len(selected) < 3:
            selected = ranked_tool_scores[:5]
            _debug_print(
                f"[INFO] filter result 3, embedding sort {len(selected)} tool LLM"
            )

        _debug_print(f"[SUCCESS] {len(selected)} tool(quota={PER_TYPE_QUOTA}, min={MIN_TOTAL_TOOLS}, max={MAX_TOTAL_TOOLS}):")
        for i, item in enumerate(selected, 1):
            tool_name = item.get("tool", {}).get("name", "Unknown")
            score = item.get("score", 0.0)
            t = item.get("tool_type", "")
            _debug_print(f"{i}. {tool_name} (score: {score:.4f}, type: {t})")

        # output(format), field
        filtered_tools: List[Dict[str, Any]] = []
        for item in selected:
            tool = item.get("tool", {}) or {}
            filtered_tools.append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema"),
                    "outputSchema": tool.get("outputSchema"),
                    "example": tool.get("example"),
                }
            )

        _debug_print(f"[SUCCESS] filter complete, return {len(filtered_tools)} tool")
        return filtered_tools

    def test_straightforward(
        self,
        question: str,
        tools: Union[str, Dict[str, Any], List[Any]],
        max_retries: int = 10,
        retry_sleep: float = 1.0,
    ) -> Dict[str, Any]:
        'directly use tool generate tool call, tool_type embedding filter. use answer_straightforward prompt template, directly use tool, use notebook. Args: question: user question. tools: available tool: - tools.json file path - dict/list(tools.json content) - format (direct prompt) max_retries: LLM call JSON failed; maximum retry. retry_sleep: retry interval(). Returns: dict: "toolchain_calls" JSON.'
        if not isinstance(question, str) or not question.strip():
            raise ValueError('question empty.')

        # load complete tools.json config(filter)
        _debug_print(f"[INFO] load tool config (directly use tool)...")
        if isinstance(tools, str):
            tools_path = os.path.expanduser(tools)
            if os.path.isfile(tools_path):
                with open(tools_path, "r", encoding="utf-8") as f:
                    full_tools_config = json.load(f)
            else:
                raise FileNotFoundError(f"tool config file does not exist: {tools_path}")
        elif isinstance(tools, (dict, list)):
            full_tools_config = tools
        else:
            raise ValueError(f"tools parameter format unsupported: {type(tools)}")
        
        # full_tools_config list format()
        if isinstance(full_tools_config, dict):
            # format, convert format
            tools_list = []
            if 'categories' in full_tools_config:
                for category in full_tools_config['categories']:
                    tools_list.extend(category.get('tools', []))
            full_tools_config = tools_list
        elif not isinstance(full_tools_config, list):
            raise ValueError(f"tools_config format unsupported: {type(full_tools_config)}")
        
        # keep field
        filtered_tools = []
        for tool in full_tools_config:
            if isinstance(tool, dict):
                filtered_tools.append({
                    'name': tool.get('name', ''),
                    'description': tool.get('description', ''),
                    'inputSchema': tool.get('inputSchema'),
                    'outputSchema': tool.get('outputSchema'),
                    'example': tool.get('example')
                })
        
        # format tool config (use notebook)
        tools_str = json.dumps(filtered_tools, ensure_ascii=False, indent=2)
        
        # call LLM generate tool call
        total_attempts = max(1, max_retries)
        last_error: Any = None

        for attempt in range(total_attempts):
            try:
                system_content = PROMPT.answer_straightforward.format(
                    context=PROMPT.context,
                    question=question,
                    tools=tools_str,
                )

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": "Please output JSON only."}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("LLM return contentis empty")

                data = self._safe_parse_json(content)
                if data is None:
                    raise ValueError('return content JSON')
                if "toolchain_calls" not in data:
                    raise ValueError("return JSON missing 'toolchain_calls' field")
                
                # call field
                tc = data.get("toolchain_calls", [])
                if not isinstance(tc, list):
                    raise ValueError("return JSON 'toolchain_calls' is not a list")
                
                for idx, call in enumerate(tc):
                    if not isinstance(call, dict):
                        raise ValueError(f"toolchain_calls[{idx}] is not: {call!r}")
                    if "tool" not in call or "arguments" not in call:
                        raise ValueError(
                            f"toolchain_calls[{idx}] missing 'tool' 'arguments' field: {call!r}"
                        )
                
                return data

            except Exception as e:
                last_error = str(e)
                _debug_print(f"[WARN] Test Agent {attempt + 1}/{total_attempts} generate toolchain_calls failed: {e}")
                if attempt < total_attempts - 1 and retry_sleep > 0:
                    time.sleep(retry_sleep)

        return {
            "toolchain_calls": [],
            "error": last_error or 'LLM retry return JSON result',
        }
    
    def test(
        self,
        question: str,
        tools: Union[str, Dict[str, Any], List[Any]],
        max_retries: int = 5,
        retry_sleep: float = 1.0,
    ) -> Dict[str, Any]:
        'call LLM generate tool call, return toolchain_calls.'
        if not isinstance(question, str) or not question.strip():
            raise ValueError('question empty.')

        # 1: tool type
        _debug_print(f"[INFO] 1: tool type...")
        try:
            tool_types = self.choose_tool_types(query=question, max_retries=max_retries, retry_sleep=retry_sleep)
        except Exception as e:
            _debug_print(f"[WARN] tool type failed: {e}, use tool")
            tool_types = []
        
        # 2: load complete tools.json config
        _debug_print(f"[INFO] 2: load tool config...")
        if isinstance(tools, str):
            tools_path = os.path.expanduser(tools)
            if os.path.isfile(tools_path):
                with open(tools_path, "r", encoding="utf-8") as f:
                    full_tools_config = json.load(f)
            else:
                raise FileNotFoundError(f"tool config file does not exist: {tools_path}")
        elif isinstance(tools, (dict, list)):
            full_tools_config = tools
        else:
            raise ValueError(f"tools parameter format unsupported: {type(tools)}")
        
        # full_tools_config list format()
        if isinstance(full_tools_config, dict):
            # format, convert format
            tools_list = []
            if 'categories' in full_tools_config:
                for category in full_tools_config['categories']:
                    tools_list.extend(category.get('tools', []))
            full_tools_config = tools_list
        elif not isinstance(full_tools_config, list):
            raise ValueError(f"tools_config format unsupported: {type(full_tools_config)}")
        
        # 3: tool type, use embedding filter tool; use tool
        if tool_types:
            _debug_print(f"[INFO] 3: embedding filter tool...")
            try:
                filtered_tools_list = self._filter_tools_by_embedding(
                    query=question,
                    tool_types=tool_types,
                    tools_config=full_tools_config,
                )
                if not filtered_tools_list:
                    raise ValueError('embedding filter return candidate tool')
                tools_str = json.dumps(filtered_tools_list, ensure_ascii=False, indent=2)
            except Exception as e:
                _debug_print(f"[WARN] tool filter failed: {e}, use tool")
                # filter failed, use tool, keep field
                filtered_all = []
                for tool in full_tools_config:
                    if isinstance(tool, dict):
                        filtered_all.append({
                            'name': tool.get('name', ''),
                            'description': tool.get('description', ''),
                            'inputSchema': tool.get('inputSchema'),
                            'outputSchema': tool.get('outputSchema'),
                            'example': tool.get('example')
                        })
                tools_str = json.dumps(filtered_all, ensure_ascii=False, indent=2)
        else:
            _debug_print(f"[INFO] 3: use tool(tool type)")
            # keep field
            filtered_all = []
            for tool in full_tools_config:
                if isinstance(tool, dict):
                    filtered_all.append({
                        'name': tool.get('name', ''),
                        'description': tool.get('description', ''),
                        'inputSchema': tool.get('inputSchema'),
                        'outputSchema': tool.get('outputSchema'),
                        'example': tool.get('example')
                    })
            tools_str = json.dumps(filtered_all, ensure_ascii=False, indent=2)
        
        # 4: similar notebook
        # notebook_str = ""
        notebook_entries = self._find_similar_notebook_entries(question)
        notebook_str = json.dumps(notebook_entries, ensure_ascii=False, indent=2) if notebook_entries else "[]"
        # print("notebook_str text:"+str(len(notebook_str)))
        
        # 5: call LLM generate tool call
        total_attempts = max(1, max_retries)
        last_error: Any = None

        for attempt in range(total_attempts):
            try:
                system_content = PROMPT.Test_toolchain.format(
                    context=PROMPT.context,
                    question=question,
                    tools=tools_str,
                    notebook=notebook_str
                )

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": "Please select at least one tool and output JSON only."}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("LLM return contentis empty")

                data = self._safe_parse_json(content)
                if data is None:
                    raise ValueError('return content JSON')
                if "toolchain_calls" not in data:
                    raise ValueError("return JSON missing 'toolchain_calls' field")
                tc = data.get("toolchain_calls", [])
                if not isinstance(tc, list):
                    raise ValueError("return JSON 'toolchain_calls' is not a list")
                if len(tc) == 0:
                    raise ValueError("return JSON 'toolchain_calls' is empty, tool")
                
                return data

            except Exception as e:
                last_error = str(e)
                _debug_print(f"[WARN] Test Agent {attempt + 1}/{total_attempts} generate toolchain_calls failed: {e}")
                if attempt < total_attempts - 1 and retry_sleep > 0:
                    time.sleep(retry_sleep)

        return {
            "toolchain_calls": [],
            "error": last_error or 'LLM retry return JSON result',
        }

    def generate_tool_dependencies(
        self,
        toolchain_calls: List[Dict[str, Any]],
        max_retries: int = 5
    ) -> List[Dict[str, Any]]:
        'analysis tool call dependencies.'
        if not toolchain_calls:
            return []

        input_data = []
        for idx, call in enumerate(toolchain_calls, 1):
            input_data.append({
                "id": idx,
                "tool_name": call.get("tool"),
                "arguments": call.get("arguments")
            })

        result_json = json.dumps(input_data, ensure_ascii=False, indent=2)
        
        for attempt in range(max_retries):
            try:
                prompt = PROMPT.Generate_tool_dependencies.format(
                    context=PROMPT.context,
                    result=result_json
                )

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Please output JSON only. If you need to wrap it in an object, use the key 'tasks'."}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                content = resp.choices[0].message.content
                if not content:
                    continue

                # try extract <PLAN> label content(exists)
                if "<PLAN>" in content and "</PLAN>" in content:
                    plan_content = content.split("<PLAN>")[1].split("</PLAN>")[0].strip()
                    try:
                        data = json.loads(plan_content)
                        if isinstance(data, list):
                            return data
                    except:
                        pass

                # try JSON
                try:
                    data = self._safe_parse_json(content)
                    if data is None:
                        raise json.JSONDecodeError("cannot parse json", content, 0)
                except json.JSONDecodeError:
                    _debug_print(f"[WARN] JSON failed,content: {content[:200]}")
                    continue
                
                # return dictionary, try list field
                if isinstance(data, dict):
                    # prefer list field
                    for key in ["tasks", "toolchain_calls", "plan", "tool_dependencies", "dependencies", "toolchain", "result"]:
                        if key in data and isinstance(data[key], list):
                            _debug_print(f"[INFO] dictionary '{key}' field extract {len(data[key])} task")
                            return data[key]
                    
                    # : model return {"1": {...}, "2": {...}} format
                    # check key
                    if all(k.isdigit() for k in data.keys()) and len(data) > 0:
                        tasks_list = []
                        for k, v in data.items():
                            if isinstance(v, dict):
                                if "id" not in v: v["id"] = int(k)
                                tasks_list.append(v)
                        if tasks_list:
                            _debug_print(f"[INFO] dictionary extract {len(tasks_list)} task")
                            return tasks_list
                    
                    # dictionary task (id), list return
                    if "id" in data or "tool_name" in data or "tool" in data:
                        _debug_print(f"[INFO] task dictionary list")
                        return [data]

                
                # direct list
                if isinstance(data, list):
                    _debug_print(f"[INFO] direct return list format, {len(data)} task")
                    return data
                
                _debug_print(f"[WARN] response extract task list,content: {content[:500]}")
            except Exception as e:
                _debug_print(f"[WARN] analysis dependency failed (retry {attempt+1}): {e}")
                time.sleep(1)
        return []

    def no_tool_answer(
        self,
        task: str,
        max_retries: int = 5,
    ) -> str:
        'use tool directlyly based on task generate answer. use no_tool_answer prompt template, model directly uses user question, use tool. Args: task: user query/task max_retries: maximum retry (default:5) Returns: str: final answer (JSON extract final_answer field)'
        _debug_print(f"[INFO] start use tool task: {task[:100]}...")
        
        # prompt
        prompt = PROMPT.no_tool_answer.format(
            context=PROMPT.context,
            task=task,
        )
        
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Please provide the final answer JSON only."}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("LLM return contentis empty")
                
                # no_tool_answer use <END_OF_PLAN>, extract JSON
                # use response_format={"type": "json_object"}, LLM direct return JSON
                # compatible, supports extract <END_OF_PLAN> label content
                parsed = None
                
                # try extract <END_OF_PLAN> label content
                if "<END_OF_PLAN>" in content:
                    parts = content.split("<END_OF_PLAN>")
                    if len(parts) >= 2:
                        # extractlabel content
                        json_content = parts[1].strip()
                        if json_content:
                            try:
                                parsed = self._safe_parse_json(json_content)
                            except json.JSONDecodeError:
                                pass
                
                # success extract, try direct content
                if parsed is None:
                    try:
                        parsed = self._safe_parse_json(content)
                        if parsed is None:
                            raise json.JSONDecodeError("cannot parse json", content, 0)
                    except json.JSONDecodeError as e:
                        # direct failed, try JSON
                        first_brace = content.find('{')
                        if first_brace != -1:
                            last_brace = content.rfind('}')
                            if last_brace > first_brace:
                                try:
                                    parsed = self._safe_parse_json(content[first_brace:last_brace + 1])
                                except json.JSONDecodeError:
                                    raise ValueError(f"JSON: {e}")
                        else:
                            raise ValueError(f"JSON: {e}")
                
                if isinstance(parsed, dict):
                    final_answer = parsed.get("final_answer", "")
                    if final_answer:
                        _debug_print(f"[SUCCESS] success generate answer(: {len(final_answer)})")
                        return final_answer
                    else:
                        _debug_print(f"[WARN] JSON not found final_answer field")
                        retry_count += 1
                        if retry_count < max_retries:
                            _debug_print(f"[INFO] retry {retry_count}/{max_retries}...")
                            continue
                else:
                    _debug_print(f"[WARN] responseis not JSON")
                    retry_count += 1
                    if retry_count < max_retries:
                        _debug_print(f"[INFO] retry {retry_count}/{max_retries}...")
                        continue
                    
            except Exception as e:
                last_error = str(e)
                _debug_print(f"[WARN] generate answer: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    _debug_print(f"[INFO] retry {retry_count}/{max_retries}...")
                    continue
        
        # retry failed
        builtins.print(f"[ERROR] maximum retry ({max_retries}), generate answer")
        if last_error:
            builtins.print(f"[ERROR] error: {last_error}")
        return ""
    
    def generate_final_answer(
        self,
        task: str,
        dag_results: Dict[str, Any],
        file_contents: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        'DAG result generate final answer.'
        sanitized_results = self._sanitize_for_json(dag_results)
        payload = {
            "files": file_contents or [],
            "dag_results": sanitized_results or {},
        }

        # ===: final prompt payload token budget, max_seq_len ===
        # default:max_seq_len=163840(environment variable override)
        max_context_length = int(os.getenv("ProFinAgent_MAX_CONTEXT_LENGTH", "163840"))
        safety_margin = int(os.getenv("ProFinAgent_SAFETY_MARGIN", "8000"))
        max_usable = max(4096, max_context_length - safety_margin)
        # system prompt/notebook/generate token empty, payload use ~55%
        max_result_tokens = int(max_usable * 0.7)

        # single-file content(file)
        if isinstance(payload.get("files"), list) and payload["files"]:
            per_file_max_tokens = int(max_result_tokens * 0.08)  # single-file 9%
            per_file_max_tokens = max(512, per_file_max_tokens)
            truncated_files = []
            for file_item in payload["files"]:
                if isinstance(file_item, dict) and "content" in file_item and isinstance(file_item.get("content"), str):
                    content = file_item.get("content", "")
                    if content and self._estimate_tokens(content) > per_file_max_tokens:
                        file_item = dict(file_item)
                        file_item["content"] = self._truncate_text(content, per_file_max_tokens, preserve_end=False)
                        file_item["content"] += '... [truncated for token budget]'
                truncated_files.append(file_item)
            payload["files"] = truncated_files

        result_str = json.dumps(payload, ensure_ascii=False, indent=2)

        # : (prefer related file content)
        if self._estimate_tokens(result_str) > max_result_tokens and isinstance(payload.get("files"), list):
            files_list = [f for f in payload["files"] if isinstance(f, dict)]
            # related; relevance_score
            def _score(f: Dict[str, Any]) -> float:
                try:
                    return float(f.get("relevance_score", -1e9))
                except Exception:
                    return -1e9

            files_list.sort(key=_score)  #
            for f in files_list:
                if self._estimate_tokens(result_str) <= max_result_tokens:
                    break
                if isinstance(f.get("content"), str) and f.get("content"):
                    f["content"] = "[omitted due to token budget; use the file path for full access]"
                    result_str = json.dumps(payload, ensure_ascii=False, indent=2)

        # : (keep structure)
        if self._estimate_tokens(result_str) > max_result_tokens:
            result_str = self._truncate_text(result_str, max_result_tokens, preserve_end=False)
            # try JSON } (structure)
            if "}" in result_str:
                result_str = result_str[: result_str.rfind("}") + 1]
        
        # _embedding_function None, use notebook
        if self._embedding_function is None:
            notebook_str = ""
        else:
            # notebook_str = ""
            notebook_entries = self._find_similar_notebook_entries(task)
            notebook_str = json.dumps(notebook_entries, ensure_ascii=False, indent=2) if notebook_entries else ""

        prompt = PROMPT.End_task.format(
            context=PROMPT.context,
            task=task,
            result=result_str,
            notebook=notebook_str
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Please provide the final answer JSON only. You can only answer questions based on your own knowledge and the time frame required for the question. Do not use any information outside the frame of the question time needed or fabricated content."}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            if not content:
                return ""

            # extract, JSONDecode Error
            return self._extract_final_answer_from_response(content)
        except Exception as e:
            builtins.print(f"[ERROR] generate final answer failed: {e}")
            traceback.print_exc()
            return ""

    def _format_dag_results_for_reflection(
        self,
        dag_results: Optional[Dict[str, Any]] = None,
        dag_status: Optional[str] = None,
        dag_error: Optional[str] = None,
    ) -> str:
        'format DAG result, self_reflection prompt. Args: dag_results: DAG result dictionary {task_id: task_result} dag_status: DAG status("success", "error", "no_tools", "no_tasks") dag_error: DAG error () Returns: format DAG result'
        if not dag_results and not dag_status:
            return "No DAG execution (no tools needed or DAG not executed)."
        
        summary = {
            "status": dag_status or "unknown",
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "task_details": []
        }
        
        if dag_results and isinstance(dag_results, dict):
            summary["total_tasks"] = len(dag_results)
            for task_id, task_result in dag_results.items():
                if isinstance(task_result, dict):
                    task_status = task_result.get("status", "unknown")
                    tool_name = task_result.get("tool", "Unknown")
                    
                    if task_status == "success":
                        summary["successful_tasks"] += 1
                        # success tool output_preview
                        summary["task_details"].append({
                            "task_id": task_id,
                            "tool": tool_name,
                            "status": task_status
                        })
                    elif task_status == "error":
                        summary["failed_tasks"] += 1
                        # failed tool output_preview write error (output field get)
                        task_output = task_result.get("output", "")
                        error_message = str(task_output) if task_output else "Unknown error"
                        # error,
                        if len(error_message) > 500:
                            error_message = error_message[:500] + "..."
                        
                        summary["task_details"].append({
                            "task_id": task_id,
                            "tool": tool_name,
                            "status": task_status,
                            "output_preview": error_message
                        })
                    else:
                        # status
                        summary["task_details"].append({
                            "task_id": task_id,
                            "tool": tool_name,
                            "status": task_status
                        })
        
        return json.dumps(summary, ensure_ascii=False, indent=2)
    
    def self_reflect_and_save(
        self,
        task: str,
        toolchain_calls: List[Dict[str, Any]],
        final_answer: str,
        reference_tools: List[str],
        reference_answer: str,
        dag_results: Optional[Dict[str, Any]] = None,
        dag_status: Optional[str] = None,
        dag_error: Optional[str] = None,
    ) -> str:
        'generate reflection save.'
        result_str = json.dumps(toolchain_calls, ensure_ascii=False, indent=2)
        reference_tools_str = ','.join(reference_tools) if reference_tools else ""
        dag_results_str = self._format_dag_results_for_reflection(
            dag_results=dag_results,
            dag_status=dag_status,
            dag_error=dag_error
        )
        
        prompt = PROMPT.self_reflection.format(
            context=PROMPT.context,
            task=task,
            result=result_str,
            dag_results=dag_results_str,
            reference_tools=reference_tools_str,
            final_answer=final_answer,
            reference_answer=reference_answer,
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Please output JSON only."}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            if not content:
                return ""
            
            # directly use format dag_results_str, is not LLM response extract
            # save DAG result, is not LLM generate
            dag_results_in_reflection = dag_results_str
            
            # dag_results_str, JSON file format
            # notebook.json format dag_results, is not
            dag_results_obj = None
            try:
                if dag_results_in_reflection and isinstance(dag_results_in_reflection, str):
                    dag_results_obj = json.loads(dag_results_in_reflection)
                elif isinstance(dag_results_in_reflection, dict):
                    dag_results_obj = dag_results_in_reflection
            except (json.JSONDecodeError, TypeError):
                # failed, ("No DAG execution...")
                dag_results_obj = dag_results_in_reflection
            
            # try extract <PLAN> <END_OF_PLAN> label content(exists)
            if "<PLAN>" in content and "</PLAN>" in content:
                plan_content = content.split("<PLAN>")[1].split("</PLAN>")[0].strip()
                if plan_content:
                    try:
                        data = self._safe_parse_json(plan_content)
                        if isinstance(data, dict):
                            reflection_text = data.get("self_reflection", "")
                            # LLM response extract dag_results, use format original data
                            if reflection_text:
                                entries = self._load_notebook_entries()
                                entries.append({
                                    "task": task,
                                    "self_reflection": reflection_text,
                                    "dag_results": dag_results_obj  # save, JSON auto format
                                })
                                self._save_notebook_entries(entries)
                                return reflection_text
                    except:
                        pass
            
            # label, direct JSON
            data = self._safe_parse_json(content)
            if not isinstance(data, dict):
                # : JSON
                preview = (content or "").strip().replace('', '')[:200]
                builtins.print(f"[WARN] self_reflection return JSON, skip.content preview: {preview}")
                return ""

            reflection_text = data.get("self_reflection", "")
            # LLM response extract dag_results, use format original data
            
            entries = self._load_notebook_entries()
            entries.append({
                "task": task,
                "self_reflection": reflection_text,
                "dag_results": dag_results_obj  # save, JSON auto format
            })
            self._save_notebook_entries(entries)
            return reflection_text
        except Exception as e:
            builtins.print(f"[ERROR] reflection failed: {e}")
            traceback.print_exc()
            return ""

    def collect_existing_file_contents(
        self, 
        dag_results: Dict[str, Any], 
        task: Optional[str] = None,
        max_chars: int = 20000,
        relevance_threshold: float = 0.3,
        use_llamaindex: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k_chunks: int = 5,
    ) -> List[Dict[str, str]]:
        'file content: (4): 1. use embedding related file(based on related threshold) 2. file, use Llama Index RAG chunking extract related fragment 3. file, direct read 4. data file: return "summary/"(time-series prompt max_seq_len), keep path tool/code read Args: dag_results: DAG result task: user query task(related content extract) max_chars: file maximum (default:30000, data file) relevance_threshold: file related threshold(default:0.3), related threshold file use_llamaindex: use Llama Index extract(default: True) chunk_size: size(default:1000) chunk_overlap: (default:200) top_k_chunks: file extract (default:5) Returns: [{"path": "...", "content": "...", "relevance_score": 0.xx, "extraction_method": "..."}]'
        if not dag_results:
            return []

        def _summarize_stock_data_text(
            path: str,
            text: str,
            task_text: Optional[str],
            max_out_chars: int,
        ) -> str:
            'generate data file summary: + field +.: token, tool path read data.'
            ext = os.path.splitext(path)[1].lower()
            size = len(text or "")
            head = (text or "")[:8000]
            tail = (text or "")[-8000:] if size > 8000 else ""

            # field(JSON key pattern, CSV header)
            fields: List[str] = []
            if ext == '.csv':
                first_line = head.splitlines()[0] if head else ""
                if first_line and "," in first_line:
                    fields = [c.strip() for c in first_line.split(",") if c.strip()][:50]
            else:
                # extract key(50)
                seen = set()
                for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_\- ]{0,40})"\s*:', head):
                    k = m.group(1).strip()
                    if k and k not in seen:
                        seen.add(k)
                        fields.append(k)
                    if len(fields) >= 50:
                        break

            # date range()
            dates_head = re.findall(r"\d{4}-\d{2}-\d{2}", head)
            dates_tail = re.findall(r"\d{4}-\d{2}-\d{2}", tail) if tail else []
            start_date = dates_head[0] if dates_head else ""
            end_date = (dates_tail[-1] if dates_tail else (dates_head[-1] if dates_head else ""))

            # field exists
            head_lower = head.lower()
            key_flags = []
            for k in ["open", "high", "low", "close", "volume", "rsi", "macd", "macd_signal", "macd_hist", "atr"]:
                if k in head_lower:
                    key_flags.append(k)

            summary_lines = [
                f"[Stock Data Summary] file={os.path.basename(path)}",
                f"- path: {os.path.abspath(path)}",
                f"- size_chars: {size}",
                f"- ext: {ext or 'unknown'}",
                f"- date_range_guess: {start_date} -> {end_date}",
                f"- detected_fields(sample): {fields[:30]}",
                f"- detected_key_flags: {key_flags}",
            ]
            if task_text:
                summary_lines.append(f"- task_hint: {task_text[:200]}")
            summary_lines.append(
                "- note: full time-series content is intentionally omitted to avoid max_seq_len; use the file path for full access."
            )
            summary_lines.append('[HEAD_SAMPLE]' + (head[:4000] if head else "[empty]"))
            if tail:
                summary_lines.append('[TAIL_SAMPLE]' + tail[-4000:])

            out = ''.join(summary_lines)
            if len(out) > max_out_chars:
                out = out[:max_out_chars] + '... [summary truncated]'
            return out
        
        # extract file path(supports folderexpand same-name file)
        paths = []
        def find_paths_correct(obj, seen=None, parent_key=None):
            if seen is None:
                seen = set()
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, str):
                candidate = os.path.expanduser(obj)
                candidate_abs = os.path.abspath(candidate)
                key = (str(parent_key) if parent_key is not None else "").strip().lower()
                is_path_key = bool(key) and (key == "path" or key.endswith("path") or key.endswith("paths"))
                
                if os.path.isfile(candidate_abs):
                    # file, direct
                    paths.append(candidate_abs)
                elif os.path.isdir(candidate_abs):
                    # folder: field "path field"(output_path/file_path/path) expand
                    # args. root config field output directory expand
                    if is_path_key:
                        _debug_print(f"[INFO] folder path: {candidate_abs}, expand folder content...")
                        folder_files = expand_folder_paths(candidate_abs)
                        paths.extend(folder_files)
                        _debug_print(f"[INFO] folder {candidate_abs} extract {len(folder_files)} file")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    find_paths_correct(v, seen, parent_key=k)
            elif isinstance(obj, list):
                for i in obj:
                    find_paths_correct(i, seen, parent_key=parent_key)
        
        def expand_folder_paths(folder_path: str) -> List[str]:
            'expand folder path, return folder file path.'
            file_paths = []
            try:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.abspath(os.path.join(root, file))
                        file_paths.append(file_path)
            except Exception as e:
                _debug_print(f"[WARN] expand folder {folder_path} failed: {e}")
            return file_paths
        
        def deduplicate_paths_by_name(paths_list: List[str]) -> List[str]:
            'same-name file: same-name file (.htm.json), prefer .json file.'
            if not paths_list:
                return paths_list
            
            # file ()
            name_to_paths: Dict[str, List[str]] = {}
            
            for path in paths_list:
                # get file ()
                base_name = os.path.splitext(os.path.basename(path))[0]
                dir_path = os.path.dirname(path)
                # use complete path directory + file, directory same-name file
                key = os.path.join(dir_path, base_name)
                
                if key not in name_to_paths:
                    name_to_paths[key] = []
                name_to_paths[key].append(path)
            
            # same name, prefer .json file
            result_paths = []
            for key, path_list in name_to_paths.items():
                if len(path_list) == 1:
                    # file, direct
                    result_paths.append(path_list[0])
                else:
                    # same-name file, prefer .json
                    json_paths = [p for p in path_list if p.lower().endswith('.json')]
                    if json_paths:
                        # .json file, keep.json(.json, keep)
                        result_paths.append(json_paths[0])
                        _debug_print(f"[INFO] same-name file, prefer .json: {json_paths[0]}")
                        _debug_print(f"[INFO] file: {[p for p in path_list if p != json_paths[0]]}")
                    else:
                        # .json file, keep (prefer, keep)
                        result_paths.append(path_list[0])
                        if len(path_list) > 1:
                            _debug_print(f"[WARN] same-name file.json, keep: {path_list[0]}")
                            _debug_print(f"[WARN] file: {path_list[1:]}")
            
            return result_paths
        
        find_paths_correct(dag_results)
        # same-name file: same name.htm.json, keep.json
        paths = deduplicate_paths_by_name(paths)
        unique_paths = list(set(paths))
        
        if not unique_paths:
            return []
        
        # task embedding unavailable, use original (read file)
        if not task or self._embedding_function is None:
            _debug_print(f"[INFO] task embedding unavailable, use original (read file)")
            contents: List[Dict[str, str]] = []
            for path in unique_paths:
                try:
                    # check data file
                    is_stock_data = self._is_stock_data_file(path)
                    
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        data = f.read()
                    
                    # data file: return summary(prompt)
                    if is_stock_data:
                        stock_summary = _summarize_stock_data_text(
                            path=path, text=data, task_text=task, max_out_chars=min(max_chars, 12000)
                        )
                        contents.append({
                            "path": path,
                            "content": stock_summary,
                            "extraction_method": "stock_data_summary",
                            "is_stock_data": True
                        })
                    else:
                        # file: application max_chars
                        if len(data) > max_chars:
                            data = data[:max_chars] + f"\n... [truncated, total {len(data)} chars]"
                        contents.append({
                            "path": path,
                            "content": data,
                            "extraction_method": "simple_truncate",
                            "is_stock_data": False
                        })
                except Exception as e:
                    _debug_print(f"[WARN] read file {path} failed: {e}")
            return contents
        
        # === 1: use embedding file related ===
        _debug_print(f"[INFO] use embedding file, query: {task[:100]}...")
        _debug_print(f"[INFO] candidate file: {len(unique_paths)}")
        
        # file path generate
        file_descriptions = []
        for path in unique_paths:
            filename = os.path.basename(path)
            path_parts = path.split(os.sep)
            relevant_parts = [p for p in path_parts[-3:] if p]
            description = f"{filename} {' '.join(relevant_parts)}"
            file_descriptions.append((path, description))
        
        # calculate query
        query_embedding = self._embed_query_cached(task)
        
        # calculate file query related (similarity + similar cache)
        file_scores = []
        descriptions_only = [desc for _, desc in file_descriptions]
        doc_embeddings = self._embed_documents_cached(descriptions_only)

        sims: List[Optional[float]] = [None] * len(descriptions_only)
        miss_vecs: List[List[float]] = []
        miss_pos: List[int] = []
        for i, desc in enumerate(descriptions_only):
            k = self._cache_key_pair(task, desc)
            cached = self._lru_get(self._similarity_cache, k)
            if cached is not None:
                sims[i] = float(cached)
            else:
                miss_pos.append(i)
                miss_vecs.append(doc_embeddings[i])

        if miss_vecs:
            miss_scores = self._compute_similarity_batch(query_embedding, miss_vecs)
            for j, score in enumerate(miss_scores):
                idx = miss_pos[j]
                sims[idx] = float(score)
                k = self._cache_key_pair(task, descriptions_only[idx])
                self._lru_set(self._similarity_cache, k, float(score), self._similarity_cache_max)

        for i, (path, description) in enumerate(file_descriptions):
            similarity = float(sims[i] if sims[i] is not None else 0.0)
            file_scores.append((path, similarity, description))
        
        # related sort
        file_scores.sort(key=lambda x: x[1], reverse=True)
        
        # : keeprelated threshold file(count)
        selected_files = [
            (path, score, desc) 
            for path, score, desc in file_scores 
            if score >= relevance_threshold
        ]
        
        _debug_print(f"[INFO] {len(selected_files)} related file(threshold: {relevance_threshold}, count)")
        
        # statistics by file type
        stock_data_count = sum(1 for path, _, _ in selected_files if self._is_stock_data_file(path))
        other_count = len(selected_files) - stock_data_count
        _debug_print(f"[INFO] - data file: {stock_data_count}")
        _debug_print(f"[INFO] - file: {other_count}")
        
        for path, score, desc in selected_files:
            file_type = "text data" if self._is_stock_data_file(path) else "normal file"
            _debug_print(f"[INFO] - {os.path.basename(path)} ({file_type}, related: {score:.3f})")
        
        # === 2: content extract ===
        contents: List[Dict[str, str]] = []

        # file parallel: parallel I/O; embedding use GPU/model
        workers = int(os.getenv("ProFinAgent_COLLECT_WORKERS", "4"))
        workers = max(1, workers)
        embed_conc = int(os.getenv("ProFinAgent_COLLECT_EMBED_CONCURRENCY", "1"))
        embed_conc = max(1, embed_conc)
        embed_sem = threading.Semaphore(embed_conc)

        total_files = len(selected_files)

        def _process_one(idx: int, path: str, relevance_score: float, desc: str) -> tuple[int, Optional[Dict[str, Any]]]:
            t0 = time.time()
            try:
                _debug_print(f"[INFO] [collect] ({idx+1}/{total_files}) start: {os.path.basename(path)}")
                is_stock_data = self._is_stock_data_file(path)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()
                file_size = len(full_text)

                if is_stock_data:
                    stock_summary = _summarize_stock_data_text(
                        path=path, text=full_text, task_text=task, max_out_chars=min(max_chars, 12000)
                    )
                    item = {
                        "path": path,
                        "content": stock_summary,
                        "relevance_score": float(relevance_score),
                        "extraction_method": "stock_data_summary",
                        "original_size": file_size,
                        "is_stock_data": True,
                    }
                    _debug_print(f"[INFO] [collect] ({idx+1}/{total_files}) done(stock_summary) size={file_size} cost={time.time()-t0:.2f}s")
                    return idx, item

                if file_size <= max_chars:
                    item = {
                        "path": path,
                        "content": full_text,
                        "relevance_score": float(relevance_score),
                        "extraction_method": "full",
                        "original_size": file_size,
                        "is_stock_data": False,
                    }
                    _debug_print(f"[INFO] [collect] ({idx+1}/{total_files}) done(full) size={file_size} cost={time.time()-t0:.2f}s")
                    return idx, item

                with embed_sem:
                    _debug_print(f"[INFO] [collect] ({idx+1}/{total_files}) large_file extract begin size={file_size}")
                    if use_llamaindex:
                        try:
                            extracted_content = self._extract_relevant_content_with_llamaindex(
                                full_text, task, chunk_size, chunk_overlap, top_k_chunks
                            )
                            method = "llamaindex_rag"
                        except ImportError:
                            extracted_content = self._extract_relevant_chunks_simple(
                                full_text, task, chunk_size, top_k_chunks
                            )
                            method = "embedding_chunking"
                    else:
                        if self._embedding_function:
                            extracted_content = self._extract_relevant_chunks_simple(
                                full_text, task, chunk_size, top_k_chunks
                            )
                            method = "embedding_chunking"
                        else:
                            extracted_content = full_text[:max_chars] + f"\n... [truncated, total {file_size} chars]"
                            method = "simple_truncate"

                item = {
                    "path": path,
                    "content": extracted_content,
                    "relevance_score": float(relevance_score),
                    "extraction_method": method,
                    "original_size": file_size,
                    "is_stock_data": False,
                }
                _debug_print(f"[INFO] [collect] ({idx+1}/{total_files}) done({method}) size={file_size} cost={time.time()-t0:.2f}s")
                return idx, item
            except Exception as e:
                _debug_print(f"[WARN] [collect] ({idx+1}/{total_files}) failed: {path} err={e}")
                return idx, None

        max_workers = min(workers, max(1, total_files))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for i, (path, relevance_score, desc) in enumerate(selected_files):
                futs.append(ex.submit(_process_one, i, path, float(relevance_score), desc))
            tmp: Dict[int, Dict[str, Any]] = {}
            for fut in as_completed(futs):
                idx, item = fut.result()
                if item is not None:
                    tmp[idx] = item

        for i in range(total_files):
            if i in tmp:
                contents.append(tmp[i])

        return contents
    
    def _extract_relevant_content_with_llamaindex(
        self,
        text: str,
        query: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
    ) -> str:
        'use Llama Index extract query related fragment.'
        try:
            from llama_index.core import Document
            from llama_index.core.node_parser import SimpleNodeParser
            
            document = Document(text=text)
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            nodes = node_parser.get_nodes_from_documents([document])
            
            if not nodes:
                return text[:chunk_size * top_k] + f"\n... [truncated, total {len(text)} chars]"
            
            if self._embedding_function:
                query_embedding = self._embed_query_cached(query)
                node_texts_all: List[str] = [
                    (n.get_content() if hasattr(n, "get_content") else str(n)) for n in nodes
                ]

                rough_m = int(os.getenv("ProFinAgent_CHUNK_ROUGH_TOP_M", "64"))
                rough_m = max(top_k, max(8, rough_m))
                cand_idx = self._rough_select_indices(node_texts_all, query, rough_m)
                cand_nodes = [nodes[i] for i in cand_idx]
                cand_texts = [node_texts_all[i] for i in cand_idx]
                _debug_print(f"[INFO] node rough filter: total={len(nodes)} -> candidates={len(cand_nodes)} (top_m={rough_m})")

                node_embeddings = self._embed_documents_cached(cand_texts)

                sims: List[Optional[float]] = [None] * len(cand_texts)
                miss_vecs: List[List[float]] = []
                miss_pos: List[int] = []
                for i, t in enumerate(cand_texts):
                    k = self._cache_key_pair(query, t)
                    cached = self._lru_get(self._similarity_cache, k)
                    if cached is not None:
                        sims[i] = float(cached)
                    else:
                        miss_pos.append(i)
                        miss_vecs.append(node_embeddings[i])

                if miss_vecs:
                    miss_scores = self._compute_similarity_batch(query_embedding, miss_vecs)
                    for j, score in enumerate(miss_scores):
                        idx = miss_pos[j]
                        sims[idx] = float(score)
                        k = self._cache_key_pair(query, cand_texts[idx])
                        self._lru_set(self._similarity_cache, k, float(score), self._similarity_cache_max)

                node_scores = []
                for i, node in enumerate(cand_nodes):
                    similarity = float(sims[i] if sims[i] is not None else 0.0)
                    node_scores.append((node, similarity))
                
                node_scores.sort(key=lambda x: x[1], reverse=True)
                selected_nodes = node_scores[:top_k]
                selected_nodes_sorted = sorted(selected_nodes, key=lambda x: nodes.index(x[0]))
                
                extracted_texts = []
                for node, score in selected_nodes_sorted:
                    content = node.get_content() if hasattr(node, 'get_content') else str(node)
                    extracted_texts.append(content)
                
                result = '... [related fragment]...'.join(extracted_texts)
                return f"[extract {len(selected_nodes)} related fragment, {len(result)} ]\n\n{result}"
            else:
                selected_nodes = nodes[:top_k]
                extracted_texts = []
                for node in selected_nodes:
                    content = node.get_content() if hasattr(node, 'get_content') else str(node)
                    extracted_texts.append(content)
                return '... [related fragment]...'.join(extracted_texts)
                
        except Exception as e:
            _debug_print(f"[WARN] Llama Index extract failed: {e},")
            return self._extract_relevant_chunks_simple(text, query, chunk_size, top_k)
    
    def _extract_relevant_chunks_simple(
        self,
        text: str,
        query: str,
        chunk_size: int = 1000,
        top_k: int = 5,
    ) -> str:
        'based on embedding related content extract.'
        if not self._embedding_function:
            return text[:chunk_size * top_k] + f"\n... [truncated, total {len(text)} chars]"
        
        #
        chunks = self._split_text_into_chunks(text, chunk_size, chunk_size // 5)
        
        if not chunks:
            return text[:chunk_size * top_k] + f"\n... [truncated, total {len(text)} chars]"
        
        # Two-stage: chunk, embedding count
        rough_m = int(os.getenv("ProFinAgent_CHUNK_ROUGH_TOP_M", "64"))
        rough_m = max(top_k, max(8, rough_m))
        chunk_texts_all = [chunk["text"] for chunk in chunks]
        cand_idx = self._rough_select_indices(chunk_texts_all, query, rough_m)
        cand_chunks = [chunks[i] for i in cand_idx]
        cand_texts = [chunk_texts_all[i] for i in cand_idx]
        _debug_print(f"[INFO] chunk rough filter: total={len(chunks)} -> candidates={len(cand_chunks)} (top_m={rough_m})")

        # calculate candidate query related
        query_embedding = self._embed_query_cached(query)
        chunk_embeddings = self._embed_documents_cached(cand_texts)

        sims: List[Optional[float]] = [None] * len(cand_texts)
        miss_vecs: List[List[float]] = []
        miss_pos: List[int] = []
        for i, t in enumerate(cand_texts):
            k = self._cache_key_pair(query, t)
            cached = self._lru_get(self._similarity_cache, k)
            if cached is not None:
                sims[i] = float(cached)
            else:
                miss_pos.append(i)
                miss_vecs.append(chunk_embeddings[i])

        if miss_vecs:
            miss_scores = self._compute_similarity_batch(query_embedding, miss_vecs)
            for j, score in enumerate(miss_scores):
                idx = miss_pos[j]
                sims[idx] = float(score)
                k = self._cache_key_pair(query, cand_texts[idx])
                self._lru_set(self._similarity_cache, k, float(score), self._similarity_cache_max)

        chunk_scores = []
        for i, chunk in enumerate(cand_chunks):
            similarity = float(sims[i] if sims[i] is not None else 0.0)
            chunk_scores.append((chunk, similarity))
        
        # top-k related
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        selected_chunks = chunk_scores[:top_k]
        
        # (original)
        selected_chunks_sorted = sorted(selected_chunks, key=lambda x: x[0]["start"])
        extracted_content = '... [related fragment]...'.join([
            chunk["text"] for chunk, _ in selected_chunks_sorted
        ])
        
        return f"[extract {len(selected_chunks)} related fragment, {len(extracted_content)} ]\n\n{extracted_content}"
    
    def _is_stock_data_file(self, path: str) -> bool:
        'file data technical indicator data file. check file column name: - CSV file: check data (Open, High, Low, Close, Volume) - JSON file: check data field Args: path: file path Returns: True data file, False'
        if not os.path.exists(path):
            return False
        
        file_ext = os.path.splitext(path)[1].lower()
        
        # data technical indicator column name
        stock_data_columns = {
            'open', 'high', 'low', 'close', 'volume', 
            'dividends', 'stock splits', 'stock_splits',
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr'
        }
        
        try:
            if file_ext == '.csv':
                # read CSV file (column name)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        return False
                    
                    # column name(supports)
                    columns = [col.strip().lower() for col in first_line.split(',')]
                    
                    # check data (3)
                    found_columns = sum(1 for col in columns if col in stock_data_columns)
                    if found_columns >= 3:
                        _debug_print(f"[INFO] data file: {os.path.basename(path)} ({found_columns} data)")
                        return True
            
            elif file_ext == '.json':
                # read JSON file, check data field
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    # read 1000
                    preview = f.read(1000)
                    
                    # check data column name
                    preview_lower = preview.lower()
                    found_columns = sum(1 for col in stock_data_columns if col in preview_lower)
                    if found_columns >= 3:
                        _debug_print(f"[INFO] data JSON file: {os.path.basename(path)} ({found_columns} data field)")
                        return True
            
        except Exception as e:
            _debug_print(f"[WARN] file {path} type: {e}")
            return False
        
        return False
    
    def _split_text_into_chunks(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[Dict[str, Any]]:
        """text"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": end
            })
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    def _sanitize_for_json(self, obj: Any, seen: Optional[set] = None) -> Any:
        if seen is None: seen = set()
        obj_id = id(obj)
        if obj_id in seen: return "[Circular Reference]"
        seen.add(obj_id)
        if isinstance(obj, (str, int, float, bool, type(None))):
            seen.remove(obj_id)
            return obj
        if isinstance(obj, dict):
            res = {str(k): self._sanitize_for_json(v, seen) for k, v in obj.items()}
            seen.remove(obj_id)
            return res
        if isinstance(obj, (list, tuple)):
            res = [self._sanitize_for_json(i, seen) for i in obj]
            seen.remove(obj_id)
            return res
        try:
            if hasattr(obj, 'text'): return self._sanitize_for_json(getattr(obj, 'text'), seen)
            if hasattr(obj, '__dict__'):
                d = {k: self._sanitize_for_json(v, seen) for k, v in obj.__dict__.items() if not k.startswith('_')}
                seen.remove(obj_id)
                return d
            s = str(obj)
            seen.remove(obj_id)
            return s
        except:
            seen.remove(obj_id)
            return f"[{type(obj).__name__}]"

    def _extract_first_json(self, text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        if not text: return None
        fb = text.find("{")
        fbr = text.find("[")
        if fb == -1 and fbr == -1: return None
        start = fb if (fb != -1 and (fbr == -1 or fb < fbr)) else fbr
        end_char = "}" if start == fb else "]"
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try: return json.loads(text[start:end+1])
            except: pass
        return None

    def _strip_code_fences(self, text: str) -> str:
        'common Markdown code (```json... ```/```... ```).'
        if not text or not isinstance(text, str):
            return ""
        s = text.strip()
        if s.startswith("```"):
            first_newline = s.find('')
            if first_newline != -1:
                header = s[:first_newline].strip().lower()
                if header in ("```", "```json", "```javascript"):
                    s = s[first_newline + 1 :]
            s = s.strip()
            if s.endswith("```"):
                s = s[:-3]
        return s.strip()

    def _safe_parse_json(self, text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        'JSON: - prefer json. loads - failedinstall json_repair, repair_json (LLM JSON) - failed: extract JSON/{}/[] failed return None(call retry).'
        if not text or not isinstance(text, str):
            return None
        s = self._strip_code_fences(text)

        # 1) direct
        try:
            return json.loads(s)
        except Exception:
            pass

        # 1.5) use json_repair (available)
        if _JSON_REPAIR_AVAILABLE and repair_json is not None:
            try:
                repaired = repair_json(s)
                if isinstance(repaired, str) and repaired.strip():
                    return json.loads(repaired)
            except Exception:
                pass

        # 2) extract JSON
        try:
            extracted = self._extract_first_json(s)
            if extracted is not None:
                return extracted
        except Exception:
            pass

        # 3)/
        fb = s.find("{")
        lb = s.rfind("}")
        if fb != -1 and lb != -1 and lb > fb:
            try:
                return json.loads(s[fb : lb + 1])
            except Exception:
                pass
        fbr = s.find("[")
        lbr = s.rfind("]")
        if fbr != -1 and lbr != -1 and lbr > fbr:
            try:
                return json.loads(s[fbr : lbr + 1])
            except Exception:
                pass
        return None

    def _extract_final_answer_from_response(self, text: str) -> str:
        'final answer extract: -: JSON return final_answer - exception: final_answer; return clean ()'
        if not text or not isinstance(text, str):
            return ""
        s = text.strip()

        # compatible End_task/no_tool_answer <END_OF_PLAN>
        if "<END_OF_PLAN>" in s:
            parts = s.split("<END_OF_PLAN>")
            if len(parts) >= 2:
                s = parts[1].strip()

        s = self._strip_code_fences(s)

        parsed = self._safe_parse_json(s)
        if isinstance(parsed, dict):
            fa = parsed.get("final_answer", "")
            return fa if isinstance(fa, str) else str(fa)

        # : final_answer content(compatible `final_answer:...`)
        lower = s.lower()
        idx = lower.find("final_answer")
        if idx != -1:
            tail = s[idx:]
            colon = tail.find(":")
            if colon != -1:
                cand = tail[colon + 1 :].strip()
                if cand.startswith(("\"", "'")):
                    cand = cand[1:]
                cand = cand.strip()
                cand = cand.rstrip("}").rstrip()
                cand = cand.rstrip("\"'").rstrip()
                if cand:
                    return cand

        return s

__all__ = ["TestAgent"]
