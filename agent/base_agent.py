import os
import re
import sys
import json
import numpy as np
import traceback
import builtins
import threading
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from json_repair import repair_json  # type: ignore
    _JSON_REPAIR_AVAILABLE = True
except Exception:
    repair_json = None  # type: ignore
    _JSON_REPAIR_AVAILABLE = False

# model path sys. path
_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pretrain/model'))
if _model_path not in sys.path:
    sys.path.insert(0, _model_path)

# prompt path sys. path
_prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../prompt'))
if _prompt_path not in sys.path:
    sys.path.insert(0, _prompt_path)

# rag path sys. path
_rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag'))
if _rag_path not in sys.path:
    sys.path.insert(0, _rag_path)

import agent_config as agent_cfg  # RAG related config
from agent.rag import CustomEmbeddingFunction  # use file embedding

try:
    from sglang_LLM import SGLangClient  # type: ignore
    import prompt.base_agent as prompt_templates  # type: ignore
except ImportError as e:
    # direct, exception, call
    raise


# basiclog (default, output)
_DEBUG_LOG = True


def _debug_print(*args, **kwargs) -> None:
    ', _DEBUG_LOG output.'
    if _DEBUG_LOG:
        builtins.print(*args, **kwargs)

_RAG_AVAILABLE = True

# try tiktoken token
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False
    _debug_print(f"[WARN] tiktoken not installed, use token. install: pip install tiktoken")

# try Llama Index
try:
    from llama_index.core import Document, Settings
    from llama_index.core.node_parser import SimpleNodeParser
    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    try:
        # try
        from llama_index import Document, ServiceContext
        _LLAMAINDEX_AVAILABLE = True
        _LLAMAINDEX_LEGACY = True
    except ImportError:
        _LLAMAINDEX_AVAILABLE = False
        _LLAMAINDEX_LEGACY = False
        _debug_print(f"[WARN] Llama Index not installed, use. install: pip install llama-index")

# model maximum (Qwen3-4B-Instruct-2507 131072)
MAX_CONTEXT_LENGTH = 131072
# keep (, token, format)
# , token, actual
SAFETY_MARGIN = 5000  # 1000 5000,
# actually available
MAX_USABLE_LENGTH = MAX_CONTEXT_LENGTH - SAFETY_MARGIN


class BaseAgent:
    # global: save dag_memory.json(default True, use)
    SAVE_TO_DAG_MEMORY = False
    'basic Agent (use SGLang model)'
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        model_name: str = './ProFinAgent/pretrain/Qwen3-4B-Instruct-2507',
        model_path: str = './ProFinAgent/pretrain/Qwen3-4B-Instruct-2507',
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 6144,
        timeout: int = 600,
        auto_load: bool = True,
        system_prompt: Optional[str] = None,
        embedding_model_path: Optional[str] = agent_cfg.EMBEDDING_MODEL_PATH,
        chroma_db_path: Optional[str] = agent_cfg.CHROMA_DB_PATH,
        chroma_collection_name: str = agent_cfg.CHROMA_COLLECTION_NAME,
        matryoshka_dim: Optional[int] = agent_cfg.MATRYOSHKA_DIM,
        tools_json_path: Optional[str] = None,
        notebook_path: Optional[str] = None,
    ):
        _debug_print(f"[INFO] initialize Base Agent...")
        _debug_print(f"[INFO] model path: {model_path}")
        _debug_print(f"[INFO] model name: {model_name}")
        _debug_print(f"[INFO] service: {base_url}")
        
        # initialize SGLang
        self.client = SGLangClient(
            base_url=base_url,
            model_name=model_name,
            model_path=model_path,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            auto_load=auto_load,
        )
        
        # save config
        self.model_name = model_name
        self.model_path = model_path
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.full_task = None  # complete task, format dag_memory.json
        self.notebook_path = notebook_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../notebook/notebook.json')
        )
        
        # load
        if system_prompt is None:
            self.system_prompt = prompt_templates.context
        else:
            self.system_prompt = system_prompt
        
        # RAG related config(initialize None)
        self._chroma_client = None
        self._chroma_collection = None
        self._embedding_function = None

        # must-have/quota config: supports configs/must_have_rules.json update
        # cachestructure:{"mtime": float, "raw": dict}
        self._tool_selection_rules_cache: Dict[str, Any] = {}

        # === Embedding/similarity LRU cache(calculate similarity call) ===
        # description:
        # - embedding_cache: cache embed_query/embed_documents ()
        # - similarity_cache: cache (query_text, doc_text) similar ()
        # environment variable; default,
        self._embedding_cache: 'OrderedDict[tuple, List[float]]' = OrderedDict()
        self._similarity_cache: 'OrderedDict[tuple, float]' = OrderedDict()
        self._embedding_cache_max = int(os.getenv("ProFinAgent_EMBED_CACHE_MAX", "4096"))
        self._similarity_cache_max = int(os.getenv("ProFinAgent_SIM_CACHE_MAX", "16384"))
        self._cache_lock = threading.Lock()
        self.tool_embedding_threshold = float(
            os.getenv(
                "ProFinAgent_TOOL_EMBED_THRESHOLD",
                getattr(agent_cfg, "TEST_AGENT_TOOL_EMBEDDING_THRESHOLD", 0.2),
            )
        )
        
        # load tool type (tool_type -> description), tool_type.json load
        tool_type_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../configs/tool_type.json')
        )
        
        self.tool_types: Dict[str, str] = {}
        try:
            if os.path.exists(tool_type_path):
                _debug_print(f"[INFO] load tool type: {tool_type_path}")
                with open(tool_type_path, 'r', encoding='utf-8') as f:
                    tool_type_config = json.load(f)
                
                # tool_type.json format, tool_type, description, tools
                if isinstance(tool_type_config, list):
                    for category in tool_type_config:
                        tool_type = category.get('tool_type', '')
                        description = category.get('description', '')
                        if tool_type:
                            self.tool_types[tool_type] = description
                            _debug_print(f"[INFO] tool type: {tool_type} -> {description[:50]}...")
                
                _debug_print(f"[SUCCESS] load {len(self.tool_types)} tool type")
            else:
                _debug_print(f"[WARN] tool_type.json file does not exist: {tool_type_path}")
                _debug_print(f"[WARN] tool_types is emptydictionary, related unavailable")
        except Exception as e:
            _debug_print(f"[WARN] load tool type failed: {e}")
            _debug_print(f"[WARN] tool_types is emptydictionary, related unavailable")
            traceback.print_exc()
        
        _debug_print(f"[SUCCESS] Base Agent initialize complete")
        _debug_print(f"[INFO]: {self.system_prompt[:100]}...")

        # RAG config parameter, initialize RAG
        if embedding_model_path is not None and chroma_db_path is not None:
            _debug_print(f"[INFO] RAG config parameter, startinitialize RAG...")
            try:
                _debug_print(f"[INFO] initialize RAG data...")
                _debug_print(f"[INFO] Embedding model path: {embedding_model_path}")
                _debug_print(f"[INFO] data path: {chroma_db_path}")
                
                # path check
                if not os.path.exists(embedding_model_path):
                    raise FileNotFoundError(f"Embedding model path does not exist: {embedding_model_path}")

                # === [ 1] Embedding ===
                # Base Agent " "
                self._embedding_function = CustomEmbeddingFunction(
                    model_path=embedding_model_path,
                    matryoshka_dim=matryoshka_dim
                )
            except Exception as e:
                _debug_print(f"[WARN] RAG initialize failed: {e}")
                _debug_print(f"[INFO] RAG unavailable,")
                traceback.print_exc()
        else:
            _debug_print('[INFO] RAG config, skip initialize')

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
            # cache failed
            return

    def _cache_key_pair(self, query_text: str, doc_text: str) -> tuple:
        # use Python hash(),
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
            # mini-batch: time response/
            bs = int(os.getenv("ProFinAgent_EMBED_BATCH_SIZE", "32"))
            bs = max(1, bs)
            total = len(miss_texts)
            for start in range(0, total, bs):
                end = min(start + bs, total)
                batch = miss_texts[start:end]
                # log(debug)
                _debug_print(f"[INFO] embed_documents batch {start//bs + 1}/{(total + bs - 1)//bs} (size={len(batch)})")
                vecs = self._embedding_function.embed_documents(batch)
                for j, vec in enumerate(vecs):
                    idx = miss_idx[start + j]
                    t = miss_texts[start + j]
                    results[idx] = vec
                    self._lru_set(self._embedding_cache, ("d", t), vec, self._embedding_cache_max)
        # type ignore: results
        return [r for r in results if r is not None]  # type: ignore

    # === Two-stage rough filter (embedding chunk/node count) ===
    def _extract_query_keywords(self, query: str, max_keywords: int = 24) -> List[str]:
        q = (query or "").lower()
        if not q:
            return []
        # English/token + Chinese fragment
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
        if not text:
            return 0.0
        if not keywords:
            return 0.0
        tl = text.lower()
        score = 0.0
        # :; count (O(N*K))
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
            # :, overwrite
            step = max(1, n // max_candidates)
            idxs = list(range(0, n, step))[:max_candidates]
            if 0 not in idxs:
                idxs[0] = 0
            if (n - 1) not in idxs and len(idxs) < max_candidates:
                idxs.append(n - 1)
            return sorted(set(idxs))
        scored = [(i, self._rough_score_text(texts[i], kws)) for i in range(n)]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [i for i, _ in scored[:max_candidates]]
        # overwrite ()
        top.append(0)
        top.append(n - 1)
        return sorted(set(top))

    # === Batch similarity (: N model. similarity 1) ===
    def _compute_similarity_batch(self, query_vec: List[float], doc_vecs: List[List[float]]) -> List[float]:
        if not doc_vecs:
            return []
        if self._is_qwen3_model():
            try:
                model = self._embedding_function.model  # type: ignore[attr-defined]
                q = np.array([query_vec], dtype=np.float32)  # [1, dim]
                d = np.array(doc_vecs, dtype=np.float32)     # [N, dim]
                sims = model.similarity(q, d)
                sims_np = np.array(sims, dtype=np.float32).reshape(-1)
                # log,
                _debug_print(f"[INFO] Qwen3 batch similarity computed for {len(doc_vecs)} docs")
                return [float(x) for x in sims_np.tolist()]
            except Exception as e:
                _debug_print(f"[WARN] Qwen3 batch similarity failed,: {e}")
                # fallthrough
        # Qwen3:
        qv = np.array(query_vec, dtype=np.float32)          # [dim]
        dv = np.array(doc_vecs, dtype=np.float32)           # [N, dim]
        q_norm = float(np.linalg.norm(qv) + 1e-12)
        d_norm = np.linalg.norm(dv, axis=1) + 1e-12
        sims = (dv @ qv) / (d_norm * q_norm)
        return [float(x) for x in sims.tolist()]
    
    def _estimate_tokens(self, text: str) -> int:
        'token count: Qwen model use cl100k_base tokenizer,., basic 20%. Args: input Returns: token count()'
        if not text:
            return 0
        
        base_estimate = 0
        
        # prefer using tiktoken()
        if _TIKTOKEN_AVAILABLE and tiktoken is not None:
            try:
                # use cl100k_base (GPT-4 model use)
                # : Qwen model use tokenizer,
                encoding = tiktoken.get_encoding("cl100k_base")
                base_estimate = len(encoding.encode(text))
            except Exception as e:
                _debug_print(f"[WARN] tiktoken failed,: {e}")
                base_estimate = 0
        
        # tiktoken failed, use
        if base_estimate == 0:
            # : (Chinese 1.5/token, English 4/token)
            # use: average 2.5/token
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            total_chars = len(text)
            # Chinese 1.5/token, 4/token
            base_estimate = int(chinese_chars / 1.5 + (total_chars - chinese_chars) / 4)
            base_estimate = max(base_estimate, total_chars // 4)  # 4/token
        
        # , basic 20%
        # Qwen model tokenizer cl100k_base, actual token
        safe_estimate = int(base_estimate * 1.2)
        
        return safe_estimate
    
    def _truncate_text(self, text: str, max_tokens: int, preserve_end: bool = False) -> str:
        'maximum token Args: input max_tokens: maximum token preserve_end: keep (True keep, False keep) Returns:'
        if not text:
            return text
        
        current_tokens = self._estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        #
        _debug_print(f"[WARN] ({current_tokens} tokens), {max_tokens} tokens")
        
        if _TIKTOKEN_AVAILABLE and tiktoken is not None:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(text)
                
                if preserve_end:
                    # keep
                    truncated_tokens = tokens[-max_tokens:]
                    return encoding.decode(truncated_tokens)
                else:
                    # keep
                    truncated_tokens = tokens[:max_tokens]
                    return encoding.decode(truncated_tokens)
            except Exception as e:
                _debug_print(f"[WARN] tiktoken failed,: {e}")
        
        # :
        # (average 2.5/token)
        max_chars = int(max_tokens * 2.5)
        if preserve_end:
            return text[-max_chars:]
        else:
            return text[:max_chars]
    
    def _truncate_json_string(self, json_str: str, max_tokens: int) -> Optional[str]:
        'JSON, JSON format. Args: json_str: JSON max_tokens: maximum token Returns: JSON, failed return None'
        if not json_str:
            return None
        
        current_tokens = self._estimate_tokens(json_str)
        if current_tokens <= max_tokens:
            return json_str
        
        try:
            # try JSON, success, field
            data = json.loads(json_str)
            
            # dictionary, try field
            if isinstance(data, dict):
                truncated_data = {}
                remaining_tokens = max_tokens
                
                # field token, prefer keeping field
                for key, value in data.items():
                    if isinstance(value, str):
                        value_tokens = self._estimate_tokens(value)
                        if value_tokens > remaining_tokens * 0.5:  # field empty 50%,
                            truncated_value = self._truncate_text(value, int(remaining_tokens * 0.4), preserve_end=False)
                            truncated_data[key] = truncated_value
                            remaining_tokens -= self._estimate_tokens(truncated_value)
                        else:
                            truncated_data[key] = value
                            remaining_tokens -= value_tokens
                    elif isinstance(value, (list, dict)):
                        # structure,
                        value_str = json.dumps(value, ensure_ascii=False, indent=2)
                        value_tokens = self._estimate_tokens(value_str)
                        if value_tokens > remaining_tokens * 0.5:
                            # structure(convert)
                            truncated_value_str = self._truncate_text(value_str, int(remaining_tokens * 0.4), preserve_end=False)
                            try:
                                truncated_data[key] = json.loads(truncated_value_str)
                            except:
                                truncated_data[key] = value  # failed, keep
                        else:
                            truncated_data[key] = value
                            remaining_tokens -= value_tokens
                    else:
                        truncated_data[key] = value
                
                # generate JSON
                result = json.dumps(truncated_data, ensure_ascii=False, indent=2)
                final_tokens = self._estimate_tokens(result)
                
                # , use
                if final_tokens > max_tokens:
                    result = self._truncate_text(result, max_tokens, preserve_end=False)
                    # try JSON format
                    if not result.rstrip().endswith('}'):
                        last_brace = result.rfind('}')
                        if last_brace > 0:
                            result = result[:last_brace + 1]
                
                return result
            elif isinstance(data, list):
                # list, list
                truncated_list = []
                remaining_tokens = max_tokens
                for item in data:
                    item_str = json.dumps(item, ensure_ascii=False, indent=2)
                    item_tokens = self._estimate_tokens(item_str)
                    if item_tokens <= remaining_tokens:
                        truncated_list.append(item)
                        remaining_tokens -= item_tokens
                    else:
                        # empty,
                        truncated_item_str = self._truncate_text(item_str, int(remaining_tokens * 0.9), preserve_end=False)
                        try:
                            truncated_item = json.loads(truncated_item_str)
                            truncated_list.append(truncated_item)
                            remaining_tokens -= self._estimate_tokens(truncated_item_str)
                        except:
                            break  # failed,
                
                result = json.dumps(truncated_list, ensure_ascii=False, indent=2)
                final_tokens = self._estimate_tokens(result)
                if final_tokens > max_tokens:
                    result = self._truncate_text(result, max_tokens, preserve_end=False)
                    if not result.rstrip().endswith(']'):
                        last_bracket = result.rfind(']')
                        if last_bracket > 0:
                            result = result[:last_bracket + 1]
                
                return result
        except json.JSONDecodeError:
            # JSON failed, use
            pass
        
        # :
        return None
    
    def _check_and_truncate_prompt(
        self,
        prompt: str,
        system_prompt: str,
        max_total_tokens: int = MAX_USABLE_LENGTH
    ) -> tuple[str, str]:
        'check prompt system_prompt, token use, actual token. Args: prompt: user prompt system_prompt: prompt max_total_tokens: maximum token (default:MAX_USABLE_LENGTH) Returns: (truncated_prompt, truncated_system_prompt)'
        #
        prompt_tokens = self._estimate_tokens(prompt)
        system_tokens = self._estimate_tokens(system_prompt)
        total_tokens = prompt_tokens + system_tokens
        
        _debug_print(f"[INFO] Token statistics: prompt={prompt_tokens}, system={system_tokens}, total={total_tokens}, max={max_total_tokens}")
        
        if total_tokens <= max_total_tokens:
            return prompt, system_prompt
        
        #
        _debug_print(f"[WARN] token ({total_tokens}) ({max_total_tokens}), start...")
        
        # : prefer keeping system_prompt(), prompt
        # system_prompt,
        system_max_tokens = min(system_tokens, max_total_tokens // 3)  # system_prompt 1/3
        prompt_max_tokens = max_total_tokens - system_max_tokens
        
        # system_prompt(keep,)
        if system_tokens > system_max_tokens:
            _debug_print(f"[WARN] system_prompt: {system_tokens} -> {system_max_tokens} tokens")
            system_prompt = self._truncate_text(system_prompt, system_max_tokens, preserve_end=False)
            system_tokens = self._estimate_tokens(system_prompt)
        
        # calculate prompt available empty(5%, token)
        prompt_max_tokens = int((max_total_tokens - system_tokens) * 0.95)  # use 95% is not 100%
        
        # prompt(keep,)
        if prompt_tokens > prompt_max_tokens:
            _debug_print(f"[WARN] prompt: {prompt_tokens} -> {prompt_max_tokens} tokens")
            prompt = self._truncate_text(prompt, prompt_max_tokens, preserve_end=False)
            prompt_tokens = self._estimate_tokens(prompt)
        
        # :, continue (5)
        max_iterations = 5
        iteration = 0
        while iteration < max_iterations:
            final_total = prompt_tokens + system_tokens
            if final_total <= max_total_tokens:
                break
            
            # , ()
            _debug_print(f"[WARN] {iteration + 1}: ({final_total} > {max_total_tokens}), continue...")
            excess_ratio = max_total_tokens / final_total
            
            # system_prompt
            if system_tokens > 0:
                new_system_max = int(system_tokens * excess_ratio * 0.9)  # 10%
                if new_system_max < system_tokens:
                    system_prompt = self._truncate_text(system_prompt, new_system_max, preserve_end=False)
                    system_tokens = self._estimate_tokens(system_prompt)
            
            # prompt
            if prompt_tokens > 0:
                new_prompt_max = int(prompt_tokens * excess_ratio * 0.9)  # 10%
                if new_prompt_max < prompt_tokens:
                    prompt = self._truncate_text(prompt, new_prompt_max, preserve_end=False)
                    prompt_tokens = self._estimate_tokens(prompt)
            
            iteration += 1
        
        final_total = prompt_tokens + system_tokens
        _debug_print(f"[INFO] complete: prompt={prompt_tokens}, system={system_tokens}, total={final_total}, max={max_total_tokens}")
        
        # final check:,
        if final_total > max_total_tokens:
            _debug_print(f"[ERROR] ({final_total} > {max_total_tokens}), 80%")
            force_max = int(max_total_tokens * 0.8)
            system_max = min(system_tokens, force_max // 3)
            prompt_max = force_max - system_max
            
            if system_tokens > system_max:
                system_prompt = self._truncate_text(system_prompt, system_max, preserve_end=False)
            if prompt_tokens > prompt_max:
                prompt = self._truncate_text(prompt, prompt_max, preserve_end=False)
            
            final_total = self._estimate_tokens(prompt) + self._estimate_tokens(system_prompt)
            _debug_print(f"[INFO]: prompt={self._estimate_tokens(prompt)}, system={self._estimate_tokens(system_prompt)}, total={final_total}")
        
        return prompt, system_prompt
    
    def _init_full_task(self, user_query: str):
        'initialize full_task data structure Args: user_query: user query'
        self.full_task = {
            "task": user_query,
            "tools_type": [],
            "steps": [],
            "result": "",
            "notes": ""
        }
    
    def _update_full_task_tools_type(self, tools_type_list: List[str]):
        'update full_task tools_type Args: tools_type_list: tool type list'
        if self.full_task is not None:
            self.full_task["tools_type"] = tools_type_list
    
    def _update_full_task_steps(self, steps: List[Dict[str, Any]]):
        'update full_task steps Args: steps: list, step, tool_type, tool_name, requirements_description, arguments'
        if self.full_task is not None:
            # tool_type is empty, try tools.json tool_name
            tool_name_to_type_cache = {}
            tool_name = None
            tool_type = None
            
            # format steps, field
            formatted_steps = []
            for step in steps:
                tool_type = step.get("tool_type", "")
                tool_name = step.get("tool_name", "")
                
                # tool_type is empty tool_name exists, try tool_type.json
                if not tool_type and tool_name:
                    # cache, try to load tool_type.json
                    if tool_name not in tool_name_to_type_cache:
                        try:
                            tool_type_path = os.path.abspath(
                                os.path.join(os.path.dirname(__file__), '../configs/tool_type.json')
                            )
                            if os.path.exists(tool_type_path):
                                with open(tool_type_path, 'r', encoding='utf-8') as f:
                                    tool_type_config = json.load(f)
                                # tool_type.json format, tool_type, tools (tool name list)
                                if isinstance(tool_type_config, list):
                                    for category in tool_type_config:
                                        cat_tool_type = category.get('tool_type', '')
                                        tools_list = category.get('tools', [])
                                        for t_name in tools_list:
                                            if t_name:
                                                tool_name_to_type_cache[t_name] = cat_tool_type
                        except Exception as e:
                            _debug_print(f"[WARN] load tool_type.json tool_type failed: {e}")
                    
                    # cache
                    tool_type = tool_name_to_type_cache.get(tool_name, "")
                    if tool_type:
                        _debug_print(f"[INFO] tool_type.json tool_type: {tool_name} -> {tool_type}")
                
                formatted_step = {
                    "step": step.get("id", step.get("step", 0)),
                    "tool_type": tool_type,
                    "tool_name": tool_name,
                    "requirements_description": step.get("requirements_description", ""),
                    "arguments": step.get("arguments", {})
                }
                formatted_steps.append(formatted_step)
            self.full_task["steps"] = formatted_steps
    
    def _update_full_task_result(self, success: bool, error_message: str = ""):
        'update full_task result notes Args: success: success error_message: error (failed)'
        if self.full_task is not None:
            if success:
                self.full_task["result"] = "Successfully."
            else:
                self.full_task["result"] = f"Failed: {error_message}"
    

    @staticmethod
    def _extract_redacted_reasoning(text: str) -> str:
        'extract </think> label final output output </think> label, return label. label, return original. Args: LLM original output Returns: extract (</think> label content)'
        if not text:
            return text
        
        # </think> label(size, supports)
        pattern = r'</think>'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            # label, extractlabel
            extracted = text[match.end():].strip()
            _debug_print(f"[INFO] </think> label, extractlabel (: {len(extracted)})")
            return extracted
        else:
            # label, return original
            return text
    
    @staticmethod
    def _extract_plan_block(text: str) -> Optional[str]:
        "extract <PLAN>...<END_OF_PLAN> content, supports </PLAN>/</END_OF_PLAN>. missing, try: '[' ']' content."
        start_markers = ["<PLAN>", "</PLAN>"]
        end_markers = ["<END_OF_PLAN>", "</END_OF_PLAN>"]

        start_idx = -1
        for m in start_markers:
            idx = text.find(m)
            if idx != -1:
                start_idx = idx + len(m)
                break

        end_idx = -1
        if start_idx != -1:
            for m in end_markers:
                idx = text.find(m, start_idx)
                if idx != -1:
                    end_idx = idx
                    break

        # complete, JSON
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            l_br = text.find('[')
            r_br = text.rfind(']')
            if l_br != -1 and r_br != -1 and r_br > l_br:
                return text[l_br:r_br + 1].strip()
            return None

        return text[start_idx:end_idx].strip()
    
    def get_tool_name(self, plan_data: List[Dict[str, Any]], user_query: str, max_retries: int = 10) -> List[Dict[str, Any]]:
    
        _debug_print(f"[INFO] start tool...")
        
        # check RAG initialize
        if self._embedding_function is None:
            raise RuntimeError(
                'RAG successfully initialized, _embedding_function is empty, embedding.'
                'check RAG initialize log.'
            )
        
        # plan_data format
        if not isinstance(plan_data, list) or len(plan_data) == 0:
            _debug_print(f"[WARN] plan_data is not list is empty, try generateplan...")
            retry_count = 0
            while retry_count < max_retries:
                retry_count += 1
                _debug_print(f"[INFO] retry {retry_count}/{max_retries}, generateplan...")
                plan_data = self.plan_with_tool_types(user_query)
                if isinstance(plan_data, list) and len(plan_data) > 0:
                    _debug_print(f"[SUCCESS] successfully got plan data, {len(plan_data)}")
                    break
            else:
                raise RuntimeError(
                    f"maximum retry ({max_retries}), get plan data."
                )
        
        _debug_print(f"[SUCCESS] plan data, {len(plan_data)}")
        
        # 2: load tools.json config
        tools_json_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../configs/tools.json')
        )
        
        if not os.path.exists(tools_json_path):
            raise FileNotFoundError(f"tools.json file does not exist: {tools_json_path}")
        
        with open(tools_json_path, 'r', encoding='utf-8') as f:
            tools_config = json.load(f)
        
        # tools_config list format()
        if isinstance(tools_config, dict):
            # format, convert format
            tools_list = []
            if 'categories' in tools_config:
                for category in tools_config['categories']:
                    tools_list.extend(category.get('tools', []))
            tools_config = tools_list
        elif not isinstance(tools_config, list):
            raise ValueError(f"tools_config format unsupported: {type(tools_config)}")
        
        # tool_type.json tool name and tool type
        tool_name_to_type = {}
        try:
            tool_type_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../configs/tool_type.json')
            )
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
        
        # tool_type -> tools
        tool_type_to_tools: Dict[str, List[Dict[str, str]]] = {}
        for tool in tools_config:
            if not isinstance(tool, dict):
                continue
            tool_name = tool.get('name', '')
            tool_desc = tool.get('description', '')
            if not tool_name or not tool_desc:
                continue
            # tool type
            tool_type = tool_name_to_type.get(tool_name, '')
            if tool_type:
                if tool_type not in tool_type_to_tools:
                    tool_type_to_tools[tool_type] = []
                tool_type_to_tools[tool_type].append({
                    'name': tool_name,
                    'description': tool_desc
                })
        
        _debug_print(f"[INFO] load {len(tool_type_to_tools)} tool type config")
        
        # 3: embedding
        results = []
        
        for step_item in plan_data:
            step_id = step_item.get('id', 0)
            tool_type = step_item.get('tool_type', '')
            requirements_desc = step_item.get('requirements_description', '')
            
            _debug_print(f"\n[INFO] {step_id}: {tool_type}")
            _debug_print(f"[INFO]: {requirements_desc[:100]}...")
            
            # check tool_type exists
            if tool_type not in tool_type_to_tools:
                _debug_print(f"[WARN] tool type '{tool_type}' tools.json does not exist")
                # data basic tool_name(None)
                result_item = {
                    **step_item,  # keep field
                    "tool_name": None  # , None
                }
                results.append(result_item)
                continue
            
            # get tool_type tool
            available_tools = tool_type_to_tools[tool_type]
            
            if len(available_tools) == 0:
                _debug_print(f"[WARN] tool type '{tool_type}' available tool")
                # data basic tool_name(None)
                result_item = {
                    **step_item,  # keep field
                    "tool_name": None  # , None
                }
                results.append(result_item)
                continue
            
            # use embedding current tool_type tool sort
            tool_scores = []
            for tool in available_tools:
                tool_name = tool['name']
                tool_desc = tool['description']
                score = 0.0
                try:
                    if self._embedding_function is not None:
                        _debug_print(f"[INFO] use embedding tool {tool_name}")
                        # ===: plan_with_tool_types ===
                        # 1) query uses embed_query(search_query)
                        enhanced_req = f"Requirement: {requirements_desc}. Tool type: {tool_type}"
                        query_embedding = self._embedding_function.embed_query(enhanced_req)
                        # 2) tool use embed_documents(search_document)
                        tool_doc = f"Tool: {tool_name}. Description: {tool_desc}"
                        tool_embedding = self._embedding_function.embed_documents([tool_doc])[0]
                        score = float(self._compute_similarity(query_embedding, tool_embedding))
                except Exception as e:
                    _debug_print(f"[WARN] tool {tool_name} failed: {e}")
                tool_scores.append({
                    'name': tool_name,
                    'description': tool_desc,
                    'score': score
                })
            
            # sort, 2
            tool_scores.sort(key=lambda x: x['score'], reverse=True)
            top_2_tools = tool_scores[:2]
            
            _debug_print(f"[SUCCESS] complete, 2 tool(embedding similar sort):")
            for i, tool in enumerate(top_2_tools, 1):
                _debug_print(f"{i}. {tool['name']}")
                _debug_print(f": {tool['description'][:100]}...")
            
            # use best_tool_name tool
            tool_name = None
            try:
                _debug_print(f"[INFO] use LLM tool...")
                tool_name = self.best_tool_name(
                    requirements_description=requirements_desc,
                    tool_type=tool_type,
                    matched_tools=top_2_tools
                )
            except Exception as e:
                _debug_print(f"[WARN] tool failed: {e}")
                # failed, usesimilar tool
                if top_2_tools:
                    tool_name = top_2_tools[0]['name']
                    _debug_print(f"[INFO] usesimilar tool: {tool_name}")
            
            # data basic tool_name matched_tools field
            result_item = {
                **step_item,  # keep field
                "tool_name": tool_name,  # tool_name field
                "matched_tools": top_2_tools  # save tool list, use
            }
            results.append(result_item)
        
        _debug_print(f"\n[SUCCESS] complete, {len(results)}")
        
        # update full_task: extract tools_type steps
        if self.full_task is not None:
            # extract tool_type
            tools_type_set = set()
            for item in results:
                tool_type = item.get('tool_type', '')
                if tool_type:
                    tools_type_set.add(tool_type)
            self._update_full_task_tools_type(list(tools_type_set))
            # update steps(arguments, generate_tool_parameters update)
            self._update_full_task_steps(results)
        
        return results
    
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
        if not _RAG_AVAILABLE:
            raise RuntimeError('numpy not installed, calculate similar')
        
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # calculate similar
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    
    def generate_tool_dependencies(
        self,
        toolchain_calls: List[Dict[str, Any]],
        max_retries: int = 5
    ) -> List[Dict[str, Any]]:
        'analysis tool call dependencies, return dependency task list. Base Agent test return toolchain_calls format, use LLM analyze task dependency, return id, tool_name, arguments, dependencies task list, DAG. Args: toolchain_calls: Base Agent test return toolchain_calls list, format: [{"tool": "tool_name", "arguments": {...}},...] max_retries: maximum retry (default:5) Returns: List[Dict[str, Any]]: dependency task list, format: [{"id": 1, "tool_name": "...", "arguments": {...}, "dependencies": [...]},...]'
        # input
        if not toolchain_calls or len(toolchain_calls) == 0:
            raise ValueError('toolchain_calls is empty')
        
        # tool call id(1 start), dependency analysis
        # input data: toolchain_calls convert id format
        # id -> original data,
        input_data = []
        id_to_input = {}  # {id: {"tool": "...", "arguments": {...}}}
        
        for idx, call in enumerate(toolchain_calls, 1):
            if not isinstance(call, dict):
                _debug_print(f"[WARN] toolchain_calls[{idx-1}] is not, skip")
                continue
            tool_name = call.get("tool", "")
            arguments = call.get("arguments", {})
            if not tool_name:
                _debug_print(f"[WARN] toolchain_calls[{idx-1}] missing 'tool' field, skip")
                continue
            input_item = {
                "id": idx,
                "tool": tool_name,
                "arguments": arguments
            }
            input_data.append(input_item)
            id_to_input[idx] = {"tool": tool_name, "arguments": arguments}
        
        if len(input_data) == 0:
            raise ValueError('toolchain_calls tool call')
        
        # get id, dependency
        valid_ids = set(id_to_input.keys())
        
        # format input data JSON (LLM)
        result_json = json.dumps(input_data, ensure_ascii=False, indent=2)
        
        _debug_print(f"[INFO] start analysis tool dependency...")
        _debug_print(f"[INFO] input tool call count: {len(input_data)}")
        _debug_print(f"[INFO] task ID: {sorted(valid_ids)}")
        
        # prompt - use replace, format() JSON
        prompt = prompt_templates.Generate_tool_dependencies
        prompt = prompt.replace('{context}', str(prompt_templates.context))
        prompt = prompt.replace('{result}', result_json)
        
        # retry
        retry_count = 0
        dependency_data = None
        
        while retry_count < max_retries:
            try:
                
                # call LLM
                response = self.client.chat(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # extract </think> label
                response = self._extract_redacted_reasoning(response)
                
                _debug_print(f"[INFO] LLM response: {response[:200]}...")
                
                # response, extract PLAN content
                plan_content = self._extract_plan_block(response)
                if plan_content:
                    # JSON, failedretry
                    try:
                        dependency_data = json.loads(plan_content)
                    except json.JSONDecodeError as e:
                        _debug_print(f"[WARN] PLAN content JSON failed: {e}")
                        _debug_print(f"[WARN] try content: {plan_content[:200]}...")
                        retry_count += 1
                        if retry_count < max_retries:
                            _debug_print(f"[INFO] retry {retry_count}/{max_retries}, analysis dependency...")
                            continue
                        else:
                            raise RuntimeError(
                                f"maximum retry ({max_retries}), get dependency."
                                f"JSON failed: {e}"
                            )
                    
                    # return format
                    if isinstance(dependency_data, list) and len(dependency_data) > 0:
                        # field
                        valid = True
                        missing_fields = []
                        invalid_dependencies = []
                        mismatched_tools = []
                        missing_ids = []
                        
                        returned_ids = set()
                        
                        for idx, item in enumerate(dependency_data):
                            if not isinstance(item, dict):
                                valid = False
                                missing_fields.append(f"{idx}: is notdictionary type")
                                break
                            
                            # check field: id, tool_name, arguments, dependencies
                            required_fields = ['id', 'tool_name', 'arguments', 'dependencies']
                            missing = [field for field in required_fields if field not in item]
                            if missing:
                                valid = False
                                missing_fields.append(f"{idx} (id={item.get('id', 'N/A')}): missing field {missing}")
                                continue
                            
                            task_id = item.get('id')
                            tool_name = item.get('tool_name', '')
                            dependencies = item.get('dependencies', [])
                            
                            # check id
                            if task_id in returned_ids:
                                valid = False
                                missing_fields.append(f"{idx}: task ID {task_id}")
                                continue
                            returned_ids.add(task_id)
                            
                            # check id range
                            if task_id not in valid_ids:
                                valid = False
                                missing_ids.append(f"task ID {task_id} range {valid_ids}")
                                continue
                            
                            # tool_name input
                            expected_tool = id_to_input[task_id].get('tool', '')
                            if tool_name != expected_tool:
                                valid = False
                                mismatched_tools.append(
                                    f"task ID {task_id}: tool '{expected_tool}', return '{tool_name}'"
                                )
                                continue
                            
                            # dependency: dependency id exists current id
                            if not isinstance(dependencies, list):
                                valid = False
                                invalid_dependencies.append(f"task ID {task_id}: dependencies is not a list")
                                continue
                            
                            for dep_id in dependencies:
                                if not isinstance(dep_id, int):
                                    valid = False
                                    invalid_dependencies.append(f"task ID {task_id}: dependency ID {dep_id} is not")
                                    break
                                if dep_id not in valid_ids:
                                    valid = False
                                    invalid_dependencies.append(f"task ID {task_id}: dependency ID {dep_id} does not exist")
                                    break
                                if dep_id == task_id:
                                    valid = False
                                    invalid_dependencies.append(f"task ID {task_id}: dependency")
                                    break
                            
                            # return task count input
                            if len(dependency_data) != len(input_data):
                                valid = False
                                missing_fields.append(
                                    f"return task count ({len(dependency_data)}) input count ({len(input_data)})"
                                )
                        
                        # check id
                        missing_task_ids = valid_ids - returned_ids
                        if missing_task_ids:
                            valid = False
                            missing_ids.append(f"missing task ID: {sorted(missing_task_ids)}")
                        
                        if valid:
                            _debug_print(f"[SUCCESS] successful dependency, {len(dependency_data)} task")
                            
                            # output dependency summary
                            for item in dependency_data:
                                deps = item.get('dependencies', [])
                                tool_name = item.get('tool_name', 'Unknown')
                                task_id = item.get('id', 'N/A')
                                if deps:
                                    _debug_print(f"task {task_id} ({tool_name}) dependency: {deps}")
                                else:
                                    _debug_print(f"task {task_id} ({tool_name}) dependency(parallel)")
                            
                            return dependency_data
                        else:
                            _debug_print(f"[WARN] return data format:")
                            if missing_fields:
                                for msg in missing_fields:
                                    _debug_print(f"- {msg}")
                            if invalid_dependencies:
                                for msg in invalid_dependencies:
                                    _debug_print(f"- {msg}")
                            if mismatched_tools:
                                for msg in mismatched_tools:
                                    _debug_print(f"- {msg}")
                            if missing_ids:
                                for msg in missing_ids:
                                    _debug_print(f"- {msg}")
                            retry_count += 1
                            if retry_count < max_retries:
                                _debug_print(f"[INFO] retry {retry_count}/{max_retries}, analysis dependency...")
                                continue
                    else:
                        _debug_print(f"[WARN] PLAN result is not list is empty")
                        _debug_print(f"[WARN] PLAN content: {plan_content[:500]}")
                else:
                    _debug_print(f"[WARN] response not found <PLAN> <END_OF_PLAN>")
                    _debug_print(f"[WARN] response content: {response[:500]}")
                
            except json.JSONDecodeError as e:
                _debug_print(f"[WARN] JSON failed: {e}")
                if 'plan_content' in locals():
                    _debug_print(f"[WARN] try content: {plan_content[:500]}")
            except Exception as e:
                _debug_print(f"[WARN] response: {e}")
                traceback.print_exc()
            
            # retry
            retry_count += 1
            if retry_count < max_retries:
                _debug_print(f"[INFO] retry {retry_count}/{max_retries}, analysis dependency...")
            else:
                raise RuntimeError(
                    f"maximum retry ({max_retries}), get dependency."
                    f"check LLM response format."
                )
        
        # ,
        if dependency_data is None:
            raise RuntimeError(
                f"get dependency data."
                f"check LLM response format."
            )
        
        return dependency_data
    

    def _extract_existing_paths(
        self,
        obj: Any,
        seen: Optional[set] = None,
        parent_key: Optional[str] = None,
    ) -> List[str]:
        'extract file path. supports folder path: field "path field"(output_path/file_path/path) folder, expand folder file, args. root config field data output directory expand. same-name file (.htm.json), prefer .json file.'
        if seen is None:
            seen = set()
        paths: List[str] = []
        if id(obj) in seen:
            return paths
        seen.add(id(obj))

        if isinstance(obj, str):
            candidate = os.path.expanduser(obj)
            candidate_abs = os.path.abspath(candidate)
            key = (parent_key or "").strip().lower()
            is_path_key = bool(key) and (key == "path" or key.endswith("path") or key.endswith("paths"))
            
            if os.path.isfile(candidate_abs):
                # file, direct
                paths.append(candidate_abs)
            elif os.path.isdir(candidate_abs):
                # folder: field "path field" expand(root='./data' expand)
                if is_path_key:
                    _debug_print(f"[INFO] folder path: {candidate_abs}, expand folder content...")
                    folder_files = self._expand_folder_paths(candidate_abs)
                    paths.extend(folder_files)
                    _debug_print(f"[INFO] folder {candidate_abs} extract {len(folder_files)} file")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                paths.extend(self._extract_existing_paths(v, seen, parent_key=str(k)))
        elif isinstance(obj, list):
            for item in obj:
                paths.extend(self._extract_existing_paths(item, seen, parent_key=parent_key))
        
        # same-name file: same name.htm.json, keep.json
        paths = self._deduplicate_paths_by_name(paths)
        
        return paths
    
    def _expand_folder_paths(self, folder_path: str) -> List[str]:
        'expand folder path, return folder file path. Args: folder_path: folder path Returns: folder file path list'
        file_paths = []
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.abspath(os.path.join(root, file))
                    file_paths.append(file_path)
        except Exception as e:
            _debug_print(f"[WARN] expand folder {folder_path} failed: {e}")
        
        return file_paths
    
    def _deduplicate_paths_by_name(self, paths: List[str]) -> List[str]:
        'same-name file: same-name file (.htm.json), prefer .json file.: -/path/to/file.htm/path/to/file.json -> keep/path/to/file.json -/path/to/file1.json/path/to/file2.htm -> keep(same name) Args: paths: file path list Returns: file path list(prefer keeping.json file)'
        if not paths:
            return paths
        
        # file ()
        name_to_paths: Dict[str, List[str]] = {}
        
        for path in paths:
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

            fields: List[str] = []
            if ext == '.csv':
                first_line = head.splitlines()[0] if head else ""
                if first_line and "," in first_line:
                    fields = [c.strip() for c in first_line.split(",") if c.strip()][:50]
            else:
                seen = set()
                for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_\- ]{0,40})"\s*:', head):
                    k = m.group(1).strip()
                    if k and k not in seen:
                        seen.add(k)
                        fields.append(k)
                    if len(fields) >= 50:
                        break

            dates_head = re.findall(r"\d{4}-\d{2}-\d{2}", head)
            dates_tail = re.findall(r"\d{4}-\d{2}-\d{2}", tail) if tail else []
            start_date = dates_head[0] if dates_head else ""
            end_date = (dates_tail[-1] if dates_tail else (dates_head[-1] if dates_head else ""))

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
        
        paths = self._extract_existing_paths(dag_results)
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
        
        # file path generate (file + path)
        file_descriptions = []
        for path in unique_paths:
            filename = os.path.basename(path)
            # extract path (year, report type)
            path_parts = path.split(os.sep)
            relevant_parts = [p for p in path_parts[-3:] if p]  # 3 path
            description = f"{filename} {' '.join(relevant_parts)}"
            file_descriptions.append((path, description))
        
        # calculate query(cache)
        query_embedding = self._embed_query_cached(task)

        # calculate file query related (similarity + similar cache)
        file_scores = []
        descriptions_only = [desc for _, desc in file_descriptions]
        doc_embeddings = self._embed_documents_cached(descriptions_only)

        # similar cache, calculate
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

        # file_scores()
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
        import time
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
                    _debug_print(f"[INFO] [collect] ({idx+1}/{total_files}) done(stock_summary) size={file_size} chars cost={time.time()-t0:.2f}s")
                    return idx, item

                # file: direct return
                if file_size <= max_chars:
                    item = {
                        "path": path,
                        "content": full_text,
                        "relevance_score": float(relevance_score),
                        "extraction_method": "full",
                        "original_size": file_size,
                        "is_stock_data": False,
                    }
                    _debug_print(f"[INFO] [collect] ({idx+1}/{total_files}) done(full) size={file_size} chars cost={time.time()-t0:.2f}s")
                    return idx, item

                # file: use extract(embedding)
                with embed_sem:
                    _debug_print(f"[INFO] [collect] ({idx+1}/{total_files}) large_file extract begin size={file_size} chars")
                    if use_llamaindex and _LLAMAINDEX_AVAILABLE:
                        extracted_content = self._extract_relevant_content_with_llamaindex(
                            full_text, task, chunk_size, chunk_overlap, top_k_chunks
                        )
                        method = "llamaindex_rag"
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
                _debug_print(f"[INFO] [collect] ({idx+1}/{total_files}) done({method}) size={file_size} chars cost={time.time()-t0:.2f}s")
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

        # selected_files output
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
        'use Llama Index extract query related fragment. Args: complete query: query chunk_size: size chunk_overlap: top_k: return top-k related Returns: extract related content'
        try:
            if not _LLAMAINDEX_AVAILABLE:
                raise ImportError('Llama Index unavailable')
            
            # Document
            document = Document(text=text)
            
            #
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            #
            nodes = node_parser.get_nodes_from_documents([document])
            
            if not nodes:
                # , return
                return text[:chunk_size * top_k] + f"\n... [truncated, total {len(text)} chars]"
            
            # calculate query related (use embedding)
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
                
                # related sort, top-k
                node_scores.sort(key=lambda x: x[1], reverse=True)
                selected_nodes = node_scores[:top_k]
                
                # original
                selected_nodes_sorted = sorted(selected_nodes, key=lambda x: nodes.index(x[0]))
                extracted_texts = []
                for node, score in selected_nodes_sorted:
                    content = node.get_content() if hasattr(node, 'get_content') else str(node)
                    extracted_texts.append(content)
                
                result = '... [related fragment]...'.join(extracted_texts)
                result = f"[extract {len(selected_nodes)} related fragment, {len(result)} ]\n\n{result}"
                
                return result
            else:
                # embedding, return top_k
                selected_nodes = nodes[:top_k]
                extracted_texts = []
                for node in selected_nodes:
                    content = node.get_content() if hasattr(node, 'get_content') else str(node)
                    extracted_texts.append(content)
                return '... [related fragment]...'.join(extracted_texts)
                
        except Exception as e:
            _debug_print(f"[WARN] Llama Index extract failed: {e},")
            traceback.print_exc()
            #
            return self._extract_relevant_chunks_simple(text, query, chunk_size, top_k)
    
    def _extract_relevant_chunks_simple(
        self,
        text: str,
        query: str,
        chunk_size: int = 1000,
        top_k: int = 5,
    ) -> str:
        'based on embedding related content extract(use Llama Index). Args: complete query: query chunk_size: size top_k: return top-k related Returns: extract related content'
        if not self._embedding_function:
            # embedding,
            return text[:chunk_size * top_k] + f"\n... [truncated, total {len(text)} chars]"
        
        #
        chunks = self._split_text_into_chunks(text, chunk_size, chunk_size // 5)
        
        if not chunks:
            return text[:chunk_size * top_k] + f"\n... [truncated, total {len(text)} chars]"

        # === Two-stage: chunk, embedding count ===
        rough_m = int(os.getenv("ProFinAgent_CHUNK_ROUGH_TOP_M", "64"))
        rough_m = max(top_k, max(8, rough_m))
        chunk_texts_all = [chunk["text"] for chunk in chunks]
        cand_idx = self._rough_select_indices(chunk_texts_all, query, rough_m)
        cand_chunks = [chunks[i] for i in cand_idx]
        cand_texts = [chunk_texts_all[i] for i in cand_idx]

        _debug_print(f"[INFO] chunk rough filter: total={len(chunks)} -> candidates={len(cand_chunks)} (top_m={rough_m})")

        # calculate candidate query related (similarity + similar cache)
        query_embedding = self._embed_query_cached(query)
        cand_embeddings = self._embed_documents_cached(cand_texts)

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
                miss_vecs.append(cand_embeddings[i])

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

    def _extract_first_json(self, text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        'extract JSON.'
        if not text:
            return None
        
        # { [
        first_brace = text.find("{")
        first_bracket = text.find("[")
        
        if first_brace == -1 and first_bracket == -1:
            return None
            
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            #
            start = first_brace
            end_char = "}"
        else:
            #
            start = first_bracket
            end_char = "]"
            
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception as e:
                _debug_print(f"[WARN] extract JSON failed: {e}")
        return None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
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
        'JSON: prefer json. loads, extract <PLAN>/<END_OF_PLAN>, extract JSON, {}/[]. failed return None(retry).'
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

        # 2) try <PLAN>...<END_OF_PLAN> extract(prompt)
        try:
            plan = self._extract_plan_block(s)
            if plan:
                try:
                    return json.loads(plan)
                except Exception:
                    pass
        except Exception:
            pass

        # 3) extract JSON
        try:
            extracted = self._extract_first_json(s)
            if extracted is not None:
                return extracted
        except Exception:
            pass

        # 4)/
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

    def _sanitize_for_json(self, obj: Any, seen: Optional[set] = None) -> Any:
        'clean, JSON serialize convert. TextContent, MCP response type.'
        if seen is None:
            seen = set()
        
        # reference
        obj_id = id(obj)
        if obj_id in seen:
            return "[Circular Reference]"
        seen.add(obj_id)
        
        # type(directserialize)
        if isinstance(obj, (str, int, float, bool, type(None))):
            seen.remove(obj_id)
            return obj
        
        # dictionary
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                result[str(k)] = self._sanitize_for_json(v, seen)
            seen.remove(obj_id)
            return result
        
        # list
        if isinstance(obj, (list, tuple)):
            result = [self._sanitize_for_json(item, seen) for item in obj]
            seen.remove(obj_id)
            return result
        
        # type (TextContent)
        try:
            # prefer trying extract attribute (MCP TextContent attribute)
            if hasattr(obj, 'text'):
                text_value = getattr(obj, 'text', None)
                seen.remove(obj_id)
                return self._sanitize_for_json(text_value, seen)
            
            # __dict__, try convert dictionary
            if hasattr(obj, '__dict__'):
                obj_dict = {}
                for k, v in obj.__dict__.items():
                    # skip attribute
                    if not k.startswith('_') or k in ['_text', '_content']:
                        obj_dict[k] = self._sanitize_for_json(v, seen)
                seen.remove(obj_id)
                return obj_dict
            
            # try convert
            str_repr = str(obj)
            seen.remove(obj_id)
            return str_repr
            
        except Exception as e:
            _debug_print(f"[WARN] clean failed: {type(obj).__name__}, {e}")
            seen.remove(obj_id)
            return f"[{type(obj).__name__}]"

    def generate_final_answer(
        self,
        task: str,
        dag_results: Dict[str, Any],
        file_contents: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        'use End_task, DAG result file content generate final answer. test, current task similar notebook, prompt.'
        # clean dag_results serialize
        sanitized_dag_results = self._sanitize_for_json(dag_results)
        
        payload = {
            "files": file_contents or [],
            "dag_results": sanitized_dag_results or {},
        }
        result_str = json.dumps(payload, ensure_ascii=False, indent=2)
        
        # check result_str token, MAX_USABLE_LENGTH, 85%
        result_str_tokens = self._estimate_tokens(result_str)
        max_result_tokens = int(MAX_USABLE_LENGTH * 0.7)  # 80%
        
        if result_str_tokens > max_result_tokens:
            _debug_print(f"[WARN] result_str token ({result_str_tokens}) ({max_result_tokens}), start...")
            
            # try: prefer file_contents content
            # file_contents exists,
            if payload.get("files") and isinstance(payload["files"], list):
                _debug_print(f"[INFO] try file_contents...")
                truncated_files = []
                for file_item in payload["files"]:
                    if isinstance(file_item, dict) and "content" in file_item:
                        content = file_item.get("content", "")
                        if content:
                            # file content token
                            content_tokens = self._estimate_tokens(content)
                            # file content 10%,
                            max_file_content_tokens = int(max_result_tokens * 0.1)
                            if content_tokens > max_file_content_tokens:
                                _debug_print(f"[WARN] file {file_item.get('path', 'unknown')} content ({content_tokens} tokens), {max_file_content_tokens} tokens")
                                truncated_content = self._truncate_text(content, max_file_content_tokens, preserve_end=False)
                                file_item = file_item.copy()
                                file_item["content"] = truncated_content
                    truncated_files.append(file_item)
                payload["files"] = truncated_files
            
            # generate result_str
            result_str = json.dumps(payload, ensure_ascii=False, indent=2)
            result_str_tokens = self._estimate_tokens(result_str)
            
            # , direct result_str(keep JSON structure)
            if result_str_tokens > max_result_tokens:
                _debug_print(f"[WARN] file_contents ({result_str_tokens} > {max_result_tokens}), direct result_str...")
                # result_str, JSON format
                # : use, JSON format
                truncated_result_str = self._truncate_json_string(result_str, max_result_tokens)
                if truncated_result_str:
                    result_str = truncated_result_str
                    result_str_tokens = self._estimate_tokens(result_str)
                    _debug_print(f"[INFO] result_str complete: {result_str_tokens} tokens")
                else:
                    # failed, use try JSON format
                    _debug_print(f"[WARN] JSON failed, use")
                    result_str = self._truncate_text(result_str, max_result_tokens, preserve_end=False)
                    # try JSON format()
                    if not result_str.rstrip().endswith('}'):
                        # try complete
                        last_brace = result_str.rfind('}')
                        if last_brace > 0:
                            result_str = result_str[:last_brace + 1]
                        else:
                            # ,
                            result_str = result_str.rstrip() + '}'
                    result_str_tokens = self._estimate_tokens(result_str)
                    _debug_print(f"[INFO] result_str complete: {result_str_tokens} tokens")
        else:
            _debug_print(f"[INFO] result_str token: {result_str_tokens}, ({max_result_tokens})")
        
        # test: current task similar notebook
        # _embedding_function None, use notebook
        if self._embedding_function is None:
            notebook_str = ""
        else:
            notebook_str = ""
            # notebook_entry = self._find_similar_notebook_entry(task)
            # notebook_str = (
            # json. dumps(notebook_entry, ensure_ascii=False, indent=2)
            # if notebook_entry
            # else ""
            # )
        
        prompt = prompt_templates.End_task.format(
            context=prompt_templates.context,
            task=task,
            result=result_str,
            notebook=notebook_str,
        )
        
        # check token()
        user_prompt = "Please provide the final answer JSON only. You can only answer based on your own knowledge and the data within the required time frame. Do not use any information outside the frame of the query time frame or fabricated content."
        
        
        response = self.client.chat(
            prompt=user_prompt,
            system_prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        # End_task use <END_OF_PLAN>, direc ract JSON
        parsed = self._extract_first_json(response)
        if isinstance(parsed, dict):
            return parsed.get("final_answer", "")
        return ""

    def no_tool_answer(
        self,
        task: str,
        max_retries: int = 5,
    ) -> str:
        'use tool directlyly based on task generate answer. use no_tool_answer prompt template, model directly uses user question, use tool. Args: task: user query/task max_retries: maximum retry (default:3) Returns: str: final answer (JSON extract final_answer field)'
        _debug_print(f"[INFO] start use tool task: {task[:100]}...")
        
        
        
        # prompt
        prompt = prompt_templates.no_tool_answer.format(
            context=prompt_templates.context,
            task=task,
        )
        
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # check token()
                user_prompt = "Please provide the final answer JSON only."
               
                response = self.client.chat(
                    prompt=user_prompt,
                    system_prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # no_tool_answer use <END_OF_PLAN>, extract JSON
                # try to use _extract_plan_block extract <END_OF_PLAN> content
                plan_content = self._extract_plan_block(response)
                if plan_content:
                    try:
                        parsed = json.loads(plan_content)
                    except json.JSONDecodeError:
                        # _extract_plan_block extract contentis not JSON, try _extract_first_json
                        parsed = self._extract_first_json(response)
                else:
                    # <END_OF_PLAN> label, use _extract_first_json
                    parsed = self._extract_first_json(response)
                
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
        _debug_print(f"[ERROR] maximum retry ({max_retries}), generate answer")
        if last_error:
            _debug_print(f"[ERROR] error: {last_error}")
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
        'current task generate reflection, write notebook.json. Args: task: task toolchain_calls: tool call final_answer: final answer reference_tools: tool list reference_answer: answer dag_results: DAG result dictionary {task_id: task_result} dag_status: DAG status("success", "error", "no_tools", "no_tasks") dag_error: DAG error () Returns: reflection'
        result_str = json.dumps(toolchain_calls, ensure_ascii=False, indent=2)
        reference_tools_str = ','.join(reference_tools) if reference_tools else ""
        dag_results_str = self._format_dag_results_for_reflection(
            dag_results=dag_results,
            dag_status=dag_status,
            dag_error=dag_error
        )
        
        prompt = prompt_templates.self_reflection.format(
            context=prompt_templates.context,
            task=task,
            result=result_str,
            dag_results=dag_results_str,
            reference_tools=reference_tools_str,
            final_answer=final_answer,
            reference_answer=reference_answer,
        )
        
        # check token()
        user_prompt = "Please output the self reflection JSON only."
       
        
        response = self.client.chat(
            prompt=user_prompt,
            system_prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # try extract <PLAN> label content(exists)
        plan_content = self._extract_plan_block(response)
        if plan_content:
            try:
                parsed = json.loads(plan_content)
            except json.JSONDecodeError:
                parsed = self._extract_first_json(response)
        else:
            parsed = self._extract_first_json(response)
        
        reflection_text = ""
        # directly use format dag_results_str, is not LLM response extract
        # save DAG result, is not LLM generate
        dag_results_in_reflection = dag_results_str
        
        if isinstance(parsed, dict):
            reflection_text = parsed.get("self_reflection", "")
            # LLM response extract dag_results, use format original data

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

        # append notebook
        entries = self._load_notebook_entries()
        entries.append({
            "task": task,
            "self_reflection": reflection_text,
            "dag_results": dag_results_obj  # save, JSON auto format
        })
        self._save_notebook_entries(entries)
        return reflection_text

    
    def test_connection(self) -> bool:
        'SGLang service Returns: success'
        return self.client.test_connection()
    
    def is_model_loaded(self) -> bool:
        'check model load Returns: model load'
        return self.client.is_model_loaded()
    
    def list_models(self) -> List[str]:
        'get service model name Returns: model name list'
        return self.client.list_models()
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        'load model SGLang service Args: model_path: model path(None, useself.model_path) Returns: load success'
        return self.client.load_model(model_path)
    
    def get_tool_type_description(self, tool_type: str) -> Optional[str]:
        'tool_type get description Args: tool_type: tool type name Returns: tool type; if it does not exist return None'
        return self.tool_types.get(tool_type)
    
    def list_tool_types(self) -> List[str]:
        'get available tool type list Returns: tool type name list'
        return list(self.tool_types.keys())
    
    def _load_tool_type_config(self) -> Dict[str, Any]:
        'load tool_type.json config.'
        tool_type_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../configs/tool_type.json')
        )
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
        'current question tool_type.json tool type. Args: query: user query/task max_retries: maximum retry (default:5) retry_sleep: retry interval() Returns: List[str]: tool type list'
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
        prompt = prompt_templates.choose_types.format(
            context=prompt_templates.context,
            query=query,
            tool_types=tool_types_str,
        )
        
        total_attempts = max(1, max_retries)
        last_error: Any = None
        
        for attempt in range(total_attempts):
            try:
                # check token()
                user_prompt = "Please output JSON only."
                
                
                # call LLM
                response = self.client.chat(
                    prompt=user_prompt,
                    system_prompt=prompt,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
                
                if not response:
                    raise ValueError("LLM return contentis empty")
                
                # extract JSON
                content = response.strip()
                if not (content.startswith('{') or content.startswith('[')):
                    first_brace = content.find('{')
                    if first_brace != -1:
                        end_brace = content.rfind('}')
                        if end_brace > first_brace:
                            content = content[first_brace:end_brace + 1]
                
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
                _debug_print(
                    f"[WARN] Base Agent {attempt + 1}/{total_attempts} tool type failed: {e}"
                )
                if attempt < total_attempts - 1 and retry_sleep > 0:
                    import time
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
                'RAG successfully initialized, _embedding_function is empty, embedding.'
                'check RAG initialize log.'
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
            tool_type_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../configs/tool_type.json')
            )
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

        strategy = load_tool_selection_strategy(
            env=os.getenv("ProFinAgent_RULES_ENV", None),
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
                        # exception,
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
                # candidate, candidate(quota)
                if not any(isinstance(td, dict) and td.get("name") == n for td, _ in candidate_tools):
                    candidate_tools.append((tool_obj, actual_type))

        _debug_print(
            f"[INFO] candidate tool (must-have): {len(candidate_tools)}; must-have: {must_have_names}"
        )

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
        if thresholded_tool_scores:
            tool_scores = thresholded_tool_scores
        else:
            _debug_print('[WARN] tool embedding threshold, use embedding sort 5 candidate')
            tool_scores = ranked_tool_scores[:5]

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
            # count,; continue (prefer)
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
    
    def _format_tools(self, tools: Union[str, Dict[str, Any], List[Any]]) -> str:
        'tools parameter convert prompt. supports: - file path: auto json. load json. dumps; - dict/list: direct json. dumps; -: return(format).'
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
        'read notebook.json, return list.'
        notebook_path = self.notebook_path
        if not notebook_path or not os.path.exists(notebook_path):
            return []
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as e:
            _debug_print(f"[WARN] read notebook failed: {e}")
        return []

    def _save_notebook_entries(self, entries: List[Dict[str, Any]]) -> None:
        'save notebook list.'
        notebook_path = self.notebook_path
        if not notebook_path:
            return
        try:
            os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _debug_print(f"[WARN] save notebook failed: {e}")

    def _find_similar_notebook_entry(
        self, task: str, threshold: float = 0.70
    ) -> Optional[Dict[str, Any]]:
        'use embedding similar notebook, return threshold.'
        entries = self._load_notebook_entries()
        if not entries:
            return None
        if self._embedding_function is None:
            return None

        try:
            query_vec = self._embedding_function.embed_query(task)
        except Exception as e:
            _debug_print(f"[WARN] notebook generate query failed: {e}")
            return None

        best = None
        best_score = -1.0
        for entry in entries:
            task_text = entry.get("task", "")
            if not task_text:
                continue
            try:
                doc_vec = self._embedding_function.embed_documents([task_text])[0]
                score = float(self._compute_similarity(query_vec, doc_vec))
                if score > best_score:
                    best_score = score
                    best = entry
            except Exception as e:
                _debug_print(f"[WARN] calculate notebook similar failed: {e}")
                continue

        if best is not None and best_score >= threshold:
            enriched = dict(best)
            enriched["_similarity"] = best_score
            return enriched
        return None
    
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
        notebook_str = "[]"  # use notebook
        
        # call LLM generate tool call
        total_attempts = max(1, max_retries)
        last_error: Any = None

        for attempt in range(total_attempts):
            try:
                system_content = prompt_templates.answer_straightforward.format(
                    context=prompt_templates.context,
                    question=question,
                    tools=tools_str,
                )

                # check token()
                user_prompt = "Please strictly follow the instructions above and output JSON only."
               
                
                # use SGLang Client chat
                response = self.client.chat(
                    prompt=user_prompt,
                    system_prompt=system_content,
                    temperature=0.0,
                    max_tokens=self.max_tokens
                )

                if not response:
                    raise ValueError("LLM return contentis empty")

                # try extract JSON(<PLAN> label, direct JSON)
                content = response
                
                # JSON extract: supports { } [ ], prefer <PLAN> label
                if "<PLAN>" in content and "</PLAN>" in content:
                    content = content.split("<PLAN>")[1].split("</PLAN>")[0].strip()
                
                content_s = content.strip()
                if not (content_s.startswith('{') or content_s.startswith('[')):
                    # try JSON structure
                    first_brace = content.find('{')
                    first_bracket = content.find('[')
                    
                    if first_brace != -1 and (first_bracket == -1 or (first_bracket != -1 and first_brace < first_bracket)):
                        #
                        start_idx = first_brace
                        end_char = '}'
                    elif first_bracket != -1:
                        #
                        start_idx = first_bracket
                        end_char = ']'
                    else:
                        start_idx = -1

                    if start_idx != -1:
                        end_idx = content.rfind(end_char)
                        if end_idx > start_idx:
                            content = content[start_idx:end_idx + 1]

                # JSON()
                data = self._safe_parse_json(content)
                if data is None:
                    raise ValueError('return content JSON')
                
                # compatible LLM direct return
                if isinstance(data, list):
                    data = {"toolchain_calls": data}

                if not isinstance(data, dict):
                    raise ValueError(f"return result is not JSON: {type(data)}")

                tc = data.get("toolchain_calls")
                if not isinstance(tc, list):
                    raise ValueError("return JSON missing 'toolchain_calls' list")

                # call field
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
                _debug_print(
                    f"[WARN] Base Agent {attempt + 1}/{total_attempts} generate toolchain_calls failed: {e}"
                )
                if attempt < total_attempts - 1 and retry_sleep > 0:
                    import time
                    time.sleep(retry_sleep)

        # retry failed, return error structure result
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
        'call LLM generate tool call, return JSON (toolchain_calls). Args: question: user question. tools: available tool: - tools.json file path - dict/list(tools.json content) - format (direct prompt) max_retries: LLM call JSON failed; maximum retry. retry_sleep: retry interval(). Returns: dict: "toolchain_calls" JSON.'
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
        notebook_entry = self._find_similar_notebook_entry(question)
        notebook_str = (
            json.dumps(notebook_entry, ensure_ascii=False, indent=2)
            if notebook_entry
            else "[]"
        )
        
        # 5: call LLM generate tool call
        total_attempts = max(1, max_retries)
        last_error: Any = None

        for attempt in range(total_attempts):
            try:
                system_content = prompt_templates.Test.format(
                    context=prompt_templates.context,
                    question=question,
                    tools=tools_str,
                    notebook=notebook_str,
                )

                # check token()
                user_prompt = "Please strictly follow the instructions above, select at least one tool, and output JSON only."
                
                # use SGLang Client chat
                response = self.client.chat(
                    prompt=user_prompt,
                    system_prompt=system_content,
                    temperature=0.0,
                    max_tokens=self.max_tokens
                )

                if not response:
                    raise ValueError("LLM return contentis empty")

                # try extract JSON(<PLAN> label, direct JSON)
                content = response
                
                # JSON extract: supports { } [ ], prefer <PLAN> label
                if "<PLAN>" in content and "</PLAN>" in content:
                    content = content.split("<PLAN>")[1].split("</PLAN>")[0].strip()
                
                content_s = content.strip()
                if not (content_s.startswith('{') or content_s.startswith('[')):
                    # try JSON structure
                    first_brace = content.find('{')
                    first_bracket = content.find('[')
                    
                    if first_brace != -1 and (first_bracket == -1 or (first_bracket != -1 and first_brace < first_bracket)):
                        #
                        start_idx = first_brace
                        end_char = '}'
                    elif first_bracket != -1:
                        #
                        start_idx = first_bracket
                        end_char = ']'
                    else:
                        start_idx = -1

                    if start_idx != -1:
                        end_idx = content.rfind(end_char)
                        if end_idx > start_idx:
                            content = content[start_idx:end_idx + 1]

                # JSON()
                data = self._safe_parse_json(content)
                if data is None:
                    raise ValueError('return content JSON')
                
                # compatible LLM direct return
                if isinstance(data, list):
                    data = {"toolchain_calls": data}

                if not isinstance(data, dict):
                    raise ValueError(f"return result is not JSON: {type(data)}")

                tc = data.get("toolchain_calls")
                if not isinstance(tc, list):
                    raise ValueError("return JSON missing 'toolchain_calls' list")
                if len(tc) == 0:
                    raise ValueError("return JSON 'toolchain_calls' is empty, tool")

                # call field
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
                _debug_print(
                    f"[WARN] Base Agent {attempt + 1}/{total_attempts} generate toolchain_calls failed: {e}"
                )
                if attempt < total_attempts - 1 and retry_sleep > 0:
                    import time
                    time.sleep(retry_sleep)

        # retry failed, return error structure result
        return {
            "toolchain_calls": [],
            "error": last_error or 'LLM retry return JSON result',
        }
    
    def query_similar_task(
        self,
        query: str,
        n_results: int = 1,
        verbose: bool = False
    ) -> List[str]:
       
        # similar threshold(Chroma similar;0.3cosine sim0.7)
        DISTANCE_THRESHOLD = 0.30
        
        _debug_print(f"[INFO] query similar task: {query}")
        
        # ===: query, structure ===
        # query task
        enhanced_query = f"Task: {query}\n Main objective: {query}"
        _debug_print(f"[INFO] query: {enhanced_query[:150]}...")
        
        _debug_print('[INFO] generate query...')
        if self._embedding_function is None:
            raise RuntimeError('RAG successfully initialized, _embedding_function is empty,. check RAG initialize log.')
        # use Base Agent embedding_function,
        # use query
        query_embedding = self._embedding_function.embed_query(enhanced_query)
        _debug_print(f"[INFO] query: {len(query_embedding)}")
        
        # use query_embeddings query data (code)
        # include parameter, use default
        results = self._chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        # output return result count
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            _debug_print(f"[INFO] return {len(results['metadatas'][0])} result")
        
        # extract result
        formatted_results: List[str] = []
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            # similar, related result
            try:
                top_distance = results.get('distances', [[None]])[0][0]
            except Exception:
                top_distance = None
            if top_distance is None or top_distance > DISTANCE_THRESHOLD:
                _debug_print(f"[WARN] Top distance={top_distance} threshold {DISTANCE_THRESHOLD}, RAG result")
                return ['Task: Tools Type:']
            
            for i, metadata in enumerate(results['metadatas'][0]):
                task = metadata.get('task', '')
                tools_type = metadata.get('tools_type', '')
                # dag_memory task result(success/failed + error)
                task_result = metadata.get('result', '')
                # steps, extract step,tool_type,requirements_description
                steps = metadata.get('steps', '')
                filtered_steps = []
                
                try:
                    # try steps(JSON list)
                    if isinstance(steps, str):
                        # try JSON
                        try:
                            steps_parsed = json.loads(steps)
                        except (json.JSONDecodeError, TypeError):
                            steps_parsed = []
                    elif isinstance(steps, list):
                        steps_parsed = steps
                    else:
                        steps_parsed = []
                    
                    # extract step field
                    if isinstance(steps_parsed, list):
                        for step_item in steps_parsed:
                            if isinstance(step_item, dict):
                                filtered_step = {
                                    'step': step_item.get('step', ''),
                                    'tool_type': step_item.get('tool_type', ''),
                                    'requirements_description': step_item.get('requirements_description', ''),
                                    # step history task dag_memory.json record result,
                                    # "Successfully." failed, Agent error.
                                    'result': task_result
                                }
                                filtered_steps.append(filtered_step)
                    
                    # format JSON
                    steps_str = json.dumps(filtered_steps, ensure_ascii=False, indent=2) if filtered_steps else "[]"
                except Exception as e:
                    _debug_print(f"[WARN] steps: {e}")
                    steps_str = "[]"

                # similar (Chroma similar)
                score_str = ""
                if results.get('distances') and len(results['distances'][0]) > i:
                    distance = results['distances'][0][i]
                    score = 1.0 / (1.0 + distance)
                    score_str = f"\n Score: {score:.4f} (distance={distance:.4f})"

                result_str = f"Task: {task}\n Tools Type: {tools_type}\n Steps: {steps_str}{score_str}"
                formatted_results.append(result_str)
                
                # output result
                _debug_print(f"\n[SUCCESS] result {i+1}:")
                _debug_print(f"Task: {task}")
                _debug_print(f"Tools Type: {tools_type}")
                if steps_str and steps_str != "[]":
                    _debug_print(f"Steps:\n" + ''.join(steps_str.splitlines()))
                if score_str:
                    _debug_print(f"{score_str.strip()}")
                
                # mode, output
                if verbose:
                    # output content(200)
                    if results.get('documents') and len(results['documents'][0]) > i:
                        doc = results['documents'][0][i]
                        _debug_print(f"Document: {doc[:200]}...")
                    
                    # output (similar)
                    if results.get('distances') and len(results['distances'][0]) > i:
                        distance = results['distances'][0][i]
                        _debug_print(f"Distance: {distance:.4f}")
                
                if len(formatted_results) >= n_results:
                    break
        else:
            _debug_print(f"[WARN] not found task")
        
        # is empty, return
        if not formatted_results:
            formatted_results.append('Task: Tools Type:')
        
        return formatted_results
