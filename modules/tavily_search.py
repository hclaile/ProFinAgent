'Tavily search tool module use Tavily API search, return structure answer fragment.'
import os
import sys
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Any

# optional dependency
try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    TavilyClient = None
    _TAVILY_AVAILABLE = False

def _print(*args, **kwargs):
    'print to stderr, avoid interfering with MCP JSON-RPC communication'
    print(*args, file=sys.stderr, **kwargs)


def _ensure_output_path(path: str) -> str:
    'user output path, exists is empty'
    if not path or not isinstance(path, str):
        raise ValueError('output_path, empty')
    path = os.path.expanduser(path)
    out_dir = os.path.dirname(path)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    return path


def _get_root_dir(args) -> str:
    """get result save root directory"""
    root = getattr(args, "root", None)
    if root:
        return os.path.expanduser(root)
    return os.getcwd()


class TavilySearcher:
    'Tavily search'
    
    def __init__(self, api_key: Optional[str] = None, root: Optional[str] = None):
        'initialize Tavily search Args: api_key: Tavily API Key, None read TAVILY_API_KEY root: result save root directory'
        if not _TAVILY_AVAILABLE:
            raise ImportError('tavily not installed, please run: pip install tavily-python')
        
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError('missing Tavily api_key, api_key TAVILY_API_KEY')
        self.root = root or os.getcwd()
        self.client = TavilyClient(api_key=self.api_key)
        self.last_search_data = None
    
    def search(
        self,
        query: str,
        search_depth: str = "advanced",
        include_answer: bool = True,
        max_results: int = 5,
        include_domains: Optional[list] = None,
        exclude_domains: Optional[list] = None,
        include_raw_content: bool = False,
    ) -> Dict[str, Any]:
        'Tavily search Args: query: search query search_depth: search,"basic" "advanced" include_answer: direct answer max_results: maximum result count include_domains: list(optional) exclude_domains: list(optional) include_raw_content: original content(optional) Returns: answer results dictionary'
        try:
            _print(f"[INFO] Tavily search: {query}")
            _print(f"[INFO] search: {search_depth}, maximum result: {max_results}")
            
            # search parameter
            search_params = {
                "query": query,
                "search_depth": search_depth,
                "include_answer": include_answer,
                "max_results": max_results,
            }
            
            # optional parameters
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            if include_raw_content:
                search_params["include_raw_content"] = include_raw_content
            
            # search
            response = self.client.search(**search_params)
            
            # response
            result = {
                "query": query,
                "answer": response.get("answer", ""),
                "results": response.get("results", []),
                "search_depth": search_depth,
                "max_results": max_results,
            }
            
            # check result
            has_answer = bool(result.get("answer", "").strip())
            has_results = bool(result.get("results")) and len(result.get("results", [])) > 0
            
            if not has_answer and not has_results:
                _print('[WARN] Tavily searchreturn empty result')
                return {"error": "Empty result", "query": query}
            
            _print(f"[SUCCESS] Tavily search complete")
            if has_answer:
                _print(f"[INFO] direct answer: {result['answer'][:100]}...")
            if has_results:
                _print(f"[INFO] return {len(result['results'])} result")
            
            self.last_search_data = result
            return result
            
        except Exception as e:
            _print(f"[ERROR] Tavily search failed: {str(e)}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {"error": str(e), "query": query}
    
    def save_to_json(self, output_path: str):
        'save search result JSON file'
        if not self.last_search_data:
            _print('[WARN] save search result')
            return False
        
        try:
            output_path = _ensure_output_path(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.last_search_data, f, ensure_ascii=False, indent=2)
            _print(f"[SUCCESS] search result save: {output_path}")
            return True
        except Exception as e:
            _print(f"[ERROR] save failed: {str(e)}")
            return False


def run_tavily_search(args) -> None:
    'Tavily search tool required parameters: query: search query output_path: output file path(JSON format) optional parameters: api_key: Tavily API Key(read TAVILY_API_KEY) search_depth: search,"basic" "advanced"(default: "advanced") include_answer: direct answer(default: True) max_results: maximum result count(default: 5) include_domains: list(optional) exclude_domains: list(optional) include_raw_content: original content(default: False) root: result save root directory(optional)'
    # get parameter
    root = _get_root_dir(args)
    query = getattr(args, "query", None)
    output_path = getattr(args, "output_path", None)
    api_key = getattr(args, "api_key", None)
    search_depth = getattr(args, "search_depth", "advanced")
    include_answer = getattr(args, "include_answer", True)
    max_results = getattr(args, "max_results", 5)
    include_domains = getattr(args, "include_domains", None)
    exclude_domains = getattr(args, "exclude_domains", None)
    include_raw_content = getattr(args, "include_raw_content", False)
    
    # parameter
    if not query:
        raise ValueError('Tavily search query parameter')
    if not output_path:
        raise ValueError('Tavily search output_path parameter')
    
    output_path = _ensure_output_path(output_path)
    
    searcher = TavilySearcher(api_key=api_key, root=root)
    result = searcher.search(
        query=query,
        search_depth=search_depth,
        include_answer=include_answer,
        max_results=max_results,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        include_raw_content=include_raw_content,
    )
    if result.get("error") or (not result.get("answer") and not result.get("results")):
        raise RuntimeError(f"Tavily search failed return empty result: {result.get('error', 'empty result')}")

    searcher.save_to_json(output_path)
    _print(f"[SUCCESS] Tavily search complete, result save: {output_path}")


def run(args) -> None:
    '(compatible MCP call)'
    run_tavily_search(args)
