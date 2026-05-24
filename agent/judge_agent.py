import json
import os
import time
from typing import Any, Dict

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

# root directory sys. path, prompt module
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import prompt.judge_agent as PROMPT  # type: ignore

try:
    # config module, getattr read config
    import agent.agent_config as agent_cfg  # type: ignore
    _AGENT_CONFIG_AVAILABLE = True
except ImportError:
    agent_cfg = None  # type: ignore
    _AGENT_CONFIG_AVAILABLE = False


class JudgeAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

        if not _OPENAI_AVAILABLE and not _REQUESTS_AVAILABLE:
            raise ImportError(
                'not installed openai/requests, initialize Judge Agent.'
                "'pip install openai requests' retry."
            )

        if not _AGENT_CONFIG_AVAILABLE:
            raise ImportError(
                'agent_config,/agent/agent_config.py exists available.'
            )

        # agent_config read Judge Agent config
        apikey = getattr(agent_cfg, "judge_agent_apikey", None) or os.getenv(
            "OPENAI_API_KEY", ""
        )
        if not apikey:
            raise ValueError(
                'Judge Agent initialize failed: agent_config.judge_agent_apikey environment variable'
                'OPENAI_API_KEY API Key.'
            )

        base_url = getattr(
            agent_cfg,
            "judge_agent_base_url"
        )

        self.model = getattr(
            agent_cfg,
            "judge_agent_model"
        )

        self.base_url = str(base_url).rstrip("/")
        self.apikey = apikey

        # openai prefer; requests/v1/chat/completions
        self.client = OpenAI(api_key=apikey, base_url=self.base_url) if _OPENAI_AVAILABLE else None
        
        # config read generate parameter
        self.temperature = getattr(agent_cfg, "TEMPERATURE", 0.0)
        self.max_tokens = getattr(agent_cfg, "MAX_TOKENS", 6144)

    def _chat_json_object(self, prompt: str) -> str:
        'return message.content(JSON). - prefer openai SDK - openai unavailable requests'
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

        # requests fallback
        if requests is None:
            raise ImportError('requests not installed, Judge Agent HTTP call')

        url = self.base_url
        # base_url.../v1
        if not url.endswith("/v1"):
            url = url + "/v1"
        url = url + "/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.apikey}",
        }
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

        r = requests.post(url, headers=headers, json=payload, timeout=600)
        # service unsupported response_format, try request
        if r.status_code >= 400:
            payload.pop("response_format", None)
            r = requests.post(url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""

    def judge(
        self,
        query: str,
        agent_execution: str,
        true_answer: str,
        max_retries: int = 3,
        retry_sleep: float = 1.0,
    ) -> Dict[str, Any]:
    
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query empty')
        if not isinstance(agent_execution, str):
            agent_execution = str(agent_execution) if agent_execution else ""
        if not isinstance(true_answer, str):
            true_answer = str(true_answer) if true_answer else ""

        # format prompt
        prompt = PROMPT.Judge_notool.format(
            context=PROMPT.context,
            query=query,
            agent_execution=agent_execution,
            true_answer=true_answer
        )

        total_attempts = max(1, max_retries)
        last_error: Any = None

        for attempt in range(total_attempts):
            try:
                content = self._chat_json_object(prompt)
                if not content:
                    raise ValueError("LLM return contentis empty")

                # JSON response
                data = json.loads(content)
                
                # extract result
                result = data.get("result", "").upper().strip()
                if result not in ["PASS", "FAIL"]:
                    # result format, try field extract
                    if "judgment" in data:
                        result = str(data["judgment"]).upper().strip()
                    elif "verdict" in data:
                        result = str(data["verdict"]).upper().strip()
                    else:
                        raise ValueError(f"response extract result: {content[:200]}")
                
                # return result
                response = {
                    "result": result if result in ["PASS", "FAIL"] else "FAIL"
                }
                
                # optional field: confidence
                if "confidence" in data:
                    response["confidence"] = float(data["confidence"])
                if "reasoning" in data:
                    response["reasoning"] = str(data["reasoning"])
                elif "explanation" in data:
                    response["reasoning"] = str(data["explanation"])
                
                return response

            except json.JSONDecodeError as e:
                last_error = f"JSON failed: {e}"
                print(f"[WARN] Judge Agent {attempt + 1}/{total_attempts} failed: {last_error}")
                if attempt < total_attempts - 1 and retry_sleep > 0:
                    time.sleep(retry_sleep)
            except Exception as e:
                last_error = str(e)
                print(f"[WARN] Judge Agent {attempt + 1}/{total_attempts} failed: {e}")
                if attempt < total_attempts - 1 and retry_sleep > 0:
                    time.sleep(retry_sleep)

        # retry failed, return default failed result
        return {
            "result": "FAIL",
            "error": last_error or 'LLM retry return JSON result'
        }


__all__ = ["JudgeAgent"]


