import os
import sys
import json
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

import requests


def _print(*args, **kwargs):
    'print to stderr, avoid interfering with RPC return'
    print(*args, file=sys.stderr, **kwargs)


DEFAULT_ENDPOINT = 'https://mineru.net/api/pdf2json'


def _ensure_dir(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def run(args: SimpleNamespace):
    'PDF -> JSON. parameter(Simple Namespace): - pdf_path: local PDF file path() - output_path: output JSON path, default pdf directory same-name.json - api_key: MinerU API Key, optional, default reads environment variable MINERU_API_KEY - endpoint: MinerU API endpoint, optional, default https://mineru.net/api/pdf2json'
    pdf_path = Path(os.path.expanduser(getattr(args, "pdf_path", "")))
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    output_path = getattr(args, "output_path", None)
    if not output_path:
        output_path = pdf_path.with_suffix('.json')
    output_path = Path(os.path.expanduser(output_path))

    endpoint = getattr(args, "endpoint", DEFAULT_ENDPOINT)
    api_key: Optional[str] = getattr(args, "api_key", None) or os.getenv("MINERU_API_KEY")

    if not api_key:
        raise ValueError('missing api_key: parameter api_key environment variable MINERU_API_KEY')

    headers = {"Authorization": f"Bearer {api_key}"}

    _print(f"call MinerU: {endpoint}")
    _print(f"input: {pdf_path}")
    _print(f"output: {output_path}")

    with pdf_path.open("rb") as f:
        files = {"file": ('document. pdf', f, "application/pdf")}
        resp = requests.post(endpoint, headers=headers, files=files, timeout=300)

    try:
        resp.raise_for_status()
    except Exception:
        _print(f"request failed: status={resp.status_code}, ={resp.text}")
        raise

    try:
        data = resp.json()
    except Exception:
        _print('response JSON, original:')
        _print(resp.text[:500])
        raise ValueError('MinerU response JSON')

    _ensure_dir(output_path)
    with output_path.open("w", encoding="utf-8") as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=2)

    _print('OK conversion complete')
    return {
        "status": "success",
        "endpoint": endpoint,
        "output_path": str(output_path),
        "bytes_written": output_path.stat().st_size if output_path.exists() else 0,
    }

