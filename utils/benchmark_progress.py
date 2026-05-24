import os
import io
import json
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


def normalize_query(query: str) -> str:
    return (query or "").strip()


def compute_case_id(query: str) -> str:
    """Stable id for a benchmark case; resilient to index changes."""
    q = normalize_query(query)
    return hashlib.sha1(q.encode("utf-8")).hexdigest()


def _load_json_any(path: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


@dataclass
class ProgressStats:
    total_records: int = 0
    total_cases_seen: int = 0
    success_cases: int = 0
    error_cases: int = 0
    skipped_cases: int = 0


class BenchmarkProgress:
    def __init__(
        self,
        results_path: str,
        resume_mode: str = "skip_success",
        run_name: Optional[str] = None,
    ) -> None:
        self.results_path = os.path.abspath(os.path.expanduser(results_path))
        self.resume_mode = resume_mode  # skip_success | skip_done | retry_errors
        self.run_name = run_name or os.path.basename(self.results_path)

        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

        self._format = "jsonl" if self.results_path.endswith('.jsonl') else "json"

        # case_id -> latest_record
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._meta: Dict[str, Any] = {}
        self._duration_seconds_accum: float = 0.0
        self._run_start_dt: Optional[datetime] = None
        self._load_existing()

        # append-only handle (opened lazily)
        self._fh: Optional[io.TextIOWrapper] = None

    def _load_existing(self) -> None:
        if not os.path.exists(self.results_path):
            return

        if self._format == "jsonl":
            for rec in _iter_jsonl(self.results_path):
                cid = rec.get("case_id") or compute_case_id(str(rec.get("query", "")))
                rec["case_id"] = cid
                self._latest[cid] = rec
            return

        # JSON: summary json with {"results":[...]} or list
        data = _load_json_any(self.results_path)
        if data is None:
            return
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            # keep meta for cumulative duration etc
            self._meta = {k: v for k, v in data.items() if k != "results"}
            try:
                self._duration_seconds_accum = float(self._meta.get("duration_seconds", 0.0) or 0.0)
            except Exception:
                self._duration_seconds_accum = 0.0
            for rec in data["results"]:
                if not isinstance(rec, dict):
                    continue
                cid = rec.get("case_id") or compute_case_id(str(rec.get("query", "")))
                rec["case_id"] = cid
                self._latest[cid] = rec
        elif isinstance(data, list):
            for rec in data:
                if not isinstance(rec, dict):
                    continue
                cid = rec.get("case_id") or compute_case_id(str(rec.get("query", "")))
                rec["case_id"] = cid
                self._latest[cid] = rec

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        if self._format == "jsonl":
            self._fh = open(self.results_path, "a", encoding="utf-8")

    def should_skip(self, query: str) -> bool:
        cid = compute_case_id(query)
        if cid not in self._latest:
            return False
        status = str(self._latest[cid].get("status", "")).lower()
        if self.resume_mode == "skip_success_and_error":
            # (success/error/exception status)
            return True
        if self.resume_mode == "skip_done":
            return True
        if self.resume_mode == "retry_errors":
            return status == "success"
        # default: skip_success
        return status == "success"

    def get_latest(self, query: str) -> Optional[Dict[str, Any]]:
        cid = compute_case_id(query)
        return self._latest.get(cid)

    def append(self, record: Dict[str, Any]) -> None:
        query = str(record.get("query", "") or "")
        cid = record.get("case_id") or compute_case_id(query)
        record["case_id"] = cid
        record.setdefault("timestamp", datetime.now().isoformat())
        record.setdefault("_run", self.run_name)

        self._latest[cid] = record
        self._write_snapshot(last_record=record)

    def iter_latest_records(self) -> List[Dict[str, Any]]:
        # deterministic order: by query_index if present, else query string
        recs = list(self._latest.values())
        def _key(r: Dict[str, Any]) -> Tuple[int, str]:
            try:
                idx = int(r.get("query_index", 1_000_000_000))
            except Exception:
                idx = 1_000_000_000
            return (idx, str(r.get("query", "")))
        recs.sort(key=_key)
        return recs

    def stats(self) -> ProgressStats:
        ps = ProgressStats()
        ps.total_cases_seen = len(self._latest)
        for rec in self._latest.values():
            st = str(rec.get("status", "")).lower()
            if st == "success":
                ps.success_cases += 1
            elif st:
                ps.error_cases += 1
        return ps

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def start_run(self, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Call at beginning of a script run to accumulate duration across resumes."""
        self._run_start_dt = datetime.now()
        self._meta.setdefault("first_start_time", self._meta.get("first_start_time") or self._run_start_dt.isoformat())
        self._meta["last_start_time"] = self._run_start_dt.isoformat()
        if extra_meta:
            self._meta.update(extra_meta)
        if self._format == "jsonl":
            self._ensure_open()
            assert self._fh is not None
            self._fh.write(
                json.dumps(
                    {
                        "record_type": "run_start",
                        "timestamp": datetime.now().isoformat(),
                        "status": "meta",
                        "meta": dict(self._meta),
                    },
                    ensure_ascii=False,
                )
                + ''
            )
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except Exception:
                pass
        else:
            self._write_snapshot()

    def finish_run(self, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        end_dt = datetime.now()
        self._meta["last_end_time"] = end_dt.isoformat()
        if self._run_start_dt is not None:
            self._duration_seconds_accum += (end_dt - self._run_start_dt).total_seconds()
        if extra_meta:
            self._meta.update(extra_meta)
        if self._format == "jsonl":
            summary = self._compute_common_summary()
            # avoid dumping huge results array into jsonl; keep key metrics only
            compact = {k: v for k, v in summary.items() if k != "results"}
            compact.update({"record_type": "run_finish", "status": "meta", "meta": dict(self._meta)})
            # write directly as a jsonl line without polluting _latest
            self._ensure_open()
            assert self._fh is not None
            self._fh.write(json.dumps(compact, ensure_ascii=False) + '')
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except Exception:
                pass
        else:
            self._write_snapshot()

    def _compute_common_summary(self) -> Dict[str, Any]:
        records = self.iter_latest_records()
        success_count = 0
        error_count = 0
        expected_sum = 0
        hit_sum = 0
        has_recall = False

        for r in records:
            st = str(r.get("status", "")).lower()
            if st == "success" or bool(r.get("success")):
                success_count += 1
            else:
                error_count += 1
            if "expected_slots" in r or "hit_slots" in r:
                has_recall = True
                try:
                    expected_sum += int(r.get("expected_slots", 0))
                except Exception:
                    pass
                try:
                    hit_sum += int(r.get("hit_slots", 0))
                except Exception:
                    pass

        summary: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(records),
            "success_count": success_count,
            "error_count": error_count,
            "results": records,
        }
        if self._duration_seconds_accum > 0:
            summary["duration_seconds"] = self._duration_seconds_accum
        if has_recall:
            summary["total_expected_slots"] = expected_sum
            summary["total_hit_slots"] = hit_sum
            summary["overall_recall"] = (hit_sum / expected_sum) if expected_sum > 0 else 0.0
        return summary

    def _atomic_write_json(self, data: Dict[str, Any]) -> None:
        tmp_path = self.results_path + '. tmp'
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.results_path)

    def _write_snapshot(self, last_record: Optional[Dict[str, Any]] = None) -> None:
        if self._format == "jsonl":
            if last_record is not None:
                self._ensure_open()
                assert self._fh is not None
                line = json.dumps(last_record, ensure_ascii=False)
                self._fh.write(line + '')
                self._fh.flush()
                try:
                    os.fsync(self._fh.fileno())
                except Exception:
                    pass
            return

        # json summary snapshot (notebook-like, same file keeps growing logically)
        summary = self._compute_common_summary()
        # preserve & merge meta (so caller's extra fields persist)
        merged = dict(self._meta)
        merged.update({k: v for k, v in summary.items() if k != "results"})
        merged["results"] = summary["results"]
        self._atomic_write_json(merged)


__all__ = ["BenchmarkProgress", "compute_case_id", "normalize_query"]


