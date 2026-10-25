#!/usr/bin/env python3
"""알려진 template 조사·양 표현 오류가 최종 bundle에 남았는지 검사한다."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data_natural_multiturn_v3_rebuild"
BAD_PATTERNS = {
    "wrong_topic_particle": re.compile(r"(버섯는|게살는)"),
    "wrong_amount_suffix": re.compile(r"(살짝로|조금로|적게로|보통로|많이로|듬뿍로|기본으로로)"),
    "wrong_subject_particle": re.compile(r"(조금가|보통가|많이가)"),
    "wrong_noun_subject_particle": re.compile(r"(양파이|소시지이|치즈이|페퍼론치노이)"),
    "wrong_copula": re.compile(r"(채소이에요|추가 재료이에요)"),
}
PARTICLE_WORDS = ("양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노", "채소", "육류", "추가 재료")


def has_batchim(word: str) -> bool:
    code = ord(word[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def particle_error(text: str) -> str | None:
    for word in PARTICLE_WORDS:
        wrong = ("는", "를", "가") if has_batchim(word) else ("은", "을", "이")
        for particle in wrong:
            candidate = word + particle
            if candidate in text:
                return candidate
    return None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect_text(row: dict, response: bool) -> list[tuple[str, str]]:
    if response:
        return [
            ("user_text", row["input"].get("user_text", "")),
            ("reply", row["target"].get("reply", "")),
        ]
    texts = [("message", row["input"].get("message", ""))]
    for index, item in enumerate(row["input"].get("recent_history", [])):
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            texts.append((f"recent_history[{index}].content", item["content"]))
    return texts


def audit(rows: list[dict], response: bool) -> list[dict]:
    errors = []
    for row in rows:
        for location, text in collect_text(row, response):
            wrong_particle = particle_error(text)
            if wrong_particle:
                errors.append({
                    "id": row["id"],
                    "location": location,
                    "error_class": "wrong_particle",
                    "match": wrong_particle,
                    "text": text,
                })
            for error_class, pattern in BAD_PATTERNS.items():
                match = pattern.search(text)
                if match:
                    errors.append({
                        "id": row["id"],
                        "location": location,
                        "error_class": error_class,
                        "match": match.group(0),
                        "text": text,
                    })
    return errors


def main() -> None:
    decision_rows = read_jsonl(BASE / "intermediate/decision_v3_intermediate.jsonl")
    response_rows = read_jsonl(BASE / "intermediate/response_v1_intermediate.jsonl")
    decision_errors = audit(decision_rows, response=False)
    response_errors = audit(response_rows, response=True)
    decision_fixed = sum(
        "unnatural_korean" in row.get("metadata", {}).get("cleanup_error_class", [])
        for row in decision_rows
    )
    response_fixed = sum(
        "unnatural_korean" in row.get("metadata", {}).get("cleanup_error_class", [])
        for row in response_rows
    )
    status = "PASS" if not decision_errors and not response_errors else "FAIL"
    report = {
        "status": status,
        "malformed_found": decision_fixed + response_fixed,
        "malformed_fixed": decision_fixed + response_fixed,
        "malformed_remaining": len(decision_errors) + len(response_errors),
        "decision": {"rows": len(decision_rows), "fixed": decision_fixed, "malformed": len(decision_errors), "examples": decision_errors[:20]},
        "response": {"rows": len(response_rows), "fixed": response_fixed, "malformed": len(response_errors), "examples": response_errors[:20]},
    }
    path = BASE / "reports/language_quality.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
