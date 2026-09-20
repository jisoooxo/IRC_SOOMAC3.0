import argparse
import copy
import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

from soomac_irc.agent_v7 import (
    build_graph,
    classify_confirmation_intent,
    commit_section_skip,
)
from soomac_irc.order_v7 import (
    SECTION_LABELS,
    SECTION_ORDER,
    build_section_plan,
    new_order,
    next_section,
    strongest_restriction_for,
)
from soomac_irc.reply_v7 import REPLY_GUARD_VERSION, build_turn_reply


DEFAULT_LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "soomac_runtime_logs"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _git_metadata() -> dict:
    def run_git(*args) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run_git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "tracked_files_dirty": bool(status) if status is not None else None,
    }


def _library_versions() -> dict:
    versions = {}

    for package in ("torch", "transformers", "peft", "xgrammar", "langgraph"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    return versions


def _state_diff(before, after, path="") -> list[dict]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []

        for key in sorted(set(before) | set(after)):
            child_path = f"{path}.{key}" if path else key

            if key not in before:
                changes.append({"path": child_path, "before": None, "after": after[key]})
            elif key not in after:
                changes.append({"path": child_path, "before": before[key], "after": None})
            else:
                changes.extend(_state_diff(before[key], after[key], child_path))

        return changes

    if before != after:
        return [{"path": path, "before": before, "after": after}]

    return []


def _decision_state(state: dict) -> dict:
    return {
        "order": copy.deepcopy(state.get("order")),
        "execution": copy.deepcopy(state.get("execution")),
        "recommendation": copy.deepcopy(state.get("recommendation")),
        "control": copy.deepcopy(state.get("control")),
    }


class DebugSession:
    """ROS·카메라 없이 V7 Tool, 정책, Reply, 주문 흐름을 실행한다."""

    def __init__(
        self,
        graph,
        call_model,
        generate_reply,
        *,
        manual_flow=False,
        log_path=None,
        runtime_metadata=None,
    ):
        self.graph = graph
        self.call_model = call_model
        self.generate_reply = generate_reply
        self.manual_flow = manual_flow
        self.log_path = Path(log_path) if log_path else None
        self.runtime_metadata = copy.deepcopy(runtime_metadata or {})
        self.log_errors = []
        self.session_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
        self.turn_index = 0
        self.last_record_id = None
        self.reset()

    def reset(self):
        self.order = new_order()
        self.section = "noodle"
        self.selected = []
        self.task_queue = []
        self.active_task = None
        self.completed_tasks = []
        self.skipped_sections = {}
        self.robot_started = False
        self.recommendation_state = {"phase": "idle", "confirmed_sections": []}
        self.awaiting_confirm = None
        self.order_finished = False
        self.history = []
        self.action_history = []

    def state(self, user_text="") -> dict:
        return {
            "user_text": user_text,
            "order": copy.deepcopy(self.order),
            "execution": {
                "section": self.section,
                "selected": copy.deepcopy(self.selected),
                "task_queue": copy.deepcopy(self.task_queue),
                "active_task": copy.deepcopy(self.active_task),
                "completed_tasks": copy.deepcopy(self.completed_tasks),
                "skipped_sections": copy.deepcopy(self.skipped_sections),
                "robot_started": self.robot_started,
            },
            "recommendation": copy.deepcopy(self.recommendation_state),
            "control": {
                "awaiting_confirm": copy.deepcopy(self.awaiting_confirm),
                "order_finished": self.order_finished,
            },
            "history": copy.deepcopy(self.history),
            "action_history": copy.deepcopy(self.action_history),
        }

    def _validate_result(self, result: dict):
        if not isinstance(result, dict):
            raise ValueError("agent 결과가 dict가 아님")

        for key in ("order", "execution", "recommendation", "transaction"):
            if not isinstance(result.get(key), dict):
                raise ValueError(f"agent 결과 {key}가 dict가 아님")

        execution = result["execution"]

        if execution.get("section") not in SECTION_ORDER:
            raise ValueError(f"agent 결과 section 이상함 : {execution.get('section')}")

        for key in ("selected", "task_queue", "completed_tasks"):
            if not isinstance(execution.get(key), list):
                raise ValueError(f"agent 결과 execution.{key}가 list가 아님")

        if execution.get("active_task") is not None and not isinstance(execution["active_task"], dict):
            raise ValueError("agent 결과 active_task가 dict 또는 None이 아님")

        if not isinstance(execution.get("skipped_sections"), dict):
            raise ValueError("agent 결과 skipped_sections가 dict가 아님")

        if type(execution.get("robot_started")) is not bool:
            raise ValueError("agent 결과 robot_started가 bool이 아님")

    def _commit(self, result: dict):
        self._validate_result(result)
        execution = result["execution"]
        self.order = copy.deepcopy(result["order"])
        self.section = execution["section"]
        self.selected = copy.deepcopy(execution["selected"])
        self.task_queue = copy.deepcopy(execution["task_queue"])
        self.active_task = copy.deepcopy(execution["active_task"])
        self.completed_tasks = copy.deepcopy(execution["completed_tasks"])
        self.skipped_sections = copy.deepcopy(execution["skipped_sections"])
        self.robot_started = execution["robot_started"]
        self.recommendation_state = copy.deepcopy(result["recommendation"])

    def _section_prompt(self) -> str:
        prompts = {
            "veggie": "다음은 야채를 고르실 차례입니다. 양파와 버섯 중 어떤 재료를 어느 정도 양으로 드시고 싶은지 말씀해 주세요.",
            "meat": "다음은 육류를 고르실 차례입니다. 소시지와 게살 중 어떤 재료를 어느 정도 양으로 드시고 싶은지 말씀해 주세요.",
            "extra": "다음은 추가 재료를 고르실 차례입니다. 치즈와 페퍼론치노 중 어떤 재료를 어느 정도 양으로 드시고 싶은지 말씀해 주세요.",
        }

        if self.section not in prompts:
            raise ValueError(f"사용자 선택 안내가 없는 section임 : {self.section}")

        return prompts[self.section]

    def _promote_next_task(self, flow_events: list[dict]) -> bool:
        if self.active_task is not None or not self.task_queue:
            return False

        self.active_task = copy.deepcopy(self.task_queue.pop(0))
        self.robot_started = True
        flow_events.append({"event": "task_started", "task": copy.deepcopy(self.active_task)})
        return True

    def _queue_current_section(self, flow_events: list[dict]) -> bool:
        if self.active_task is not None or self.task_queue:
            return False

        tasks = build_section_plan(self.order, self.section)

        if not tasks:
            return False

        queued_keys = []

        for task in tasks:
            task_class = task["class"]

            if task_class in ("얇은면", "넓은면"):
                queued_keys.extend(["noodle_type", "noodle_portion"])
            elif task_class in ("양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노"):
                queued_keys.append(f"toppings.{task_class}")
            elif task_class in ("오일", "토마토", "크림"):
                queued_keys.append("sauce")

        self.selected = [key for key in self.selected if key not in queued_keys]
        self.task_queue.extend(copy.deepcopy(tasks))
        return self._promote_next_task(flow_events)

    def _finish_order(self, flow_events: list[dict]) -> str:
        self.awaiting_confirm = None
        self.order_finished = True
        flow_events.append({"event": "order_finished"})
        return "소스까지 모두 담았어요. 이용해 주셔서 감사합니다."

    def _clear_model_traces(self):
        if hasattr(self.call_model, "last_trace"):
            self.call_model.last_trace = None

        if hasattr(self.generate_reply, "last_trace"):
            self.generate_reply.last_trace = None

    def _advance_after_section(self, completed_section: str, flow_events: list[dict]) -> str:
        section_label = SECTION_LABELS[completed_section]

        if completed_section in self.skipped_sections:
            completed_reply = f"{section_label}는 제외했어요."
        elif completed_section == "lid":
            completed_reply = "뚜껑을 닫았어요."
        else:
            completed_reply = f"{section_label} 담기가 끝났어요."

        next_step = next_section(completed_section)

        while next_step in self.skipped_sections:
            next_step = next_section(next_step)

        if next_step is None:
            raise ValueError(f"{completed_section} 다음 section이 없음")

        self.section = next_step
        flow_events.append({"event": "section_advanced", "from": completed_section, "to": next_step})

        if self.section == "lid":
            if not self._queue_current_section(flow_events):
                raise RuntimeError("lid 작업을 만들지 못함")

            return f"{completed_reply} 이제 뚜껑을 닫을게요."

        if self.section == "sauce":
            if not self._queue_current_section(flow_events):
                raise RuntimeError("선택된 소스 작업을 만들지 못함")

            return f"{completed_reply} 마지막으로 고르신 소스를 올릴게요."

        confirmed_sections = self.recommendation_state.get("confirmed_sections", [])

        if self.section in confirmed_sections and self._queue_current_section(flow_events):
            return f"{completed_reply} 추천한 {SECTION_LABELS[self.section]} 재료 담기를 시작할게요."

        return f"{completed_reply} {self._section_prompt()}"

    def _complete_one_task(self, flow_events: list[dict]) -> str:
        if self.active_task is None:
            return "진행 중인 가상 작업이 없어요."

        completed = copy.deepcopy(self.active_task)
        self.active_task = None

        if completed not in self.completed_tasks:
            self.completed_tasks.append(completed)

        flow_events.append({"event": "task_completed", "task": completed})

        if self.task_queue:
            self._promote_next_task(flow_events)
            return f"{completed['class']} 작업을 완료하고 다음 작업을 시작했어요."

        if self.section == "sauce":
            return self._finish_order(flow_events)

        return self._advance_after_section(self.section, flow_events)

    def _drain_auto(self, flow_events: list[dict]) -> list[str]:
        replies = []

        while self.active_task is not None and not self.order_finished:
            replies.append(self._complete_one_task(flow_events))

        return replies

    def next_task(self) -> dict:
        self._clear_model_traces()
        before = self.state()
        flow_events = []
        started_at = time.perf_counter()

        if self.manual_flow and self.section == "sauce" and self.active_task is not None:
            reply = "생산 V7과 같은 계약으로 소스 완료는 /finish를 입력하세요."
            source = "manual_flow_wrong_signal"
        else:
            reply = self._complete_one_task(flow_events)
            source = "manual_flow"

        self.history.append({"role": "assistant", "content": reply})
        return self._make_record(
            user_text="/next",
            state_before=before,
            result=None,
            final_reply=reply,
            flow_events=flow_events,
            total_ms=(time.perf_counter() - started_at) * 1000,
            source=source,
        )

    def finish_task(self) -> dict:
        self._clear_model_traces()
        before = self.state()
        flow_events = []
        started_at = time.perf_counter()

        if not self.manual_flow:
            reply = "auto 모드에서는 소스 작업도 자동으로 완료됩니다."
            source = "finish_not_required"
        elif self.section != "sauce" or self.active_task is None:
            reply = "완료할 소스 가상 작업이 없어요."
            source = "finish_without_sauce"
        else:
            reply = self._complete_one_task(flow_events)
            source = "manual_finish"

        self.history.append({"role": "assistant", "content": reply})
        return self._make_record(
            user_text="/finish",
            state_before=before,
            result=None,
            final_reply=reply,
            flow_events=flow_events,
            total_ms=(time.perf_counter() - started_at) * 1000,
            source=source,
        )

    def _handle_pending_confirmation(self, user_text: str) -> dict:
        self._clear_model_traces()
        before = self.state(user_text)
        flow_events = []
        started_at = time.perf_counter()
        intent = classify_confirmation_intent(user_text, "section_skip")
        self.history.append({"role": "user", "content": user_text})

        if intent is None:
            reply = "네 또는 아니요로 말씀해 주세요."
        else:
            self.turn_index += 1
            pending = copy.deepcopy(self.awaiting_confirm)
            self.awaiting_confirm = None
            action = f"{pending['type']}_{'accept' if intent else 'reject'}"
            self.action_history.append({"turn": self.turn_index, "action": action})
            self.action_history = self.action_history[-12:]

            if not intent:
                if pending.get("source") == "all_options_restricted":
                    dislikes = [
                        item
                        for item in pending["items"]
                        if (strongest_restriction_for(self.order, item) or {}).get("reason") == "dislike"
                    ]
                    reply = f"{SECTION_LABELS[self.section]} 단계를 제외하지 않을게요. {'나 '.join(dislikes)}은 취향 제한이라 원하면 재료와 양을 다시 말씀해 주세요."
                else:
                    reply = f"{SECTION_LABELS[self.section]} 단계에서 계속 고를게요. {self._section_prompt()}"
            else:
                committed = commit_section_skip(
                    self.order,
                    self.state()["execution"],
                    self.recommendation_state,
                    pending,
                )
                self._commit({
                    "order": committed["order"],
                    "execution": committed["execution"],
                    "recommendation": committed["recommendation"],
                    "transaction": {},
                })
                reply = self._advance_after_section(pending["section"], flow_events)

                if not self.manual_flow:
                    auto_replies = self._drain_auto(flow_events)
                    if auto_replies:
                        reply = " ".join([reply, *auto_replies])

        self.history.append({"role": "assistant", "content": reply})
        return self._make_record(
            user_text=user_text,
            state_before=before,
            result=None,
            final_reply=reply,
            flow_events=flow_events,
            total_ms=(time.perf_counter() - started_at) * 1000,
            source="pending_confirmation",
        )

    def turn(self, user_text: str) -> dict:
        self._clear_model_traces()
        state_before = self.state(user_text if isinstance(user_text, str) else "")
        started_at = time.perf_counter()

        try:
            return self._turn_impl(user_text)
        except Exception as error:
            self._make_record(
                user_text=user_text,
                state_before=state_before,
                result=None,
                final_reply=None,
                flow_events=[],
                total_ms=(time.perf_counter() - started_at) * 1000,
                source="turn_error",
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise

    def _turn_impl(self, user_text: str) -> dict:
        user_text = user_text.strip()

        if not user_text:
            raise ValueError("빈 사용자 발화는 처리할 수 없음")

        if self.awaiting_confirm is not None:
            return self._handle_pending_confirmation(user_text)

        if self.order_finished:
            raise RuntimeError("주문이 끝났습니다. /reset 후 다시 테스트하세요.")

        if self.manual_flow and (self.active_task is not None or self.task_queue):
            before = self.state(user_text)
            blocked_started_at = time.perf_counter()
            completion_command = "/finish" if self.section == "sauce" else "/next"
            reply = f"가상 로봇 작업 중입니다. {completion_command}로 완료 신호를 보내세요."
            return self._make_record(
                user_text=user_text,
                state_before=before,
                result=None,
                final_reply=reply,
                flow_events=[],
                total_ms=(time.perf_counter() - blocked_started_at) * 1000,
                source="blocked_during_manual_flow",
            )

        state_before = self.state(user_text)
        started_at = time.perf_counter()
        warnings = []
        flow_events = []

        self._clear_model_traces()

        result = self.graph.invoke(state_before)
        self._validate_result(result)
        reply = build_turn_reply(
            user_text,
            state_before,
            result,
            self.generate_reply,
            warning=warnings.append,
        )

        self._commit(result)
        self.turn_index += 1
        self.history.append({"role": "user", "content": user_text})
        transaction = result["transaction"]
        action = transaction["action"]
        action_event = {"turn": self.turn_index, "action": action}

        if transaction["accepted"]:
            action_event["accepted"] = copy.deepcopy(transaction["accepted"])

        if transaction["restriction_applied"]:
            action_event["restriction_applied"] = copy.deepcopy(transaction["restriction_applied"])

        if action == "recommend_order":
            action_event["scope"] = (result.get("tool_call") or {}).get("changes", {}).get("scope")

        if action == "confirm_section":
            action_event["section"] = self.section

        self.action_history.append(action_event)
        self.action_history = self.action_history[-12:]
        section_skip = transaction["section_skip"]

        if section_skip is not None and section_skip["needs_confirmation"]:
            self.awaiting_confirm = copy.deepcopy(section_skip)
            self.awaiting_confirm["type"] = "refuse_section"

        if section_skip is not None and section_skip["applied"]:
            reply = self._advance_after_section(section_skip["section"], flow_events)

        confirm = transaction["confirm_validation"]
        recommendation = transaction["recommendation_validation"]
        recommendation_confirmed = (
            recommendation["proposal_confirmed"]
            and self.section in self.recommendation_state.get("confirmed_sections", [])
        )

        if (confirm["requested"] and confirm["allowed"]) or recommendation_confirmed:
            if self._queue_current_section(flow_events):
                start_reply = f"{SECTION_LABELS[self.section]} 담기를 시작할게요."
                reply = f"{reply} {start_reply}" if reply else start_reply
            elif self.section == "extra":
                self.skipped_sections["extra"] = "empty_confirm"
                reply = self._advance_after_section("extra", flow_events)
            else:
                raise RuntimeError(f"{self.section} 확정 후 실행할 작업을 만들지 못함")

        if not self.manual_flow:
            auto_replies = self._drain_auto(flow_events)
            if auto_replies:
                reply = " ".join(part for part in [reply, *auto_replies] if part)

        if reply:
            self.history.append({"role": "assistant", "content": reply})

        return self._make_record(
            user_text=user_text,
            state_before=state_before,
            result=result,
            final_reply=reply,
            flow_events=flow_events,
            total_ms=(time.perf_counter() - started_at) * 1000,
            source="model_turn",
            warnings=warnings,
        )

    def _make_record(
        self,
        *,
        user_text,
        state_before,
        result,
        final_reply,
        flow_events,
        total_ms,
        source,
        warnings=None,
        error=None,
    ) -> dict:
        state_after = self.state()
        record_id = f"{self.session_id}_{self.turn_index:04d}_{uuid.uuid4().hex[:6]}"
        self.last_record_id = record_id
        transaction = result.get("transaction") if result else None
        risk_flags = []

        if transaction:
            confirm = transaction.get("confirm_validation", {})

            if confirm.get("requested") and not confirm.get("allowed"):
                risk_flags.append(f"confirmation_blocked:{confirm.get('reason')}")

            if transaction.get("action") == "respond" and getattr(self.generate_reply, "last_trace", None) is not None:
                risk_flags.append("free_reply_generated")

        if any("자유응답의 주문 흐름" in message for message in (warnings or [])):
            risk_flags.append("free_reply_sentence_removed")

        if any("respond Tool이 주문 동작 표현" in message for message in (warnings or [])):
            risk_flags.append("respond_on_operational_text")

        if error is not None:
            risk_flags.append("turn_error")

        record = {
            "schema_version": 1,
            "record_type": "llm_debug_turn",
            "timestamp": datetime.now().astimezone().isoformat(),
            "session_id": self.session_id,
            "record_id": record_id,
            "turn_index": self.turn_index,
            "flow_mode": "manual" if self.manual_flow else "auto",
            "runtime": copy.deepcopy(self.runtime_metadata),
            "source": source,
            "user_text": user_text,
            "tool_trace": copy.deepcopy(getattr(self.call_model, "last_trace", None)) if source in ("model_turn", "turn_error") else None,
            "tool_call": copy.deepcopy(result.get("tool_call")) if result else None,
            "parser_status": result.get("parser_status") if result else None,
            "transaction": copy.deepcopy(transaction),
            "policy_reply": result.get("policy_reply") if result else None,
            "reply_trace": copy.deepcopy(getattr(self.generate_reply, "last_trace", None)) if source in ("model_turn", "turn_error") else None,
            "final_reply": final_reply,
            "warnings": copy.deepcopy(warnings or []),
            "error": copy.deepcopy(error),
            "risk_flags": risk_flags,
            "flow_events": copy.deepcopy(flow_events),
            "state_before": copy.deepcopy(state_before),
            "state_after": state_after,
            "state_diff": _state_diff(
                _decision_state(state_before),
                _decision_state(state_after),
            ),
            "timing": {"total_ms": round(total_ms, 3)},
        }
        self._append_log(record)
        return record

    def mark(self, text: str) -> dict:
        if self.last_record_id is None:
            raise RuntimeError("표시할 이전 턴이 없습니다.")

        parts = shlex.split(text)
        verdict = parts[0].lower() if parts else ""

        if verdict not in ("ok", "bad"):
            raise ValueError("/mark 뒤에는 ok 또는 bad가 필요합니다.")

        fields = {}
        notes = []

        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
            else:
                notes.append(part)

        record = {
            "schema_version": 1,
            "record_type": "llm_debug_label",
            "timestamp": datetime.now().astimezone().isoformat(),
            "session_id": self.session_id,
            "target_record_id": self.last_record_id,
            "verdict": verdict,
            "expected_tool": fields.get("expected_tool"),
            "expected_reply": fields.get("expected_reply"),
            "note": " ".join(notes) or fields.get("note"),
        }
        self._append_log(record)
        return record

    def _append_log(self, record: dict):
        if self.log_path is None:
            return

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as error:
            # 관찰 로그 실패가 이미 검증된 대화 상태와 최종 응답을 깨뜨리면 안 된다.
            self.log_errors.append({
                "type": type(error).__name__,
                "message": str(error),
                "record_id": record.get("record_id"),
            })


def _print_record(record: dict):
    print(f"\nSOURCE> {record['source']} / {record['flow_mode']}")
    print(f"RUNTIME> {_json(record.get('runtime'))}")
    tool_trace = record.get("tool_trace") or {}
    reply_trace = record.get("reply_trace") or {}
    print(f"RAW_TOOL> {tool_trace.get('raw_output')}")
    print(f"TOOL> {_json(record.get('tool_call'))}")

    if record.get("transaction") is not None:
        transaction = record["transaction"]
        summary = {
            key: transaction.get(key)
            for key in (
                "action",
                "accepted",
                "invalid_value",
                "blocked_active_or_completed",
                "blocked_past_section",
                "blocked_constraint",
                "kept_physical_conflict",
                "restriction_applied",
                "section_skip",
                "confirm_validation",
                "recommendation_validation",
            )
        }
        print(f"TRANSACTION> {_json(summary)}")

    print(f"POLICY> {record.get('policy_reply')}")
    print(f"RAW_REPLY> {reply_trace.get('raw_output')}")
    print(f"FINAL> {record.get('final_reply')}")
    print(f"STATE_DIFF> {_json(record.get('state_diff'))}")
    print(f"FLOW> {_json(record.get('flow_events'))}")
    timing = {
        "tool_ms": tool_trace.get("latency_ms"),
        "reply_ms": reply_trace.get("latency_ms"),
        "total_ms": record.get("timing", {}).get("total_ms"),
    }
    print(f"TIME> {_json(timing)}")
    print(f"RISK> {_json(record.get('risk_flags'))}")

    if record.get("error") is not None:
        print(f"ERROR> {_json(record['error'])}")


def _default_log_path() -> Path:
    file_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return DEFAULT_LOG_DIRECTORY / f"llm_debug_{file_id}.jsonl"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROS·카메라 없이 SOOMAC V7 LLM 대화를 검사합니다."
    )
    parser.add_argument(
        "--manual-flow",
        action="store_true",
        help="가상 작업을 /next까지 유지하며, 마지막 소스만 /finish로 완료합니다.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Tool LoRA 없이 base model만 사용합니다.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Tool LoRA 경로입니다.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="디버그 JSONL 저장 경로입니다. 기본값은 soomac_runtime_logs입니다.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="JSONL 로그를 저장하지 않습니다.",
    )
    parser.add_argument(
        "--once",
        action="append",
        default=[],
        help="대화형 입력 대신 한 문장을 실행합니다. 여러 번 지정할 수 있습니다.",
    )
    return parser


def main(argv=None):
    # 순수 DebugSession 단위 테스트에서는 GPU·Transformers import가 필요하지 않다.
    from soomac_irc.call_model_v7 import (
        DEFAULT_TOOL_ADAPTER_PATH,
        FREE_REPLY_SYSTEM,
        FREE_REPLY_PROMPT_VERSION,
        MODEL_PATH,
        MODEL_QUANTIZATION,
        TOOL_PROMPT_VERSION,
        TOOL_SYSTEM,
        load_model,
        make_call_model,
        make_generate_reply,
    )

    args = _build_parser().parse_args(argv)
    adapter_path = None if args.base_only else (args.adapter_path or DEFAULT_TOOL_ADAPTER_PATH)
    log_path = None if args.no_log else (args.log_path or _default_log_path())
    model_path = Path(MODEL_PATH)
    adapter_config_path = Path(adapter_path) / "adapter_config.json" if adapter_path else None
    runtime_metadata = {
        "model_path": MODEL_PATH,
        "model_config_sha256": _sha256_file(model_path / "config.json"),
        "quantization": MODEL_QUANTIZATION,
        "tool_adapter_path": adapter_path,
        "tool_adapter_name": Path(adapter_path).name if adapter_path else "base",
        "tool_adapter_config_sha256": (
            _sha256_file(adapter_config_path) if adapter_config_path else None
        ),
        "tool_prompt_version": TOOL_PROMPT_VERSION,
        "tool_prompt_sha256": _sha256_text(TOOL_SYSTEM),
        "free_reply_prompt_version": FREE_REPLY_PROMPT_VERSION,
        "free_reply_prompt_sha256": _sha256_text(FREE_REPLY_SYSTEM),
        "reply_guard_version": REPLY_GUARD_VERSION,
        "git": _git_metadata(),
        "libraries": _library_versions(),
    }
    load_started = time.perf_counter()
    print(f"MODEL> loading / tool_adapter={adapter_path or 'base'}")
    model, processor = load_model(adapter_path)
    call_model = make_call_model(model, processor)
    generate_reply = make_generate_reply(model, processor)
    graph = build_graph(call_model)
    session = DebugSession(
        graph,
        call_model,
        generate_reply,
        manual_flow=args.manual_flow,
        log_path=log_path,
        runtime_metadata=runtime_metadata,
    )
    print(f"MODEL> ready / {round(time.perf_counter() - load_started, 2)}s")
    print(f"MODE> {'manual' if args.manual_flow else 'auto'}")
    print(f"LOG> {log_path or 'off'}")
    print("COMMANDS> /state /history /reset /next /finish /mark ok|bad [expected_tool=...] [note=...] /quit")

    def run_text(text: str):
        if text == "/state":
            print(_json(session.state()))
            return

        if text == "/history":
            print(_json(session.history))
            return

        if text == "/reset":
            session.reset()
            print("RESET> 주문·대화 상태를 초기화했습니다.")
            return

        if text == "/next":
            _print_record(session.next_task())
            return

        if text == "/finish":
            _print_record(session.finish_task())
            return

        if text.startswith("/mark "):
            print(f"LABEL> {_json(session.mark(text[len('/mark '):]))}")
            return

        _print_record(session.turn(text))

    if args.once:
        for text in args.once:
            run_text(text)
        return

    while True:
        try:
            text = input("\nUSER> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text in ("/quit", "/exit"):
            break

        if not text:
            continue

        try:
            run_text(text)
        except Exception as error:
            print(f"ERROR> {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
