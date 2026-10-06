import copy
import json
import os
from datetime import datetime
from pathlib import Path


class LLMSessionJsonlLogger:
    """세션별 LLM trace와 보조 runtime event를 JSONL로 저장한다.

    설계 원칙:
    - 별도 logging thread를 만들지 않는다. 기존 llm worker가 한 job씩 순차 처리하므로
      한 줄 append를 같은 worker에서 수행한다.
    - 주문 세션마다 디렉토리를 하나 만들고 agent_turns.jsonl / runtime_events.jsonl을 분리한다.
    - 로그 write 실패는 실제 주문 runtime을 중단시키지 않는다.
    - 성공 turn과 실패 turn을 모두 남길 수 있도록 log_turn()은 독립적으로 호출한다.
    """

    SCHEMA_VERSION = 1

    def __init__(self, base_dir: str | None = None, logger=None):
        configured = base_dir or os.environ.get("SOOMAC_LLM_LOG_DIR")
        self.base_dir = (
            Path(configured).expanduser()
            if configured
            else Path.home() / "IRC_SOOMAC3.0" / "logs" / "llm"
        )
        self.logger = logger
        self.session_id: str | None = None
        self.session_dir: Path | None = None
        self.turn_log_path: Path | None = None
        self.event_log_path: Path | None = None
        self._event_seq = 0
        self._turn_id = 0
        self._write_errors = 0

    @property
    def active(self) -> bool:
        return (
            self.session_id is not None
            and self.session_dir is not None
            and self.turn_log_path is not None
            and self.event_log_path is not None
        )

    def start_session(self, initial_session: dict, robot_state: dict, reason: str = "ui_start") -> str:
        # 이미 세션이 열려 있으면 기존 세션을 닫고 새 세션을 시작한다.
        if self.active:
            self.end_session(
                reason="replaced_by_new_session",
                final_session=initial_session,
                robot_state=robot_state,
            )

        now = datetime.now().astimezone()
        self.session_id = f"{now.strftime('%Y%m%d_%H%M%S_%f')}_pid{os.getpid()}"
        self.session_dir = self.base_dir / now.strftime("%Y-%m-%d") / f"session_{self.session_id}"
        self.turn_log_path = self.session_dir / "agent_turns.jsonl"
        self.event_log_path = self.session_dir / "runtime_events.jsonl"
        self._event_seq = 0
        self._turn_id = 0

        # 디렉토리/파일 생성 실패도 runtime을 죽이지 않는다.
        try:
            self.session_dir.mkdir(parents=True, exist_ok=False)
            self.turn_log_path.touch(exist_ok=False)
            self.event_log_path.touch(exist_ok=False)
        except FileExistsError:
            # timestamp+pid라 충돌 가능성은 매우 낮지만, 충돌하면 append 가능한 상태로 둔다.
            try:
                self.session_dir.mkdir(parents=True, exist_ok=True)
                self.turn_log_path.touch(exist_ok=True)
                self.event_log_path.touch(exist_ok=True)
            except Exception as error:
                self._write_errors += 1
                self._warn(f"LLM session 로그 파일 준비 실패 ({type(error).__name__}): {error}")
        except Exception as error:
            self._write_errors += 1
            self._warn(f"LLM session 로그 파일 준비 실패 ({type(error).__name__}): {error}")

        self.log_event(
            "session_start",
            {
                "reason": reason,
                "session": copy.deepcopy(initial_session),
                "robot_state": copy.deepcopy(robot_state),
            },
        )
        return str(self.session_dir)

    def next_turn_id(self) -> int:
        self._turn_id += 1
        return self._turn_id

    def log_turn(self, payload: dict) -> None:
        if not self.active or self.turn_log_path is None:
            return

        row = {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": "agent_turn",
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            **copy.deepcopy(payload),
        }
        self._append_jsonl(self.turn_log_path, row, "agent turn")

    def log_event(self, event_type: str, payload: dict | None = None) -> None:
        if not self.active or self.event_log_path is None:
            return

        self._event_seq += 1
        row = {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": "runtime_event",
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "event_seq": self._event_seq,
            "event_type": event_type,
            **copy.deepcopy(payload or {}),
        }
        self._append_jsonl(self.event_log_path, row, "runtime event")

    def end_session(self, reason: str, final_session: dict, robot_state: dict) -> None:
        if not self.active:
            return

        self.log_event(
            "session_end",
            {
                "reason": reason,
                "session": copy.deepcopy(final_session),
                "robot_state": copy.deepcopy(robot_state),
                "turn_count": self._turn_id,
                "write_errors": self._write_errors,
            },
        )
        self.session_id = None
        self.session_dir = None
        self.turn_log_path = None
        self.event_log_path = None

    def close(self) -> None:
        # 동기 append 방식이므로 종료할 writer thread나 flush queue가 없다.
        return

    def _append_jsonl(self, path: Path, row: dict, label: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                    + "\n"
                )
                handle.flush()
        except Exception as error:
            self._write_errors += 1
            self._warn(f"LLM {label} 로그 저장 실패 ({type(error).__name__}): {error}")

    def _warn(self, message: str) -> None:
        try:
            if self.logger is not None:
                self.logger.warning(message)
            else:
                print(message)
        except Exception:
            pass
