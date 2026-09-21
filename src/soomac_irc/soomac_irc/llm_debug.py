import argparse
import copy
import json

import rclpy

import soomac_irc.llm_node as llm_runtime


# Debug는 운영 LLM 흐름을 그대로 쓰되 카메라와 VLM 판정만 사용하지 않는다.
llm_runtime.ENABLE_VLM = False
llm_runtime.ENABLE_VLM_UI_IMAGES = False


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class _DebugPublisher:
    # 운영 메서드의 publish 호출을 메모리에만 보관해 실제 ROS 토픽 발행을 막는다.
    def __init__(self):
        self.last_message = None

    def publish(self, message):
        self.last_message = copy.deepcopy(message)


class LLMDebugNode(llm_runtime.LLMNode):
    # 운영 LLMNode를 직접 재사용하고 터미널 입력과 가상 로봇 완료 신호만 추가한다.
    def __init__(self, manual_flow=False):
        self.manual_flow = manual_flow
        self.last_reply = None
        super().__init__()

        # Debug는 운영 상태 전이만 사용하고 로봇·UI·TTS 토픽은 절대 발행하지 않는다.
        real_publishers = [self.reply_pub, self.plan_pub, self.done_pub, self.status_pub, self.stt_enable_pub, self.vlm_result_pub]

        for publisher in real_publishers:
            self.destroy_publisher(publisher)

        self.reply_pub = _DebugPublisher()
        self.plan_pub = _DebugPublisher()
        self.done_pub = _DebugPublisher()
        self.status_pub = _DebugPublisher()
        self.stt_enable_pub = _DebugPublisher()
        self.vlm_result_pub = _DebugPublisher()
        self.vlm_ui_image_pub = None


    def _publish_reply(self, reply: str):
        # 운영과 동일하게 ROS/UI/TTS 토픽과 history에 반영하면서 터미널에도 표시한다.
        super()._publish_reply(reply)
        self.last_reply = reply.strip()
        print(f"\nROBOT> {self.last_reply}")


    def _record_tool_turn(self, user_text: str, state_before: dict, result: dict, reply: str | None):
        # 운영 JSONL 로그를 유지하고 Tool·정책·자유응답 결과를 터미널에도 표시한다.
        super()._record_tool_turn(user_text, state_before, result, reply)
        reply_trace = getattr(self.generate_reply, "last_trace", None) or {}
        print(f"TOOL> {_json(result.get('tool_call'))}")
        print(f"TURN_RESULT> {_json(result.get('turn_result'))}")
        print(f"POLICY> {result.get('policy_reply')}")
        print(f"RAW_REPLY> {reply_trace.get('raw_output')}")


    def state_snapshot(self) -> dict:
        # 모델에 들어가는 상태와 Debug 제어 상태를 한 번에 확인한다.
        state = self._build_agent_state("")
        state["control"] = {
            "manual_flow": self.manual_flow,
            "awaiting_confirm": copy.deepcopy(self.awaiting_confirm),
            "conversation_started": self.conversation_started,
            "order_finished": self.order_finished,
            "vlm_enabled": llm_runtime.ENABLE_VLM,
        }
        return state


    def restart_session(self):
        # 외부 /ui/start와 첫 /llm/next 대신 새 주문을 터미널에서 바로 시작한다.
        self._clear_state()
        self._prepare_order()
        self._start_order()
        self._publish_status()


    def complete_active_task(self, finish_sauce=False):
        # 카메라 없이 현재 가상 로봇 작업 하나를 완료 처리한다.
        if self.active_task is None:
            print("DEBUG> 진행 중인 가상 작업이 없습니다.")
            return

        if self.section == "sauce":
            if not finish_sauce:
                print("DEBUG> 소스 작업은 /finish로 완료하세요.")
                return

            self._process_finish()
        else:
            self._process_next(llm_runtime.BY_VLM_NUM)

        self._publish_status()


    def drain_auto_tasks(self):
        # auto 모드에서는 시작된 가상 로봇 작업을 연속 완료해 다음 사용자 선택까지 이동한다.
        for _ in range(64):
            if self.active_task is None or self.order_finished:
                return

            if self.section == "sauce":
                self.complete_active_task(finish_sauce=True)
            else:
                self.complete_active_task()

        raise RuntimeError("가상 작업 자동 진행이 64회를 초과함")


    def handle_user_text(self, user_text: str):
        # 터미널 한 문장을 운영 _process_turn에 그대로 전달한다.
        user_text = user_text.strip()

        if not user_text:
            return

        if self.manual_flow and (self.active_task is not None or self.task_queue):
            completion_command = "/finish" if self.section == "sauce" else "/next"
            print(f"DEBUG> 가상 로봇 작업 중입니다. {completion_command}를 입력하세요.")
            return

        self._process_turn(user_text)

        if not self.manual_flow:
            self.drain_auto_tasks()

        self._publish_status()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROS 외부 입력과 카메라 없이 SOOMAC LLM을 검사합니다.")
    parser.add_argument("--manual-flow", action="store_true", help="로봇 작업을 /next와 /finish 명령으로 직접 완료합니다.")
    parser.add_argument("--base-only", action="store_true", help="Tool LoRA 없이 base model만 사용합니다.")
    parser.add_argument("--adapter-path", default=None, help="기본값 대신 사용할 Tool LoRA 경로입니다.")
    parser.add_argument("--once", action="append", default=[], help="대화형 입력 대신 지정한 문장을 순서대로 실행합니다.")
    return parser


def _run_command(node: LLMDebugNode, text: str) -> bool:
    # Debug 전용 명령과 일반 사용자 발화를 구분한다.
    if text in ("/quit", "/exit"):
        return False

    if text == "/state":
        print(_json(node.state_snapshot()))
    elif text == "/history":
        print(_json(node.history))
    elif text == "/reset":
        node.restart_session()
    elif text == "/next":
        node.complete_active_task()
    elif text == "/finish":
        node.complete_active_task(finish_sauce=True)
    else:
        node.handle_user_text(text)

    return True


def main(argv=None):
    args = _build_parser().parse_args(argv)
    llm_runtime.ENABLE_TOOL_LORA = not args.base_only

    if args.adapter_path:
        llm_runtime.TOOL_ADAPTER_PATH = args.adapter_path

    rclpy.init(args=None)
    node = LLMDebugNode(manual_flow=args.manual_flow)

    try:
        node.restart_session()
        print(f"MODE> {'manual' if args.manual_flow else 'auto'} / VLM=OFF")
        print("COMMANDS> /state /history /reset /next /finish /quit")

        if args.once:
            for text in args.once:
                print(f"\nUSER> {text}")

                if not _run_command(node, text):
                    break

            return

        while True:
            try:
                text = input("\nUSER> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not text:
                continue

            try:
                if not _run_command(node, text):
                    break
            except Exception as error:
                print(f"ERROR> {type(error).__name__}: {error}")
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
