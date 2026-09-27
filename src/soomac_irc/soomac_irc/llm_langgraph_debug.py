import argparse
import copy
import json

import rclpy

import soomac_irc.llm_langgraph_node as llm_runtime


# 운영 LangGraph와 정책은 그대로 사용하고 카메라·VLM 추론만 끈다.
llm_runtime.ENABLE_VLM = False
llm_runtime.ENABLE_VLM_UI_IMAGES = False


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class _DebugPublisher:
    # 운영 publish 결과를 메모리에 보관해 실제 UI·TTS·MAIN 토픽으로 내보내지 않는다.
    def __init__(self):
        self.last_message = None

    def publish(self, message):
        self.last_message = copy.deepcopy(message)


class LLMLangGraphDebugNode(llm_runtime.LLMLangGraphNode):
    def __init__(self, manual_flow=False, call_decision=None):
        # 역할: 운영 Node의 모델·graph·정책·멀티턴 상태를 재사용하고 외부 ROS I/O만 차단한다.
        # 입력/상태 변경/호출자: 수동 진행 여부와 선택적 테스트 모델 / 운영 Node 상태 / main·검사 코드.
        self.manual_flow = manual_flow
        self.last_reply = None
        super().__init__(call_decision=call_decision)
        self._disconnect_external_ros_io()

    def _disconnect_external_ros_io(self):
        # 터미널 입력과 가상 MAIN 신호만 사용하므로 운영 publisher와 subscriber를 모두 끊는다.
        publishers = [self.reply_pub, self.plan_pub, self.done_pub, self.status_pub, self.stt_enable_pub, self.vlm_result_pub]

        if self.vlm_ui_image_pub is not None:
            publishers.append(self.vlm_ui_image_pub)

        for publisher in publishers:
            self.destroy_publisher(publisher)

        subscriptions = [
            self.stt_sub, self.ui_start_sub, self.ui_reset_sub, self.next_trigger_sub,
            self.confirm_start_sub, self.llm_reset_sub, self.camera_sub,
        ]

        for subscription in subscriptions:
            if subscription is not None:
                self.destroy_subscription(subscription)

        self.reply_pub = _DebugPublisher()
        self.plan_pub = _DebugPublisher()
        self.done_pub = _DebugPublisher()
        self.status_pub = _DebugPublisher()
        self.stt_enable_pub = _DebugPublisher()
        self.vlm_result_pub = _DebugPublisher()
        self.vlm_ui_image_pub = None

    def _publish_reply(self, reply: str):
        # 운영 history와 응답 검사를 그대로 거친 결과를 터미널에도 표시한다.
        super()._publish_reply(reply)
        self.last_reply = reply.strip()
        print(f"\nROBOT> {self.last_reply}")

    def _publish_next_task(self) -> bool:
        # 운영 plan 생성이 성공했을 때 실제 MAIN 대신 터미널에 payload를 보여준다.
        published = super()._publish_next_task()

        if published and self.plan_pub.last_message is not None:
            print(f"PLAN> {self.plan_pub.last_message.data}")

        return published

    def state_snapshot(self) -> dict:
        # 모델이 이어받는 session과 운영 로봇 상태를 한 화면에서 확인한다.
        return {
            "session": copy.deepcopy(self.graph_state),
            "robot": self._build_robot_state(),
            "runtime": {
                "manual_flow": self.manual_flow,
                "conversation_started": self.conversation_started,
                "order_finished": self.order_finished,
                "vlm_enabled": llm_runtime.ENABLE_VLM,
            },
        }

    def print_progress(self):
        print(
            "STATE> "
            f"section={self.section}, active_task={self.active_task}, "
            f"queued={len(self.task_queue)}, completed={len(self.completed_tasks)}, "
            f"finished={self.order_finished}"
        )

    def restart_session(self):
        # 실제 /ui/start와 첫 /llm/next가 처리되는 운영 순서를 그대로 실행한다.
        self._clear_state()
        self._process_start()
        self._process_next(llm_runtime.BY_VLM_NUM)
        self._publish_status()
        self.print_progress()

    def complete_active_task(self, finish_sauce=False):
        # 실제 MAIN의 confirm_start와 next/reset을 순서대로 대신한다.
        if self.active_task is None:
            print("DEBUG> 진행 중인 가상 작업이 없습니다.")
            return

        if self.section == "sauce" and not finish_sauce:
            print("DEBUG> 소스 작업은 /finish로 완료하세요.")
            return

        if self.section != "sauce" and finish_sauce:
            print("DEBUG> 소스 전 작업은 /next로 완료하세요.")
            return

        if not self.vlm_confirmed:
            self._process_vlm_confirm(True)

        if self.section == "sauce":
            self._process_finish()
        else:
            self._process_next(llm_runtime.BY_VLM_NUM)

        self._publish_status()
        self.print_progress()

    def drain_auto_tasks(self):
        # 한 발화로 생긴 작업만 완료하고 다음 사용자 선택이 필요한 section에서 멈춘다.
        for _ in range(64):
            if self.active_task is None or self.order_finished:
                return

            self.complete_active_task(finish_sauce=self.section == "sauce")

        raise RuntimeError("가상 작업 자동 진행이 64회를 초과함")

    def handle_user_text(self, user_text: str):
        # 각 문장을 운영 _process_turn에 전달하므로 history·action_history·정책이 계속 이어진다.
        user_text = user_text.strip()

        if not user_text:
            return

        if self.order_finished:
            print("DEBUG> 주문이 완료됐습니다. /reset으로 새 세션을 시작하세요.")
            return

        if self.manual_flow and (self.active_task is not None or self.task_queue):
            command = "/finish" if self.section == "sauce" else "/next"
            print(f"DEBUG> 가상 로봇 작업 중입니다. {command}를 입력하세요.")
            return

        previous_action_count = len(self.graph_state["action_history"])
        self._process_turn(user_text)
        new_actions = self.graph_state["action_history"][previous_action_count:]

        if new_actions:
            print(f"ACTIONS> {_json(new_actions)}")

        if not self.manual_flow:
            self.drain_auto_tasks()

        self._publish_status()
        self.print_progress()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="운영 LangGraph와 정책을 유지한 채 카메라·VLM 없이 검사합니다.")
    parser.add_argument("--manual-flow", action="store_true", help="가상 로봇 작업을 /next와 /finish로 직접 완료합니다.")
    parser.add_argument("--once", action="append", default=[], help="대화형 입력 대신 지정한 문장을 순서대로 실행합니다.")
    return parser


def _run_command(node: LLMLangGraphDebugNode, text: str) -> bool:
    if text in ("/quit", "/exit"):
        return False

    if text == "/state":
        print(_json(node.state_snapshot()))
    elif text == "/history":
        print(_json(node.graph_state["history"]))
    elif text == "/actions":
        print(_json(node.graph_state["action_history"]))
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
    node = None
    rclpy.init(args=None)

    try:
        node = LLMLangGraphDebugNode(manual_flow=args.manual_flow)
        node.restart_session()
        print(f"MODE> {'manual' if args.manual_flow else 'auto'} / VLM=OFF / CURRENT_LANGGRAPH_POLICY")
        print("COMMANDS> /state /history /actions /reset /next /finish /quit")

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
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
