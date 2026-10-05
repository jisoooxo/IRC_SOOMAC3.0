"""Real-model review through the existing debug node, with all publishers stubbed."""
import hashlib
import json
from pathlib import Path

import rclpy
from soomac_irc.llm_langgraph_debug import LLMLangGraphDebugNode, _DebugPublisher


class IsolatedDebugNode(LLMLangGraphDebugNode):
    def _disconnect_external_ros_io(self):
        super()._disconnect_external_ros_io()
        # The production debug node omits this newer publisher from its list.
        self.destroy_publisher(self.rail_move_ahead_pub)
        self.rail_move_ahead_pub = _DebugPublisher()
        assert all(p.topic_name in ('/parameter_events', '/rosout') for p in self.publishers), "Unexpected live ROS publisher"
        assert not list(self.subscriptions), "Unexpected live ROS subscription"


SCENARIOS = [
    ("01_normal_and_execution", [
        "토마토 소스에 넓은면, 보통 양으로 주세요",
        "바로 진행해주세요",
    ]),
    ("02_ambiguous_normal_noodle", ["보통면으로 주세요"]),
    ("03_stt_misspelling", ["널적면으로 주세요"]),
    ("04_unsupported_no_substitution", [
        "토마토 소스에 넓은면, 보통 양으로 하고 소시지 넣어줘",
        "햄 빼줘",
    ]),
    ("05_mixed_history", [
        "토마토 소스에 넓은면, 보통 양으로. 토마토 소스 역사는?",
    ]),
    ("06_quoted_execution_without_question_mark", [
        "토마토 소스에 넓은면, 보통 양으로 주세요",
        "‘바로 진행해’라는 표현이 무슨 뜻이야",
    ]),
    ("07_quoted_execution_with_question_mark", [
        "토마토 소스에 넓은면, 보통 양으로 주세요",
        "‘바로 진행해’라는 표현이 무슨 뜻이야?",
    ]),
    ("08_missing_noodle_and_portion", [
        "토마토소스의 면은 보통으로 양은 적게주세요.",
        "보통량으로 주세요.",
    ]),
    ("09_general_name", ["문재인"]),
    ("10_original_mixed_misspelling", [
        "토마토 소스로 널적면에 보통량으로 근데 토마토소스는 역사가 어떻게 돼?",
    ]),
]


def main():
    root = Path('/home/roma/IRC_SOOMAC3.0')
    for name in ('agent_prompts.py', 'llm_policy.py', 'model_runtime.py'):
        path = root / 'src/soomac_irc/soomac_irc' / name
        print('SOURCE_SHA256', name, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
    node = None
    rclpy.init()
    try:
        node = IsolatedDebugNode(manual_flow=True)
        print('ISOLATION_PASS no live application publishers/subscriptions; VLM disabled', flush=True)
        for name, utterances in SCENARIOS:
            node.restart_session()
            print('CASE', json.dumps({'name': name, 'log': str(node.runtime_log.turn_log_path)}, ensure_ascii=False), flush=True)
            for utterance in utterances:
                print('USER>', utterance, flush=True)
                node.handle_user_text(utterance)
                print('SNAPSHOT', json.dumps(node.state_snapshot(), ensure_ascii=False), flush=True)
        print('REVIEW_COMPLETE', len(SCENARIOS), 'scenarios', sum(len(turns) for _, turns in SCENARIOS), 'turns', flush=True)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
