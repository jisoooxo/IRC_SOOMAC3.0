#!/usr/bin/env python3

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Bool, Int16, String

SAUCE_CLASSES = {'sauce_tomato', 'sauce_cream', 'sauce_oil'}
CP_CLASSES = {'cheese', 'pepperoncino'}

# MAIN 상태
STATE_WAIT_START = 'WAIT_START'
STATE_WAIT_FIRST_SYNC = 'WAIT_FIRST_SYNC'
STATE_WAIT_LLM_PLAN = 'WAIT_LLM_PLAN'
STATE_WAIT_RAIL_DONE = 'WAIT_RAIL_DONE'
STATE_WAIT_POINT1_DONE = 'WAIT_POINT1_DONE'
STATE_WAIT_PICK_DONE = 'WAIT_PICK_DONE'
STATE_WAIT_PLACE_DONE = 'WAIT_PLACE_DONE'
STATE_WAIT_CP_DONE = 'WAIT_CP_DONE'
STATE_WAIT_HOME_DONE = 'WAIT_HOME_DONE'
STATE_WAIT_LLM_DONE = 'WAIT_LLM_DONE'


class MainNode(Node):

    def __init__(self):
        super().__init__('main_node')

        # MAIN -> LLM
        self.llm_next_pub = self.create_publisher(Int16, '/llm/next', 10)
        self.llm_reset_pub = self.create_publisher(String, '/llm/reset', 10)

        # MAIN -> CONTROL
        self.control_start_pub = self.create_publisher(Int16, '/control/start', 10)
        self.control_plan_pub = self.create_publisher(String, '/control/plan', 10)
        self.control_motion_pub = self.create_publisher(String, '/control/motion', 10)
        self.control_vision_request_pub = self.create_publisher(Int16, '/control/vision_request', 10)

        # MAIN -> RAIL
        self.rail_motion_pub = self.create_publisher(String, '/rail/motion', 10)

        # MAIN -> 전체 노드 초기화
        self.reset_pub = self.create_publisher(String, '/reset', 10)

        # UI -> MAIN
        self.create_subscription(String, '/ui/start', self._ui_start_callback, 10)

        # LLM -> MAIN
        self.create_subscription(String, '/llm/plan', self._llm_plan_callback, 10)
        self.create_subscription(Bool, '/llm/done', self._llm_done_callback, 10)

        # CONTROL -> MAIN
        self.create_subscription(Empty, '/arm/motion_done', self._arm_motion_done_callback, 10)
        self.create_subscription(Int16, '/control/cp_done', self._cp_done_callback, 10)

        # RAIL -> MAIN
        self.create_subscription(String, '/rail/motion_done', self._rail_motion_done_callback, 10)

        
        # MAIN 상태
        self.state = STATE_WAIT_START
        self.current_class = None
        self.repeat_count = 0
        self.repeat_completed = 0

        # 처음 시작할 때만 LLM과 CONTROL을 동시 실행하므로 두 결과가 모두 도착했는지 확인
        self.first_llm_plan_received = False
        self.initial_motion_done = False

        self.get_logger().info('MAIN 준비 완료')

    ## 순서 
    def _arm_motion_done_callback(self, _msg):
        # 초기 용기 이동 완료
        if self.state == STATE_WAIT_FIRST_SYNC:
            self.initial_motion_done = True
            self._try_start_first_rail_motion()
            return

        # POINT1 완료 → Vision 요청
        if self.state == STATE_WAIT_POINT1_DONE:
            self.state = STATE_WAIT_PICK_DONE
            self._publish_vision_request()
            return

        # PICK 완료 → PLACE
        if self.state == STATE_WAIT_PICK_DONE:
            self.state = STATE_WAIT_PLACE_DONE
            self._publish_control_motion('place')
            return

        # PLACE 완료 → 반복 또는 HOME
        if self.state == STATE_WAIT_PLACE_DONE:
            self.repeat_completed += 1

            if self.repeat_completed < self.repeat_count:
                self.state = STATE_WAIT_POINT1_DONE
                self._publish_control_motion('point1')
                return

            self.state = STATE_WAIT_HOME_DONE
            self._publish_control_motion('home')
            return

        # HOME 완료 → 현재 class 종료
        if self.state == STATE_WAIT_HOME_DONE:
            self._finish_current_class()

    def _finish_current_class(self):
        finished_class = self.current_class
        self._clear_current_task()

        # cover까지 완료
        if finished_class == 'cover':
            self.state = STATE_WAIT_LLM_DONE

            msg = String()
            msg.data = 'reset'
            self.llm_reset_pub.publish(msg)
            return

        # sauce 다음에는 바로 cover
        if finished_class in SAUCE_CLASSES:
            self.current_class = 'cover'
            self.repeat_count = 1
            self.repeat_completed = 0

            self._publish_control_plan()
            self._start_rail_motion()
            return

        # 일반 재료 완료 → 다음 LLM plan
        self.state = STATE_WAIT_LLM_PLAN
        self._publish_llm_next()

    ## ================ UI START ==================== ##
    def _ui_start_callback(self, msg):
        if self.state != STATE_WAIT_START:
            return

        self._reset_main_values()

        self.state = STATE_WAIT_FIRST_SYNC

        # 처음에는 LLM의 첫 계획과 CONTROL의 초기 용기 이동을 동시에 시작한다.
        self._publish_llm_next()

        control_msg = Int16()
        control_msg.data = 0
        self.control_start_pub.publish(control_msg)

        self.get_logger().info(
            '면, 소스 주문 받기 및 용기 옮기기 시작'
        )

    ## ================ LLM ==================== ##
    def _publish_llm_next(self):
        msg = Int16()
        msg.data = 3
        self.llm_next_pub.publish(msg)
    
    def _llm_plan_callback(self, msg):
        if self.state not in (STATE_WAIT_FIRST_SYNC, STATE_WAIT_LLM_PLAN):
            return

        try:
            class_name, repeat_count = self._read_llm_plan(msg)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        self.current_class = class_name
        self.repeat_count = repeat_count
        self.repeat_completed = 0

        self._publish_control_plan()

        self.get_logger().info(
            f'class={self.current_class}, repeat_count={self.repeat_count}'
        )

        # 첫 면 단계에서는 control의 초기 용기 이동도 동시에 진행 중이다.
        if self.state == STATE_WAIT_FIRST_SYNC:
            self.first_llm_plan_received = True
            self._try_start_first_rail_motion()
            return

        # 이후 야채/육류/추가재료/소스 단계는 LLM plan을 받는 즉시 rail 이동을 시작한다.
        self._start_rail_motion()

    def _llm_done_callback(self, msg):
        if self.state != STATE_WAIT_LLM_DONE:
            self.get_logger().warning(f'LLM done ignored. Current state: {self.state}')
            return

        if not msg.data:
            return  ## LLM이 주는 True 값을 받으면 나머지 초기화 시작

        reset_msg = String()
        reset_msg.data = 'reset'
        self.reset_pub.publish(reset_msg)

        self._reset_main_values()
        self.state = STATE_WAIT_START

    ## ================ CONTROL ==================== ##
    def _publish_control_plan(self):
        msg = String()
        msg.data = json.dumps({'class': self.current_class,}, ensure_ascii=False)

        self.control_plan_pub.publish(msg)

    def _publish_control_motion(self, command):
        msg = String()
        msg.data = command
        self.control_motion_pub.publish(msg)

    def _publish_vision_request(self):
        msg = Int16()
        msg.data = 1
        self.control_vision_request_pub.publish(msg)

    def _cp_done_callback(self, _msg):
        if self.state != STATE_WAIT_CP_DONE:
            return

        self.repeat_completed += 1

        if self.repeat_completed < self.repeat_count:
            self._publish_control_motion('cp')
            return

        self._finish_current_class()

    ## ================ RAIL ==================== ##
    def _try_start_first_rail_motion(self):
        if not (
            self.first_llm_plan_received
            and self.initial_motion_done
        ):
            return

        self._start_rail_motion()

    def _start_rail_motion(self):
        if self.current_class is None:
            return

        msg = String()
        msg.data = self.current_class

        self.state = STATE_WAIT_RAIL_DONE
        self.rail_motion_pub.publish(msg)

        self.get_logger().info(f'Rail class: {self.current_class}')

    def _rail_motion_done_callback(self, _msg):
        if self.state != STATE_WAIT_RAIL_DONE:
            return

        if self.current_class in CP_CLASSES:
            self.state = STATE_WAIT_CP_DONE
            self._publish_control_motion('cp')
            return

        self.state = STATE_WAIT_POINT1_DONE
        self._publish_control_motion('point1')

    @staticmethod
    def _read_llm_plan(msg):
        try:
            data = json.loads(msg.data)
            class_name = str(data['class']).strip()
            repeat_count = int(data['repeat_count'])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError('LLM plan 형식이 잘못되었습니다.') from error

        if not class_name:
            raise ValueError('LLM class가 비어 있습니다.')

        if repeat_count < 1:
            raise ValueError('repeat_count는 1 이상이어야 합니다.')

        return class_name, repeat_count
    
    def _clear_current_task(self):
        self.current_class = None
        self.repeat_count = 0
        self.repeat_completed = 0

    def _reset_main_values(self):
        self.current_class = None
        self.repeat_count = 0
        self.repeat_completed = 0

        self.first_llm_plan_received = False
        self.initial_motion_done = False


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()