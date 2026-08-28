#!/usr/bin/env python3

import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from irc_control_pkg.kinematics import Kinematics
from std_msgs.msg import Empty, Bool, Float64MultiArray, MultiArrayDimension, String, Int16

DOF = 6

SPOON_PICK_POSITION = np.array([0.4, 0.01, 0.265], dtype=float)
SPOON_Q6 = math.radians(90.0)

## 공압으로 최대한 가까이, 낮게 잡을 수 있는 위치: [0.23, 0.0, 0.065], *base x = 7
# [0.20, 0.00, 0.27] -> 카메라가 수직으로 보는 위치
POINT1 = np.array([0.25, 0.00, 0.27], dtype=float) ## 카메라를 수직으로 바라보는 위치
VLM_CONFIRM_POINT = np.deg2rad([83.0, -3.0, 0.0, 87.0, 90.0, 0.0])

INITIAL_PACK_PICK_POINT = np.array([0.25, 0.007, 0.035], dtype=float) # 용기 실제 좌표 x = 0.23.5
INITIAL_PACK_PLACE_POINT = np.array([-0.005, 0.25, 0.05], dtype=float)

PLACE_POINTS = {
    'noodle': {'position': np.array([0.012, 0.3, 0.07], dtype=float), 'yaw_deg': 90.0,},
    'sauce': {'position': np.array([0.03, 0.3, 0.07], dtype=float), 'yaw_deg': 180.0,},  ##yaw 고정
    'mushroom': {'position': np.array([0.03, 0.3, 0.07], dtype=float), 'yaw_deg': 90.0,},
    'onion': {'position': np.array([0.03, 0.3, 0.07], dtype=float), 'yaw_deg': 90.0,},
    'crab': {'position': np.array([0.05, 0.25, 0.07], dtype=float), 'yaw_deg': 90.0,},
    'sausage': {'position': np.array([-0.035, 0.25, 0.07], dtype=float), 'yaw_deg': 90.0,},
    'cover': {'position': np.array([0.03, 0.3, 0.1], dtype=float), 'yaw_deg': 180.0,},  ##yaw 고정
}

LIFT_HEIGHT = 0.15


class PointPoseNode(Node):

    def __init__(self):
        super().__init__('point_pose_node')

        self.kinematics = Kinematics(self.get_logger())

        self.grip_plan_pub = self.create_publisher(Float64MultiArray, '/arm/joint_waypoints', 10)
        self.pack_plan_pub = self.create_publisher(Float64MultiArray, '/arm/joint_waypoints_pack', 10)
        self.joint_target_pub = self.create_publisher(Float64MultiArray, '/arm/joint_target', 10)
        self.create_subscription(Empty, '/arm/motion_done', self.arm_motion_done_callback, 10)

        self.cp_done_pub = self.create_publisher(Int16, '/control/cp_done', 10)
        self.create_subscription(String, '/reset', self.reset_callback, 10)
        self.create_subscription(Int16, '/control/start', self.start_callback, 10)
        self.create_subscription(String, '/control/plan', self.control_plan_callback, 10)
        self.create_subscription(String, '/control/motion', self.control_motion_callback, 10)
        self.create_subscription(Int16, '/control/vision_request', self.vision_request_callback, 10)

        self.vlm_confirm_ready_pub = self.create_publisher(Bool, '/control/vlm_confirm_ready', 10)
        self.control_home_pub = self.create_publisher(Int16, '/control/home', 10)
        self.create_subscription(String, '/main/confirm', self.main_confirm_callback, 10)

        self.control_motion_done = self.create_publisher(String, '/control/motion_done', 10)
        self.create_subscription(String, '/vision/pick_pose', self.pick_callback, 10)
        
        self.current_ingredient = None
        self.pick_mode = None
        self.point1_q = None
        self.pick_lift_q = None

        self.cheese_commands = []
        self.cheese_command_index = 0
        self.cp_repeat_count = 1
        self.cheese_delay_timer = None

        self.confirm_retry_phase = None
        self.vlm_confirm_pending = False
        self.home_pending = False
        self.vlm_confirm_delay = None

    def start_callback(self, _msg):
        self.plan_initial_pack()

    def control_plan_callback(self, msg):
        data = json.loads(msg.data)
        class_name = str(data['class']).strip()
        mode = self.select_mode(class_name)

        self.current_ingredient = class_name
        self.pick_mode = mode
        self.point1_q = None
        self.pick_lift_q = None
        self.cp_repeat_count = max(1, int(data.get('repeat_count', 1)))

    def control_motion_callback(self, msg):
        command = msg.data.strip()

        if command == 'point1':
            self.move_point(POINT1)
            return

        if command == 'place':
            if self.current_ingredient is None:
                return

            mode = self.select_mode(self.current_ingredient)
            position, yaw = self.move_place_pose()
            self.plan_place(position, yaw, mode)
            return

        if command == 'cp':
            if self.current_ingredient is None:
                return

            self.cheese_commands = self.cp_path(
                self.current_ingredient,
                self.cp_repeat_count
            )
            self.cheese_command_index = 0
            self.after_cp_path()
            return

        if command == 'vlm_confirm':
            self.move_vlm_confirm()
            return

        if command == 'home':
            self.move_home()
            return

    def arm_motion_done_callback(self, _msg):
        if self.cheese_commands:
            self.after_cp_path()
            return

        if self.vlm_confirm_pending:
            self.vlm_confirm_pending = False
            self.vlm_confirm_delay = self.create_timer(
                2.0,
                self.vlm_confirm_delay_done
            )
            return

        if self.home_pending:
            self.home_pending = False

            msg = Int16()
            msg.data = 1
            self.control_home_pub.publish(msg)
            return

        if self.confirm_retry_phase == 'point1':
            self.confirm_retry_phase = 'pick'
            self.request_vision()
            return

        if self.confirm_retry_phase == 'pick':
            self.confirm_retry_phase = 'place'
            position, yaw = self.move_place_pose()
            self.plan_place(position, yaw, self.pick_mode)
            return

        if self.confirm_retry_phase == 'place':
            self.confirm_retry_phase = None
            self.move_vlm_confirm()
            return

    def vlm_confirm_delay_done(self):
        self.vlm_confirm_delay.cancel()
        self.destroy_timer(self.vlm_confirm_delay)

        msg = Bool()
        msg.data = True
        self.vlm_confirm_ready_pub.publish(msg)


    def main_confirm_callback(self, msg):
        if msg.data.strip() == 'success':
            self.move_home()
            return

        data = json.loads(msg.data)
        self.current_ingredient = str(data['class']).strip()
        self.pick_mode = self.select_mode(self.current_ingredient)

        if self.pick_mode == 'cp':
            self.confirm_retry_phase = 'cp'
            self.cheese_commands = self.cp_path(self.current_ingredient, 1)
            self.cheese_command_index = 0
            self.after_cp_path()
            return

        self.confirm_retry_phase = 'point1'
        self.move_point(POINT1)

    def pick_callback(self, msg):
        position, yaw, class_name = self.vision_pick_pose(msg)
        mode = self.select_mode(class_name)
        self.pick_mode = mode

        self.plan_pick(position, yaw, mode)

    def vision_request_callback(self, _msg):
        self.request_vision()

    def reset_callback(self, _msg):
        self.current_ingredient = None
        self.pick_mode = None
        self.point1_q = None
        self.pick_lift_q = None

        self.cheese_commands = []
        self.cheese_command_index = 0
        self.cp_repeat_count = 1

        if self.cheese_delay_timer is not None:
            self.cheese_delay_timer.cancel()
            self.destroy_timer(self.cheese_delay_timer)
            self.cheese_delay_timer = None

        self.confirm_retry_phase = None
        self.vlm_confirm_pending = False
        self.home_pending = False

        if self.vlm_confirm_delay is not None:
            self.vlm_confirm_delay.cancel()
            self.destroy_timer(self.vlm_confirm_delay)
            self.vlm_confirm_delay = None

        self.get_logger().info('초기화 완료')

    def request_vision(self):
        msg = String()
        msg.data = self.current_ingredient
        self.control_motion_done.publish(msg)

    ## 숟가락 잡을 때 x축으로 평행하게 접근 계산
    def spoon_linear_x(self, start_position, end_position, previous_q, q6, step=0.01):
        distance = abs(end_position[0] - start_position[0])
        count = max(1, int(math.ceil(distance / step)))

        waypoints = []
        q_previous = previous_q.copy()

        for i in range(1, count + 1):
            ratio = i / count

            position = start_position.copy()
            position[0] = start_position[0] + ratio * (end_position[0] - start_position[0])

            q = self.kinematics.solve_cp_path(position, q_previous, q6)

            waypoints.append(q)
            q_previous = q.copy()

        return waypoints

    def cp_path(self, class_name, repeat_count=1):
        q_home = np.zeros(DOF)
        q_start = q_home.copy()

        # 숟가락 접근
        spoon_approach_position = SPOON_PICK_POSITION.copy()
        spoon_approach_position[0] -= 0.08

        # 숟가락 lift
        spoon_lift_position = SPOON_PICK_POSITION.copy()
        spoon_lift_position[2] += 0.10

        # 치즈 접근 전 lift, 치즈 접근
        if class_name == 'cheese':
            q_cheese_approach_lift = np.deg2rad([-13.0, 5.0, 0.0, 108.0, 70.0, 0.0]) # 치즈 접근 위치
            q_cheese_approach_lift[3] += math.radians(-20)
            q_cheese_approach_lift[4] += math.radians(20)
            q_cheese_approach = np.deg2rad([-13.0, 5.0, 0.0, 108.0, 70.0, 0.0]) # 치즈 접근 위치

        else: 
            q_cheese_approach_lift = np.deg2rad([-8.0, 30.0, 0.0, 63.0, 85.0, 0.0]) # 페퍼론치노 접근 위치
            q_cheese_approach_lift[3] += math.radians(-20)
            q_cheese_approach_lift[4] += math.radians(20)
            q_cheese_approach = np.deg2rad([-8.0, 30.0, 0.0, 63.0, 85.0, 0.0]) # 페퍼론치노 접근 위치

        # 치즈 푸기: 2번, 3번 모터 place 할 때랑 맞추기
        q_cheese_touch_1 = q_cheese_approach.copy()
        q_cheese_touch_1[0] += math.radians(95)
        q_cheese_touch_1[2] = math.radians(-90)

        q_cheese_touch_2 = q_cheese_touch_1.copy()
        q_cheese_touch_2[1] = math.radians(-90)

        # 치즈 푸고 나서 lift
        q_cheese_lift = q_cheese_touch_2.copy()
        q_cheese_lift[2] += math.radians(40)
        q_cheese_lift[5] += math.radians(40)

        # 숟가락 접근
        q_spoon_approach = self.kinematics.solve_cp_path(spoon_approach_position, q_start, SPOON_Q6)

        q_spoon_pick_path = self.spoon_linear_x(spoon_approach_position, SPOON_PICK_POSITION, q_spoon_approach, SPOON_Q6, step=0.01)

        q_spoon_pick = q_spoon_pick_path[-1]

        # 숟가락 잡은 후 lift
        q_spoon_lift = self.kinematics.solve_cp_path(spoon_lift_position, q_spoon_pick, SPOON_Q6) 

        # 치즈 / 페페론치노 place 준비
        if class_name == 'cheese':
            q_cheese_place_ready = np.deg2rad([130.0, -90.0, -90.0, 130.0, 20.0, 0.0,]) # 치즈 place 위치
       
        else: q_cheese_place_ready = np.deg2rad([140.0, -90.0, -90.0, 130.0, 25.0, 0.0,]) # 페퍼론치노 place 위치

        # 치즈, 페퍼론치노 place
        q_cheese_release_1 = q_cheese_place_ready.copy()
        q_cheese_release_1[5] += math.radians(-110.0)

        q_cheese_release_2 = q_cheese_release_1.copy()
        q_cheese_release_2[5] += math.radians(110.0)

        # 숟가락 내려놓고 x축 빠지기
        q_spoon_retreat_path = self.spoon_linear_x(SPOON_PICK_POSITION, spoon_approach_position, q_spoon_pick, SPOON_Q6, step=0.01)

        commands_1 = [
            # POINT1 -> 숟가락 접근 -> 숟가락 위치
            ('motion', [q_spoon_approach, *q_spoon_pick_path]),

            # 숟가락 잡기
            ('gripper', 'grip_pick:spoon', q_spoon_pick),

            # 숟가락 lift -> 첫 번째 치즈/페퍼론치노 푸기 -> 놓기
            ('motion', [
                q_spoon_lift,
                q_cheese_approach_lift,
                q_cheese_approach,
                q_cheese_touch_1,
                q_cheese_touch_2,
                q_cheese_lift,
                q_cheese_place_ready,
                q_cheese_release_1,
                q_cheese_release_2
            ]),
            ('delay', 1.5),
        ]

        commands_2 = [
            # 숟가락은 이미 잡고 있으므로 다시 푸기 -> 놓기만 수행
            ('motion', [
                q_cheese_approach_lift,
                q_cheese_approach,
                q_cheese_touch_1,
                q_cheese_touch_2,
                q_cheese_lift,
                q_cheese_place_ready,
                q_cheese_release_1,
                q_cheese_release_2
            ]),
            ('delay', 1.5),
        ]

        commands = commands_1.copy()

        for i in range(repeat_count - 1):
            commands.extend(commands_2)

        # 모든 반복이 끝난 뒤에만 숟가락 반납
        commands.extend([
            ('motion', [q_spoon_lift, q_spoon_pick]),
            ('gripper', 'grip_place:spoon', q_spoon_pick),
            ('motion', q_spoon_retreat_path)
        ])

        return commands

    def after_cp_path(self):
        if self.cheese_command_index >= len(self.cheese_commands):
            self.cheese_commands = []
            self.cheese_command_index = 0

            # VLM fail로 CP를 한 번 더 하는 경우
            if self.confirm_retry_phase == 'cp':
                self.confirm_retry_phase = None
                self.move_vlm_confirm()
                return

            # 정상 CP 1회 완료
            msg = Int16()
            msg.data = 1
            self.cp_done_pub.publish(msg)
            return

        command = self.cheese_commands[self.cheese_command_index]
        self.cheese_command_index += 1
        kind = command[0]

        if kind == 'motion':
            msg = Float64MultiArray()
            msg.data = np.asarray(command[1]).reshape(-1).tolist()
            self.joint_target_pub.publish(msg)
            return

        if kind == 'gripper':
            phase = command[1]
            q = command[2]

            msg = Float64MultiArray()
            msg.layout.dim = [
                MultiArrayDimension(
                    label=phase,
                    size=4,
                    stride=DOF * 4
                )
            ]
            msg.data = np.tile(q, 4).tolist()
            self.grip_plan_pub.publish(msg)
            return

        if kind == 'delay':
            self.cheese_delay_timer = self.create_timer(
                command[1],
                self.cheese_delay_done
            )

    def cheese_delay_done(self):
        self.cheese_delay_timer.cancel()
        self.destroy_timer(self.cheese_delay_timer)
        self.cheese_delay_timer = None
        self.after_cp_path()

    def move_point(self, point):
        q_target = self.kinematics.solve_pose(
            point,
            np.zeros(DOF, dtype=float),
            math.pi,
            'grip'
        )

        if np.allclose(point, POINT1):
            self.point1_q = q_target

        msg = Float64MultiArray()
        msg.data = q_target.tolist()
        self.joint_target_pub.publish(msg)

    def move_vlm_confirm(self):
        q_target = VLM_CONFIRM_POINT.copy()

        self.vlm_confirm_pending = True

        msg = Float64MultiArray()
        msg.data = q_target.tolist()
        self.joint_target_pub.publish(msg)

    def move_home(self):
        self.home_pending = True

        msg = Float64MultiArray()
        msg.data = [0.0] * DOF
        self.joint_target_pub.publish(msg)

    @staticmethod
    def select_mode(class_name):
        if class_name in {'cheese', 'pepperoncino'}:
            return 'cp'
        if class_name in {'noodle_thick', 'noodle_thin', 'mushroom', 'onion', 'crab', 'sausage'}:
            return 'grip'
        if class_name in {'cover', 'sauce_tomato', 'sauce_cream', 'sauce_oil'}:
            return 'pack'
        raise ValueError(f'클래스 안맞음: {class_name}')

    def move_place_pose(self):
        class_name = self.current_ingredient
        if class_name in {'sauce_tomato', 'sauce_cream', 'sauce_oil'}:
            place_key = 'sauce'
        elif class_name in {'noodle_thick', 'noodle_thin'}:
            place_key = 'noodle'
        else: place_key = class_name

        place_pose = PLACE_POINTS[place_key]
        position = place_pose['position']
        yaw = math.radians(float(place_pose['yaw_deg']))

        return position, yaw

    @staticmethod
    def vision_pick_pose(msg):
        data = json.loads(msg.data)

        position = np.array([
            float(data['x']),
            float(data['y']),
            float(data['z']),
        ], dtype=float)

        yaw_value = data.get('yaw', 180.0)
        if yaw_value is None: yaw_value = 180.0

        yaw = math.radians(float(yaw_value))
        class_name = str(data['class_name']).strip()

        return position, yaw, class_name

    ## 공압 IK에서는 q3을 홈 각도(180)로 강제
    def solve_pack_pick_lift(self, position, previous_q):
        lift_position = position.copy()
        lift_position[2] += LIFT_HEIGHT

        q_pick = self.kinematics.solve_pose(
            position,
            previous_q,
            math.pi,
            'pack'
        )

        q_lift = self.kinematics.solve_pose(
            lift_position,
            q_pick,
            math.pi,
            'pack'
        )

        return q_pick, q_lift

    def plan_pick(self, position, yaw, mode):
        approach = position.copy()
        approach[2] += LIFT_HEIGHT

        if self.point1_q is not None:
            q_motion_start = self.point1_q
        else: q_motion_start = np.zeros(DOF, dtype=float)

        if mode == 'pack':
            q_pick, q_lift = self.solve_pack_pick_lift(position, q_motion_start)
            q_approach = q_lift
        else:
            q_approach = self.kinematics.solve_pose(
                approach,
                q_motion_start,
                yaw,
                'grip'
            )

            q_pick = self.kinematics.solve_pose(
                position,
                q_approach,
                yaw,
                'grip'
            )

            q_lift = self.kinematics.solve_pose(
                approach,
                q_pick,
                yaw,
                'grip'
            )
        # lift 된 지점에서 바로 place 시작
        self.pick_lift_q = q_lift

        if mode == 'pack':
            phase = 'pack_pick'
        else: phase = f'grip_pick:{self.current_ingredient}'

        self.publish_waypoints(
            self.pack_plan_pub
            if mode == 'pack'
            else self.grip_plan_pub,
            phase,
            q_approach,
            q_pick,
            q_lift,
            q_lift
        )
    
    def plan_place(self, position, yaw, mode):
        approach = position.copy()
        approach[2] += LIFT_HEIGHT

        if mode == 'pack':
            q_approach = self.kinematics.solve_pose(
                approach,
                self.pick_lift_q,
                math.pi,
                'pack'
            )

            q_place = self.kinematics.solve_pose(
                position,
                q_approach,
                math.pi,
                'pack'
            )

            q_lift = q_approach

        else:
            q_approach = self.kinematics.solve_grip_place_pose(approach, self.pick_lift_q, yaw)

            q_place = self.kinematics.solve_grip_place_pose(position, q_approach, yaw)

            q_lift = self.kinematics.solve_grip_place_pose(approach, q_place, yaw)

        if mode == 'pack':
            phase = 'pack_place'
        else: phase = f'grip_place:{self.current_ingredient}'

        self.publish_waypoints(
            self.pack_plan_pub
            if mode == 'pack'
            else self.grip_plan_pub,
            phase,
            q_approach,
            q_place,
            q_lift,
            q_lift
        )

    def plan_initial_pack(self):
        p1 = INITIAL_PACK_PICK_POINT
        p2 = INITIAL_PACK_PLACE_POINT

        p2_lift = p2.copy()
        p2_lift[2] = p1[2] + LIFT_HEIGHT

        q_home = np.zeros(DOF, dtype=float)
        q_p1, q_p1_lift = self.solve_pack_pick_lift(p1, q_home)
        q_p2_lift = self.kinematics.solve_pose(
            p2_lift,
            q_p1_lift,
            math.pi,
            'pack'
        )
        q_p2 = self.kinematics.solve_pose(
            p2,
            q_p2_lift,
            math.pi,
            'pack'
        )

        self.publish_waypoints(
            self.pack_plan_pub,
            'pack_full',
            q_p1,
            q_p1_lift,
            q_p2_lift,
            q_p2
        )

    @staticmethod
    def publish_waypoints(publisher, phase, q1, q2, q3, q4):
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(
                label=phase,
                size=4,
                stride=DOF * 4
            )
        ]
        msg.data = np.concatenate([q1, q2, q3, q4]).tolist()
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PointPoseNode()

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