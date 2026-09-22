#!/usr/bin/env python3

import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from irc_control_pkg.kinematics_irc import IRCKinematics
from irc_control_pkg.cp_motion import CPMotion
from std_msgs.msg import Empty, Bool, Float64MultiArray, MultiArrayDimension, String, Int16

DOF = 6
CONTROL_READY = np.deg2rad([0.0, -90.0, 0.0, 113.0, 67.0, 0.0])
CONTROL_READY2 = np.deg2rad([180.0, -90.0, 0.0, 113.0, 67.0, 0.0])

## 공압으로 최대한 가까이, 낮게 잡을 수 있는 위치: [0.23, 0.0, 0.065], *base x = 7
POINT1 = np.deg2rad([0.0, -5.8, 0.0, 70.0, 111.0, 0.0]) ## 카메라가 수직으로 바라보는 위치
POINT2 = np.deg2rad([85.0, -5.8, 0.0, 70.0, 87.0, -7.0]) ## 카메라를 수직으로 바라보는 위치_뚜껑
VLM_CONFIRM_POINT = np.deg2rad([-102.0, -5.8, 0.0, 70.0, 114.0, -11.0])
VLM_CONFIRM_POINT2 = np.deg2rad([258.0, -5.8, 0.0, 70.0, 114.0, -11.0])

INITIAL_PACK_PICK_POINT = np.array([-0.25, 0.000, 0.04], dtype=float) # 용기 실제 좌표 x = 0.23.5
INITIAL_PACK_PLACE_POINT = np.array([-0.01, -0.25, 0.04], dtype=float)

TOMATO_PICK_POINT = np.array([-0.25, 0.14, 0.04], dtype=float)
CREAM_PICK_POINT = np.array([-0.25, 0.000, 0.04], dtype=float)
OIL_PICK_POINT = np.array([-0.25, -0.14, 0.04], dtype=float)
SAUCE_PLACE_POINT = np.array([0.000, -0.25, 0.07], dtype=float)
PLACE_POINTS = {
    'noodle': {'position': np.array([-0.012, -0.19, 0.06], dtype=float), 'yaw_deg': 180.0,},
    'mushroom': {'position': np.array([-0.065, -0.305, 0.06], dtype=float), 'yaw_deg': 180.0,},
    'onion': {'position': np.array([-0.055, -0.305, 0.06], dtype=float), 'yaw_deg': 180.0,},
    'crab': {'position': np.array([-0.065, -0.24, 0.06], dtype=float), 'yaw_deg': 180.0,},
    'sausage': {'position': np.array([0.062, -0.30, 0.06], dtype=float), 'yaw_deg': 180.0,},
    'cover': {'position': np.array([-0.01, -0.25, 0.06], dtype=float), 'yaw_deg': 180.0,},  ##yaw 고정
}

LIFT_HEIGHT = 0.15


class PointPoseNode(Node):

    def __init__(self):
        super().__init__('point_pose_node')

        self.kinematics = IRCKinematics(self.get_logger())
        self.cp_motion = CPMotion(self.kinematics)

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

        self.cp_commands = []
        self.cp_command_index = 0
        self.cp_repeat_count = 1
        self.vlm_hold_timer = None

        self.cover_vision_delay = None
        self.confirm_retry_phase = None
        self.vlm_confirm_pending = False
        self.home_pending = False
        self.vlm_confirm_delay = None

    def start_callback(self, _msg):
        self.plan_initial_pack()

    def control_plan_callback(self, msg):
        data = json.loads(msg.data)
        class_name = str(data['class']).strip()

        self.current_ingredient = class_name
        self.point1_q = None
        self.pick_lift_q = None
        self.cp_repeat_count = max(1, int(data.get('repeat_count', 1)))

        if class_name in {'sauce_tomato', 'sauce_cream', 'sauce_oil'}:
            self.pick_mode = 'sauce'
            self.plan_sauce()
            return

        self.pick_mode = self.select_mode(class_name)

    def control_motion_callback(self, msg):
        command = msg.data.strip()

        if command == 'point1':
            self.move_point1()
            return

        if command == 'place':
            if self.current_ingredient is None:
                return

            position, yaw = self.move_place_pose()
            self.plan_place(position, yaw, self.pick_mode)
            return

        if command == 'cp':
            if self.current_ingredient is None:
                return

            self.cp_commands = self.cp_motion.build_path(self.current_ingredient, self.cp_repeat_count)
            self.cp_command_index = 0
            self.after_cp_path()
            return

        if command == 'vlm_confirm':
            self.move_vlm_confirm()
            return

        if command == 'home':
            self.move_home()
            return

    def arm_motion_done_callback(self, _msg):
        if self.cp_commands:
            self.after_cp_path()
            return

        if self.vlm_confirm_pending:
            self.vlm_confirm_pending = False
            self.vlm_confirm_delay = self.create_timer(
                2.0,
                self.vlm_confirm_delay_done
            )
            return

        if self.confirm_retry_phase == 'sauce':
            self.confirm_retry_phase = None
            self.move_vlm_confirm()
            return

        if self.home_pending:
            self.home_pending = False

            msg = Int16()
            msg.data = 1
            self.control_home_pub.publish(msg)
            return

        if self.confirm_retry_phase == 'point1':
            self.confirm_retry_phase = 'pick'

            if self.current_ingredient == 'cover':
                self.cover_vision_delay = self.create_timer(
                    2.0,
                    self.cover_vision_delay_done
                )
            else:
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

    def cover_vision_delay_done(self):
        self.cover_vision_delay.cancel()
        self.destroy_timer(self.cover_vision_delay)
        self.cover_vision_delay = None

        self.request_vision()

    def vlm_confirm_delay_done(self):
        self.vlm_confirm_delay.cancel()
        self.destroy_timer(self.vlm_confirm_delay)
        self.vlm_confirm_delay = None

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

        if self.pick_mode == 'sauce':
            self.confirm_retry_phase = 'sauce'
            self.plan_sauce()
            self.home_pending = False
            return

        if self.pick_mode == 'cp':
            self.confirm_retry_phase = 'cp'
            self.cp_commands = self.cp_motion.build_path(self.current_ingredient, 1)
            self.cp_command_index = 0
            self.after_cp_path()
            return

        self.confirm_retry_phase = 'point1'
        self.move_point1()

    def pick_callback(self, msg):
        position, yaw, class_name = self.vision_pick_pose(msg)
        mode = self.select_mode(class_name)
        self.pick_mode = mode

        self.plan_pick(position, yaw, mode)

    def vision_request_callback(self, _msg):
        if self.current_ingredient == 'cover':
            self.cover_vision_delay = self.create_timer(
                2.0,
                self.cover_vision_delay_done
            )
            return

        self.request_vision()

    def reset_callback(self, _msg):
        self.get_logger().info('POINT reset 시작')

        cover_delay = self.cover_vision_delay
        self.cover_vision_delay = None

        if cover_delay is not None:
            cover_delay.cancel()
            self.destroy_timer(cover_delay)

        hold_timer = self.vlm_hold_timer
        self.vlm_hold_timer = None

        if hold_timer is not None:
            hold_timer.cancel()
            self.destroy_timer(hold_timer)

        confirm_delay = self.vlm_confirm_delay
        self.vlm_confirm_delay = None

        if confirm_delay is not None:
            confirm_delay.cancel()
            self.destroy_timer(confirm_delay)

        self.current_ingredient = None
        self.pick_mode = None
        self.point1_q = None
        self.pick_lift_q = None

        self.cp_commands = []
        self.cp_command_index = 0
        self.cp_repeat_count = 1

        self.confirm_retry_phase = None
        self.vlm_confirm_pending = False
        self.home_pending = False

        self.get_logger().info('POINT 초기화 완료')

    def request_vision(self):
        msg = String()
        msg.data = self.current_ingredient
        self.control_motion_done.publish(msg)

    def after_cp_path(self):
        if self.cp_command_index >= len(self.cp_commands):
            self.cp_commands = []
            self.cp_command_index = 0

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

        command = self.cp_commands[self.cp_command_index]
        self.cp_command_index += 1
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
            self.vlm_hold_timer = self.create_timer(
                command[1],
                self.cheese_delay_done
            )

    def cheese_delay_done(self):
        self.vlm_hold_timer.cancel()
        self.destroy_timer(self.vlm_hold_timer)
        self.vlm_hold_timer = None
        self.after_cp_path()

    def move_point1(self):
        if self.current_ingredient == 'cover':
            self.point1_q = POINT2.copy()
        else:
            self.point1_q = POINT1.copy()
        
        msg = Float64MultiArray()
        msg.data = self.point1_q.tolist()
        self.joint_target_pub.publish(msg)

    def move_vlm_confirm(self):
        if self.current_ingredient in {'cover', 'sauce_tomato', 'sauce_cream', 'sauce_oil'}:
            q_target = VLM_CONFIRM_POINT2.copy()
        else:
            q_target = VLM_CONFIRM_POINT.copy()

        self.vlm_confirm_pending = True

        msg = Float64MultiArray()
        msg.data = q_target.tolist()
        self.joint_target_pub.publish(msg)

    def move_home(self):
        self.home_pending = True

        if self.current_ingredient == 'cover':
            waypoints = [CONTROL_READY2.copy()]

        elif self.current_ingredient in {'sauce_tomato', 'sauce_cream', 'sauce_oil'}:
            waypoints = [
                CONTROL_READY2.copy(),
                CONTROL_READY.copy()
            ]

        else:
            waypoints = [CONTROL_READY.copy()]

        msg = Float64MultiArray()
        msg.data = np.asarray(waypoints).reshape(-1).tolist()
        self.joint_target_pub.publish(msg)

    @staticmethod
    def select_mode(class_name):
        if class_name in {'cheese', 'pepperoncino'}:
            return 'cp'
        if class_name in {'noodle_thick', 'noodle_thin', 'mushroom', 'onion', 'crab', 'sausage'}:
            return 'grip'
        if class_name in {'cover'}:
            return 'pack'
        if class_name in {'sauce_tomato', 'sauce_cream', 'sauce_oil'}:
            return 'sauce'
        raise ValueError(f'클래스 안맞음: {class_name}')

    def move_place_pose(self):
        class_name = self.current_ingredient
    
        if class_name in {'noodle_thick', 'noodle_thin'}:
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

        q_pick = self.kinematics.solve_pose(position, previous_q, math.pi, 'pack')

        q_lift = self.kinematics.solve_pose(lift_position, q_pick, math.pi, 'pack')

        return q_pick, q_lift

    def plan_pick(self, position, yaw, mode):
        approach = position.copy()
        approach[2] += LIFT_HEIGHT

        if self.point1_q is not None:
            q_motion_start = self.point1_q
        else:
            q_motion_start = np.zeros(DOF, dtype=float)

        if mode == 'pack':
            q_pick, q_lift = self.solve_pack_pick_lift(position, q_motion_start)
            q_approach = q_lift

        else:
            corrected_yaw = self.base_target(position, yaw)

            q_approach = self.kinematics.solve_pose(approach, q_motion_start, corrected_yaw, 'grip')

            q_pick = self.kinematics.solve_pose(position, q_approach, corrected_yaw, 'grip')

            q_lift = self.kinematics.solve_pose(approach, q_pick, corrected_yaw, 'grip')

        self.pick_lift_q = q_lift

        if mode == 'pack':
            publisher = self.pack_plan_pub
            phase = 'pack_pick'
        else:
            publisher = self.grip_plan_pub
            phase = f'grip_pick:{self.current_ingredient}'

        self.publish_waypoints(
            publisher,
            phase,
            q_approach,
            q_pick,
            q_lift
        )

    @staticmethod
    def base_target(position, target_yaw):
        base_yaw = math.atan2(position[1], position[0])

        desired_raw = base_yaw - float(target_yaw)

        desired_yaw = (desired_raw + math.pi) % (2.0 * math.pi) - math.pi

        return desired_yaw

    @staticmethod
    def extended_base(q):
        q = q.copy()

        if q[0] < 0.0:
            q[0] += 2.0 * math.pi

        return q
    
    def plan_place(self, position, yaw, mode):
        
        approach = position.copy()
        approach[2] += LIFT_HEIGHT

        if mode == 'pack':

            if self.current_ingredient == 'cover':
                q_previous = self.pick_lift_q.copy()
                q_previous[0] = np.deg2rad(360.0)
            else:
                q_previous = self.pick_lift_q

            q_approach = self.kinematics.solve_pose(approach, q_previous, math.pi, 'pack')

            q_place = self.kinematics.solve_pose(position, q_approach, math.pi, 'pack')

            q_lift = q_approach.copy()

            if self.current_ingredient == 'cover':

                q_approach = self.extended_base(q_approach)
                q_place = self.extended_base(q_place)
                q_lift = q_approach.copy()

                self.publish_waypoints(
                    self.pack_plan_pub,
                    'pack_place',
                    q_previous,
                    q_approach,
                    q_place,
                    q_lift
                )

                return

        else:
            corrected_yaw = self.base_target(position, yaw)

            q_approach = self.kinematics.solve_grip_place_pose(
                approach, self.pick_lift_q, corrected_yaw
            )

            q_place = self.kinematics.solve_grip_place_pose(
                position, q_approach, corrected_yaw
            )

            q_lift = self.kinematics.solve_grip_place_pose(
                approach, q_place, corrected_yaw
            )

        if mode == 'pack':
            phase = 'pack_place'
        else:
            phase = f'grip_place:{self.current_ingredient}'

        self.publish_waypoints(
            self.pack_plan_pub
            if mode == 'pack'
            else self.grip_plan_pub,
            phase,
            q_approach,
            q_place,
            q_lift
        )

    def plan_initial_pack(self):
        p1 = INITIAL_PACK_PICK_POINT
        p2 = INITIAL_PACK_PLACE_POINT

        p2_lift = p2.copy()
        p2_lift[2] = p1[2] + LIFT_HEIGHT

        q_home = np.zeros(DOF, dtype=float)
        q_p1, q_p1_lift = self.solve_pack_pick_lift(p1, q_home)
        q_p2_lift = self.kinematics.solve_pose(p2_lift, q_p1_lift, math.pi, 'pack')
        q_p2 = self.kinematics.solve_pose(p2, q_p2_lift, math.pi, 'pack')

        self.publish_waypoints(
            self.pack_plan_pub,
            'pack_full',
            q_p1,
            q_p1_lift,
            q_p2_lift,
            q_p2
        )

    def plan_sauce(self):
        q_start = CONTROL_READY2.copy()
        if self.current_ingredient == 'sauce_tomato':
            pick = TOMATO_PICK_POINT.copy()

        elif self.current_ingredient == 'sauce_cream':
            pick = CREAM_PICK_POINT.copy()

        elif self.current_ingredient == 'sauce_oil':
            pick = OIL_PICK_POINT.copy()

        pick_lift = pick.copy()
        pick_lift[2] += LIFT_HEIGHT

        place = SAUCE_PLACE_POINT

        place_lift = place.copy()
        place_lift[2] = pick[2] + LIFT_HEIGHT

        q_pick_lift = self.kinematics.solve_pose(pick_lift, q_start, math.pi, 'pack')

        q_pick = self.kinematics.solve_pose(pick, q_pick_lift, math.pi, 'pack')

        q_place_lift = self.kinematics.solve_pose(place_lift, q_pick_lift, math.pi, 'pack')

        q_place = self.kinematics.solve_pose(place, q_place_lift, math.pi, 'pack')

        q_place_lift = self.extended_base(q_place_lift)
        q_place = self.extended_base(q_place)

        self.publish_waypoints(
            self.pack_plan_pub,
            'sauce_full',
            q_pick,
            q_pick_lift,
            q_place_lift,
            q_place
        )

    @staticmethod
    def publish_waypoints(publisher, phase, *waypoints):
        count = len(waypoints)

        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(
                label=phase,
                size=count,
                stride=DOF * count
            )
        ]
        msg.data = np.concatenate(waypoints).tolist()
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