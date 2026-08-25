#!/usr/bin/env python3

import json
import math
import numpy as np
import rclpy
from ikpy.chain import Chain
from ikpy.link import Link, OriginLink
from rclpy.node import Node
from scipy.optimize import least_squares
from std_msgs.msg import Empty, Float64MultiArray, MultiArrayDimension, String, Int16

DOF = 6
DH_D = np.array([0.1271, 0.0, 0.22000, 0.0, 0.0, 0.188046], dtype=float)
DH_A = np.array([0.0, 0.0, 0.0, 0.220, 0.0, 0.0], dtype=float)
DH_ALPHA = np.deg2rad([-90.0, 90.0, -90.0, 0.0, 90.0, 0.0])
DH_THETA = np.deg2rad([0.0, 0.0, 0.0, -90.0, 90.0, 0.0])

## 공압은 5번 모터 DH 계산에 회전행렬이 필요없기 때문에 DH를 따로 정의
PACK_DH_THETA = DH_THETA.copy()
PACK_DH_THETA[4] = 0.0
PACK_DH_ALPHA = DH_ALPHA.copy()
PACK_DH_ALPHA[4] = 0.0 

PACK_POSITION_WEIGHT = 100.0
PACK_Q5_LIMIT_WEIGHT = 100.0
PACK_CONTINUITY_WEIGHT = 0.02

SPOON_PICK_POSITION = np.array([0.4, 0.01, 0.265], dtype=float)

SPOON_Q6 = math.radians(90.0)
CHEESE_APPROACH_Q = np.deg2rad([-13.0, 5.0, 0.0, 108.0, 70.0, 0.0]) # 치즈 접근 위치
PEPPERONCINO_APPROACH_Q = np.deg2rad([-8.0, 30.0, 0.0, 63.0, 85.0, 0.0]) # 페퍼론치노 접근 위치
CHEESE_PLACE_Q = np.deg2rad([130.0, -90.0, -90.0, 130.0, 20.0, 0.0,]) # 치즈 place 위치
PEPPERONCINO_PLACE_Q = np.deg2rad([140.0, -90.0, -90.0, 130.0, 25.0, 0.0,]) # 페퍼론치노 place 위치
CHEESE_RELEASE_HOLD_TIME = 1.5

## 공압으로 최대한 가까이, 낮게 잡을 수 있는 위치: [0.23, 0.0, 0.065], *base x = 7
# [0.20, 0.00, 0.27] -> 카메라가 수직으로 보는 위치
POINT1 = np.array([0.25, 0.00, 0.27], dtype=float) ## 카메라를 수직으로 바라보는 위치

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

JOINT_MIN = np.deg2rad([-170.0, -120.0, -170.0, -140.0, -120.0, -360.0])
JOINT_MAX = np.deg2rad([170.0, 120.0, 170.0, 140.0, 120.0, 360.0])

STATE_IDLE = '시작!'
STATE_WAIT_INITIAL_DONE = '용기 옮기기'
STATE_WAIT_POINT1_DONE = '카메라-통 수직 위치 이동'
STATE_WAIT_PICK = 'PICK 좌표값 송수신'
STATE_WAIT_PICK_DONE = 'PICK'
STATE_WAIT_PLACE_DONE = 'PLACE'
STATE_WAIT_HOME_DONE = 'HOME 이동'
STATE_WAIT_CP_DONE = '추가 재료 경로 이동'


def wrap_to_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

def wrapped_q_delta(q_goal, q_start):
    return (q_goal - q_start + np.pi) % (2.0 * np.pi) - np.pi

def dh_matrix(theta, d, a, alpha):
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)

class DHLink(Link):

    def __init__(self, name, d, a, alpha, theta, bounds):
        super().__init__(
            name=name,
            length=max(abs(float(d)), abs(float(a)), 1.0e-9),
            bounds=bounds,
        )
        self.d = float(d)
        self.a = float(a)
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.joint_type = 'revolute' ## 회전 관절
        self.rotation = True
        self.translation = False

    def get_link_frame_matrix(self, actuator_parameter):
        return dh_matrix(float(actuator_parameter) + self.theta, self.d, self.a, self.alpha)


class PointPoseNode(Node):

    def __init__(self):
        super().__init__('point_pose_node')

        self.preferred_chain = self.make_chain((1, 3, 4))
        self.fallback_chain = self.make_chain((1, 2, 3, 4))
        self.fk_chain = self.make_chain(tuple())

        self.grip_plan_pub = self.create_publisher(Float64MultiArray, '/arm/joint_waypoints', 10)
        self.pack_plan_pub = self.create_publisher(Float64MultiArray, '/arm/joint_waypoints_pack', 10)
        self.joint_target_pub = self.create_publisher(Float64MultiArray, '/arm/joint_target', 10)
        self.create_subscription(Int16, '/control/start', self.start_callback, 10)
        self.create_subscription(String, '/control/plan', self.control_plan_callback, 10)
        self.control_ready_pub = self.create_publisher(Int16, '/control/ready', 10)
        self.control_home_pub = self.create_publisher(Int16, '/control/home', 10)
        self.control_motion_done = self.create_publisher(String, '/control/motion_done', 10)
        self.create_subscription(String, '/vision/pick_pose', self.pick_callback, 10)
        self.create_subscription(Empty, '/arm/motion_done', self.motion_done_callback, 10)

        self.state = STATE_IDLE
        self.current_ingredient = None
        self.repeat_total = 0
        self.repeat_completed = 0   
        self.pick_mode = None
        self.point1_q = None
        self.pick_lift_q = None

        self.cheese_commands = []
        self.cheese_command_index = 0
        self.cheese_delay_timer = None

    def start_callback(self, _msg):
        if self.state != STATE_IDLE:
            return

        self.state = STATE_WAIT_INITIAL_DONE
        self.plan_initial_pack()

    def control_plan_callback(self, msg):
        if self.state != STATE_IDLE:
            return

        data = json.loads(msg.data)
        class_name = str(data['class']).strip()
        repeat_count = int(data['repeat_count'])
        mode = self.select_mode(class_name)

        self.current_ingredient = class_name
        self.repeat_total = repeat_count
        self.repeat_completed = 0
        self.pick_mode = mode
        self.pick_lift_q = None

        if mode == 'cp':
            self.state = STATE_WAIT_CP_DONE

            self.cheese_commands = self.cp_path(class_name)
            self.cheese_command_index = 0
            self.after_cp_path()

            return

        self.move_point(POINT1, STATE_WAIT_POINT1_DONE)

    def motion_done_callback(self, _msg):

        if self.state == STATE_WAIT_CP_DONE:
            self.after_cp_path()
            return

        if self.state == STATE_WAIT_INITIAL_DONE:
            self.state = STATE_IDLE

            msg = Int16()
            msg.data = 1
            self.control_ready_pub.publish(msg)
            return

        if self.state == STATE_WAIT_POINT1_DONE:
            self.state = STATE_WAIT_PICK

            # LLM에서 전달받은 클래스 전송
            msg = String()
            msg.data = self.current_ingredient
            self.control_motion_done.publish(msg)
            return

        if self.state == STATE_WAIT_PICK_DONE:
            place_position, place_yaw = self.move_place_pose()

            self.state = STATE_WAIT_PLACE_DONE
            self.plan_place(
                place_position,
                place_yaw,
                self.pick_mode
            )

            return

        if self.state == STATE_WAIT_PLACE_DONE:
            self.repeat_completed += 1
            self.pick_mode = None
            self.pick_lift_q = None

            if self.repeat_completed < self.repeat_total:
                self.move_point(
                    POINT1,
                    STATE_WAIT_POINT1_DONE
                )
            else:
                msg = Float64MultiArray()
                msg.data = [0.0] * DOF
                self.state = STATE_WAIT_HOME_DONE
                self.joint_target_pub.publish(msg)

            return

        if self.state == STATE_WAIT_HOME_DONE:
            self.current_ingredient = None
            self.repeat_total = 0
            self.repeat_completed = 0
            self.pick_mode = None
            self.point1_q = None
            self.pick_lift_q = None

            self.state = STATE_IDLE

            msg = Int16()
            msg.data = 1
            self.control_home_pub.publish(msg)
            return

    def pick_callback(self, msg):
        if self.state != STATE_WAIT_PICK:
            return

        position, yaw, class_name = self.vision_pick_pose(msg)
        mode = self.select_mode(class_name)

        self.pick_mode = mode
        self.state = STATE_WAIT_PICK_DONE

        self.plan_pick(position, yaw, mode)

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

            q = self.solve_cp_path(position, q_previous, q6)

            waypoints.append(q)
            q_previous = q.copy()

        return waypoints

    def cp_path(self, class_name):
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
            q_cheese_approach_lift = CHEESE_APPROACH_Q.copy()
            q_cheese_approach_lift[3] += math.radians(-20)
            q_cheese_approach_lift[4] += math.radians(20)
            q_cheese_approach = CHEESE_APPROACH_Q.copy()

        else: 
            q_cheese_approach_lift = PEPPERONCINO_APPROACH_Q.copy()
            q_cheese_approach_lift[3] += math.radians(-20)
            q_cheese_approach_lift[4] += math.radians(20)
            q_cheese_approach = PEPPERONCINO_APPROACH_Q.copy()

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
        q_spoon_approach = self.solve_cp_path(spoon_approach_position, q_start, SPOON_Q6)

        q_spoon_pick_path = self.spoon_linear_x(spoon_approach_position, SPOON_PICK_POSITION, q_spoon_approach, SPOON_Q6, step=0.01)
        q_spoon_pick = q_spoon_pick_path[-1]

        # 숟가락 잡은 후 lift
        q_spoon_lift = self.solve_cp_path(spoon_lift_position, q_spoon_pick, SPOON_Q6) 

        # 치즈 / 페페론치노 place 준비
        if self.current_ingredient == 'cheese':
            q_cheese_place_ready = CHEESE_PLACE_Q.copy()
       
        else: PEPPERONCINO_PLACE_Q.copy()

        # 치즈, 페퍼론치노 place
        q_cheese_release_1 = q_cheese_place_ready.copy()
        q_cheese_release_1[5] += math.radians(-110.0)

        q_cheese_release_2 = q_cheese_release_1.copy()
        q_cheese_release_2[5] += math.radians(110.0)

        # 숟가락 내려놓고 x축 빠지기
        q_spoon_retreat_path = self.spoon_linear_x(SPOON_PICK_POSITION, spoon_approach_position, q_spoon_pick, SPOON_Q6, step=0.01)

        return [
            # POINT1 -> 숟가락 x축 -이동 -> 숟가락 위치
            ('motion', [q_spoon_approach, *q_spoon_pick_path]),

            # 숟가락 잡기
            ('gripper', 'grip_pick:spoon', q_spoon_pick),

            # 숟가락 lift -> 치즈 푸기 -> 치즈 놓기
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

            # 치즈가 떨어질 때까지 대기
            ('delay', CHEESE_RELEASE_HOLD_TIME),

            # 숟가락 lift 위치 -> 숟가락 위치
            ('motion', [q_spoon_lift, q_spoon_pick]),

            # 숟가락 내려놓기
            ('gripper', 'grip_place:spoon', q_spoon_pick),

            # 숟가락 위치에서 x축 -로 빠지기
            ('motion', q_spoon_retreat_path),

            # HOME 복귀
            ('motion', [q_home]),
        ]

    def after_cp_path(self):
        if self.cheese_command_index >= len(self.cheese_commands):
            self.cheese_commands = []
            self.repeat_completed += 1

            if self.repeat_completed < self.repeat_total:
                self.cheese_commands = self.cp_path(self.current_ingredient)
                self.cheese_command_index = 0
                self.after_cp_path()
            else:
                self.current_ingredient = None
                self.repeat_total = 0
                self.repeat_completed = 0
                self.pick_mode = None
                self.point1_q = None
                self.pick_lift_q = None
                self.state = STATE_IDLE

                msg = Int16()
                msg.data = 1
                self.control_home_pub.publish(msg)

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

    def solve_cp_path(self, position, previous_q, q6):
        self.get_logger().info(f'cp 경로 시작: target={position.tolist()}')

        q1_target = self.target_q1(position, previous_q[0])
        active_indices = np.array([1, 2, 3], dtype=int)
        candidates = []

        for seed in self.seeds(previous_q, q1_target):
            def build_q(active_q):
                q = previous_q.copy()
                q[0] = q1_target
                q[active_indices] = active_q
                q[4] = self.horizontal_q5(q[1], q[2], q[3], previous_q[4])
                q[5] = q6
                return q

            def residual(active_q):
                q = build_q(active_q)
                position_error = self.fk_matrix(q)[:3, 3] - position
                joint_delta = wrapped_q_delta(q[active_indices], previous_q[active_indices])
                return np.concatenate([100.0 * position_error, 0.02 * joint_delta])

            result = least_squares(
                residual,
                seed[active_indices],
                bounds=(JOINT_MIN[active_indices], JOINT_MAX[active_indices]),
                max_nfev=300
            )

            q = build_q(result.x)
            position_error = float(np.linalg.norm(self.fk_matrix(q)[:3, 3] - position))
            joint_motion = float(np.linalg.norm(wrapped_q_delta(q, previous_q)))
            candidates.append((position_error, joint_motion, q))

        if not candidates:
            raise RuntimeError('IK 자세 계산 실패')

        selected = min(candidates, key=lambda item: (item[0], item[1]))

        self.get_logger().info(
            f'cp IK 결과: target={position.tolist()}, '
            f'error={selected[0]:.4f}m, '
            f'q={np.rad2deg(selected[2]).round(1).tolist()}'
        )

        if selected[0] > 0.005:
            return

        return selected[2]

    def horizontal_q5(self, q2, q3, q4, reference_q5):
        base_q5 = math.atan2(
                math.cos(q2),
                math.sin(q2) * math.cos(q3)
            )- q4

        candidates = []

        for branch in (0.0, math.pi):
            for turn in (-2.0 * math.pi, 0.0, 2.0 * math.pi):
                candidate = base_q5 + branch + turn

                if (JOINT_MIN[4] <= candidate <= JOINT_MAX[4]):
                    candidates.append(candidate)

        if not candidates:
            return float(np.clip(
                base_q5,
                JOINT_MIN[4],
                JOINT_MAX[4]
            ))

        return float(min(
            candidates,
            key=lambda value: abs(value - reference_q5)
        ))

    def solve_grip_place_pose(self, target, previous_q, place_yaw):
        q = self.solve_point(target, previous_q)

        ##### Place할 때 q1 회전량만큼 목표 yaw 보정 -> 하면서 안맞으면 부호 바꾸기  #####
        corrected_yaw = wrap_to_pi(float(place_yaw) + float(q[0]))

        # 보정된 yaw를 기준으로 q6 계산
        q[5] = self.target_q6(
            corrected_yaw,
            q,
            previous_q[5]
        )

        return q

    def move_point(self, point, next_state):
        try:
            q_target = self.solve_pose(
                point,
                np.zeros(DOF, dtype=float),
                math.pi,
                'grip'
            )
        except RuntimeError:
            return

        if np.allclose(point, POINT1):
            self.point1_q = q_target

        msg = Float64MultiArray()
        msg.data = q_target.tolist()
        self.state = next_state
        self.joint_target_pub.publish(msg)

    @staticmethod
    def select_mode(class_name):
        if class_name in {'cheese', 'pepperoncino'}:
            return 'cp'
        if class_name in {'noodle', 'mushroom', 'onion', 'crab', 'sausage'}:
            return 'grip'
        if class_name in {'cover', 'tomato', 'cream', 'oil'}:
            return 'pack'
        raise ValueError(f'클래스 안맞음: {class_name}')

    def move_place_pose(self):
        class_name = self.current_ingredient
        if class_name in {'tomato', 'cream', 'oil'}:
            place_key = 'sause'
        else: place_key = class_name

        place_pose = PLACE_POINTS[place_key]
        position = place_pose['position']
        yaw = math.radians(float(place_pose['yaw_deg']))

        return position, yaw

    @staticmethod
    def vision_pick_pose(msg):
        try:
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

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return

        return position, yaw, class_name

    ## 공압 IK에서는 q3을 홈 각도(180)로 강제
    def solve_pack_pick_lift(self, position, previous_q):
        lift_position = position.copy()
        lift_position[2] += LIFT_HEIGHT

        q_pick = self.solve_pose(
            position,
            previous_q,
            math.pi,
            'pack'
        )

        q_lift = self.solve_pose(
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
            q_approach = self.solve_pose(
                approach,
                q_motion_start,
                yaw,
                'grip'
            )

            q_pick = self.solve_pose(
                position,
                q_approach,
                yaw,
                'grip'
            )

            q_lift = self.solve_pose(
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
            q_approach = self.solve_pose(
                approach,
                self.pick_lift_q,
                math.pi,
                'pack'
            )

            q_place = self.solve_pose(
                position,
                q_approach,
                math.pi,
                'pack'
            )

            q_lift = q_approach

        else:
            q_approach = self.solve_grip_place_pose(approach, self.pick_lift_q, yaw)

            q_place = self.solve_grip_place_pose(position, q_approach, yaw)

            q_lift = self.solve_grip_place_pose(approach, q_place, yaw)

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
        q_p2_lift = self.solve_pose(
            p2_lift,
            q_p1_lift,
            math.pi,
            'pack'
        )
        q_p2 = self.solve_pose(
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

    def solve_pose(self, target, previous_q, yaw, mode):
        if mode == 'pack':
            q = self.solve_pack_point(target, previous_q)

            q[2] = 0 # 고정
            q[5] = previous_q[5]
            return q

        q = self.solve_point(target, previous_q)
        q[5] = self.target_q6(yaw, q, previous_q[5])
        return q

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

    def make_chain(self, active_joints):
        links = [OriginLink()]
        active_mask = [False]

        for i in range(DOF):
            links.append(
                DHLink(
                    name=f'joint_{i + 1}',
                    d=DH_D[i],
                    a=DH_A[i],
                    alpha=DH_ALPHA[i],
                    theta=DH_THETA[i],
                    bounds=(float(JOINT_MIN[i]), float(JOINT_MAX[i]))
                )
            )
            active_mask.append(i in active_joints)

        return Chain(
            name='irc_dh',
            links=links,
            active_links_mask=active_mask
        )

    @staticmethod
    def full_q(q):
        return np.concatenate(([0.0], q))

    def fk_matrix(self, q):
        return self.fk_chain.forward_kinematics(self.full_q(q))

    def pack_fk_q5(self, q):
        transform = np.eye(4, dtype=float)
        for i in range(5):
            transform = transform @ dh_matrix(
                q[i] + PACK_DH_THETA[i],
                DH_D[i],
                DH_A[i],
                PACK_DH_ALPHA[i]
            )
        return transform

    def pack_fk(self, q):
        transform = self.pack_fk_q5(q)

        return (
            transform[:3, 3] + transform[:3, :3] @ np.array([0.080845, 0.09195, 0.0], dtype=float) ## 5번 모터에서 공압 끝까지 translation 적용
        )

    @staticmethod
    def pack_q5(q2, q3, q4, branch, reference_q5):
        link_angle = math.atan2(math.cos(q2), math.sin(q2) * math.cos(q3))
        raw_q5 = link_angle + branch - q4
        return float(reference_q5 + wrap_to_pi(raw_q5 - reference_q5))

    def pack_seeds(self, previous_q, q1): ## 공압 자세 휴리스틱 후보 계산
        primary_postures = [
            (20.0, 20.0),
            (35.0, 30.0),
            (50.0, 40.0),
            (65.0, 55.0),
            (80.0, 65.0),
        ]

        result = []

        # q1 = 목표 방향
        seed = previous_q.copy()
        seed[0] = q1
        seed[2] = 0

        result.append(np.clip(seed, JOINT_MIN, JOINT_MAX))

        for q2_deg, q4_deg in primary_postures:
            seed = previous_q.copy()

            seed[0] = q1
            seed[1] = math.radians(q2_deg)
            seed[2] = 0
            seed[3] = math.radians(q4_deg)

            result.append(np.clip(seed, JOINT_MIN, JOINT_MAX))
        return result

    def solve_pack_point(self, target, previous_q):
        q1_target = self.target_q1(target, previous_q[0])
        candidates = []

        for seed in self.pack_seeds(previous_q, q1_target):
            for branch in (0.0, math.pi):

                def build_q(active_q):
                    q = previous_q.copy()

                    # 공압을 사용할 때 해 결정 변수는 q1, q2, q4만 사용
                    q[np.array([0, 1, 3], dtype=int)] = active_q
                    q[2] = 0 # 고정
                    q[4] = self.pack_q5(
                        q[1],
                        q[2],
                        q[3],
                        branch,
                        previous_q[4]
                    )
                    return q

                def residual(active_q):
                    q = build_q(active_q)

                    q5_low = max(0.0, JOINT_MIN[4] - q[4])
                    q5_high = max(0.0, q[4] - JOINT_MAX[4])
                    joint_delta = wrapped_q_delta(
                        q[np.array([0, 1, 3], dtype=int)],
                        previous_q[np.array([0, 1, 3], dtype=int)]
                    )

                    q1_target_error = wrap_to_pi(q[0] - q1_target)

                    q2_backward = max(0.0, -q[1])
                    q4_backward = max(0.0, -q[3])

                    continuity_weights = np.array([0.25, 1.0, 1.0])

                    return np.concatenate([
                        PACK_POSITION_WEIGHT * (self.pack_fk(q) - target),

                        np.array([
                            PACK_Q5_LIMIT_WEIGHT * q5_low,
                            PACK_Q5_LIMIT_WEIGHT * q5_high,
                        ]),

                        PACK_CONTINUITY_WEIGHT * continuity_weights * joint_delta,

                        np.array([
                            2.0 * PACK_CONTINUITY_WEIGHT * q1_target_error,
                            2.0 * PACK_CONTINUITY_WEIGHT * q2_backward,
                            2.0 * PACK_CONTINUITY_WEIGHT * q4_backward,
                        ]),
                    ])

                result = least_squares(
                    residual,
                    seed[np.array([0, 1, 3], dtype=int)],
                    bounds=(
                        JOINT_MIN[np.array([0, 1, 3], dtype=int)],
                        JOINT_MAX[np.array([0, 1, 3], dtype=int)]
                    ),
                    max_nfev=300 ## 최적화 해 계산 300번 제한
                )

                q = build_q(result.x)

                if not (JOINT_MIN[4] <= q[4] <= JOINT_MAX[4]):
                    continue

                position_error = float(np.linalg.norm(self.pack_fk(q) - target))
                horizontal_error = abs(self.pack_fk_q5(q)[2, 0])

                joint_delta = wrapped_q_delta(
                    q[np.array([0, 1, 3], dtype=int)],
                    previous_q[np.array([0, 1, 3], dtype=int)]
                )

                q1_direction_error = abs(wrap_to_pi(q[0] - q1_target))

                q2_backward = max(0.0, -q[1])
                q4_backward = max(0.0, -q[3])

                total_motion = float(np.linalg.norm(joint_delta))

                # 위치 오차가 비슷한 후보끼리 비교할 자세 점수
                posture_score = (
                    3.0 * q1_direction_error
                    + 2.0 * q2_backward
                    + 2.0 * q4_backward
                    + total_motion
                )

                candidates.append((
                    position_error,
                    horizontal_error,
                    posture_score,
                    total_motion,
                    q
                ))

        if not candidates:
            return

        min_error = min(item[0] for item in candidates)

        selected = min(
            (
                item
                for item in candidates
                if item[0] <= min_error + 0.002 ## IK 후보 여유값
            ),
            key=lambda item: (
                item[1],  # 공압 링크 수평 오차
                item[2],  # q1 방향, q3 유지, 전방 자세 점수
                item[3],  # 전체 관절 이동량
            )
        )

        if selected[0] > 0.005:
            raise RuntimeError(f'IK가 닿을 수 없는 곳에 있음: {selected[0]:.4f} m')  ## tolerance 검사

        return selected[4] 
    
    def target_q6(self, target_yaw, q, previous_q6):
        q_without_q6 = q.copy()
        q_without_q6[5] = 0.0

        rotation = self.fk_matrix(q_without_q6)[:3, :3]
        current_yaw = math.atan2(
            float(rotation[1, 0]),
            float(rotation[0, 0])
        )
        desired = wrap_to_pi(target_yaw - current_yaw)

        candidates = np.array([
            desired - 2.0 * math.pi,
            desired,
            desired + 2.0 * math.pi,
        ])
        valid = candidates[(candidates >= JOINT_MIN[5]) & (candidates <= JOINT_MAX[5])]

        if valid.size == 0:
            return float(np.clip(
                desired,
                JOINT_MIN[5],
                JOINT_MAX[5]
            ))

        return float(valid[np.argmin(np.abs(valid - previous_q6))])

    def target_q1(self, target, previous_q1):
        xy = target[:2]

        if np.linalg.norm(xy) < 1.0e-8:
            return float(previous_q1)

        desired = wrap_to_pi(math.atan2(float(xy[1]), float(xy[0])))
        candidates = np.array([
            desired - 2.0 * math.pi,
            desired,
            desired + 2.0 * math.pi,
        ])
        valid = candidates[(candidates >= JOINT_MIN[0]) & (candidates <= JOINT_MAX[0])]

        if valid.size == 0:
            return float(np.clip(
                desired,
                JOINT_MIN[0], JOINT_MAX[0]
            ))

        return float(valid[np.argmin(np.abs(valid - previous_q1))])

    def seeds(self, previous_q, q1): ## 일반 그리파 휴리스틱 자세 후보 계산
        postures = [
            (20.0, 20.0, 15.0),
            (35.0, 30.0, 25.0),
            (50.0, 40.0, 35.0),
            (65.0, 55.0, 45.0),
            (45.0, -30.0, 70.0),
            (60.0, -50.0, 90.0),
        ]

        result = []
        seed = previous_q.copy()
        seed[0] = q1
        result.append(seed)

        for q2_deg, q4_deg, q5_deg in postures:
            seed = previous_q.copy()
            seed[0] = q1
            seed[1] = math.radians(q2_deg)
            seed[3] = math.radians(q4_deg)
            seed[4] = math.radians(q5_deg)
            result.append(np.clip(seed, JOINT_MIN, JOINT_MAX))
        return result

    def candidates(self, chain, target, previous_q, q1):
        result = []

        for seed in self.seeds(previous_q, q1):
            full = chain.inverse_kinematics(
                target_position=target,
                initial_position=self.full_q(seed),
                max_iter=100
            )
            q = np.clip(full[1:], JOINT_MIN, JOINT_MAX)
            q[0] = q1
            q[5] = previous_q[5]

            error = float(np.linalg.norm(target - self.fk_matrix(q)[:3, 3]))
            q3_use = abs(wrap_to_pi(q[2] - previous_q[2]))
            continuity = float(np.linalg.norm(
                wrapped_q_delta(q[[1, 3, 4]], previous_q[[1, 3, 4]])
            ))
            wrong_q2 = max(0.0, -float(q[1])) ** 2
            cost = (
                100.0 * q3_use
                + 2.0 * continuity
                + 15.0 * wrong_q2
                + 0.3 * float(np.linalg.norm(q[[1, 3, 4]]))
            )
            result.append((error, cost, q))

        return result

    @staticmethod
    def best(candidates):
        min_error = min(item[0] for item in candidates)
        return min(
            (
                item
                for item in candidates
                if item[0] <= min_error + 0.002
            ),
            key=lambda item: item[1]
        )

    def solve_point(self, target, previous_q):
        q1 = self.target_q1(target, previous_q[0])
        preferred = self.best(
            self.candidates(
                self.preferred_chain,
                target,
                previous_q,
                q1
            )
        )

        if preferred[0] <= 0.005:
            selected = preferred
        else:
            fallback = self.best(
                self.candidates(
                    self.fallback_chain,
                    target,
                    previous_q,
                    q1
                )
            )
            selected = (
                fallback
                if fallback[0] + 0.002 < preferred[0]
                else preferred
            )

        if selected[0] > 0.005:
            raise RuntimeError(f'IK 위치가 닿을 수 없는 곳에 있음: {selected[0]:.4f} m')
        return selected[2]


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