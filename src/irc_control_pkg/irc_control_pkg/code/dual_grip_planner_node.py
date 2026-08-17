#!/usr/bin/env python3

import json
import math

import numpy as np
import rclpy
from ikpy.chain import Chain
from ikpy.link import Link, OriginLink
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.optimize import least_squares
from std_msgs.msg import Empty, Float64MultiArray, MultiArrayDimension, String

DOF = 6

DH_D = np.array([
    0.1271, 0.0, 0.22000, 0.0, 0.0, 0.188046,
], dtype=float)

DH_A = np.array([
    0.0, 0.0, 0.0, 0.220, 0.0, 0.0,
], dtype=float)

DH_ALPHA = np.deg2rad([
    -90.0, 90.0, -90.0, 0.0, 90.0, 0.0,
])

DH_THETA_OFFSET = np.deg2rad([
    0.0, 0.0, 0.0, -90.0, 90.0, 0.0,
])

## 공압은 5번 모터에서 회전행렬이 필요없기 때문에 DH를 따로 정의해줘야 됨
PACK_DH_THETA_OFFSET = DH_THETA_OFFSET.copy()
PACK_DH_THETA_OFFSET[4] = 0.0
PACK_DH_ALPHA = DH_ALPHA.copy()
PACK_DH_ALPHA[4] = 0.0 

PREFERRED_ACTIVE = (1, 3, 4)
FALLBACK_ACTIVE = (1, 2, 3, 4)

Q_FULL_START = 2
FULL_Q_SIZE = 9

IK_MAX_ITER = 100
PREFERRED_IK_TOLERANCE_M = 0.004
IK_TOLERANCE_M = 0.005
IK_ERROR_MARGIN_M = 0.002

PACK_MAX_NFEV = 300
PACK_POSITION_WEIGHT = 100.0
PACK_Q5_LIMIT_WEIGHT = 100.0
PACK_CONTINUITY_WEIGHT = 0.02

'''
여기서 POINT1은 재료를 수직을 바라보는 위치, POINT2는 용기를 바라보는 위치
INITIAL_PACK_PICK_POINT랑 INITIAL_PACK_PLACE_POINT는 맨 처음 용기 p2p 위치
FIXED_PLACE_POINTS는 각 클래스 별로 놓을 place 위치
공압으로 최대한 가까이, 낮게 잡을 수 있는 위치: [0.23, 0.0, 0.065] 이 정도..
'''
POINT1 = np.array([0.20, 0.00, 0.27], dtype=float)
POINT2 = np.array([0.20, 0.00, 0.27], dtype=float)

INITIAL_PACK_PICK_POINT = np.array([0.0, -0.3, 0.1], dtype=float)
INITIAL_PACK_PLACE_POINT = np.array([0.03, 0.23, 0.065], dtype=float)

FIXED_PLACE_POINTS = {
    'noodle': {'position': np.array([0.03, 0.3, 0.07], dtype=float), 'yaw_deg': 180.0,},
    '소스': {'position': np.array([0.03, 0.3, 0.07], dtype=float), 'yaw_deg': 180.0,},  ##yaw 고정
    '버섯': {'position': np.array([0.03, 0.3, 0.07], dtype=float), 'yaw_deg': 180.0,},
    '양파': {'position': np.array([0.03, 0.3, 0.07], dtype=float), 'yaw_deg': 180.0,},
    'crab': {'position': np.array([0.05, 0.25, 0.07], dtype=float), 'yaw_deg': 180.0,},
    'sausage': {'position': np.array([-0.035, 0.25, 0.07], dtype=float), 'yaw_deg': 180.0,},
    '뚜껑': {'position': np.array([0.03, 0.3, 0.1], dtype=float), 'yaw_deg': 180.0,},  ##yaw 고정
}

DEFAULT_YAW_RAD = math.pi

GRIP_CLASSES = {
    'noodle',
    '버섯',
    '양파',
    'crab',
    'sausage',
}

PACK_CLASSES = {
    '뚜껑',
    '소스1',
    '소스2',
    '소스3',
}

STATE_IDLE = '시작!'
STATE_WAIT_INITIAL_DONE = '용기 옮기기 완료'
STATE_WAIT_LLM_TASK = '재료 및 반복 횟수'
STATE_WAIT_RAIL_DONE = '레일 이동 완료'
STATE_WAIT_POINT1_DONE = '카메라-통 수직 위치 이동 완료'
STATE_WAIT_PICK = 'PICK 좌표값 보내주세용'
STATE_WAIT_PICK_DONE = 'PICK 완료'
STATE_WAIT_PLACE_DONE = 'PLACE_완료'
STATE_WAIT_HOME_DONE = 'HOME 이동 완료'


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


class FixedLink(Link):

    def __init__(self, name, transform):
        transform = np.asarray(transform, dtype=float)
        super().__init__(
            name=name,
            length=max(float(np.linalg.norm(transform[:3, 3])), 1.0e-9),
            bounds=(0.0, 0.0),
        )
        self.transform = transform
        self.joint_type = 'fixed'
        self.has_rotation = False
        self.has_translation = False

    def get_link_frame_matrix(self, actuator_parameter):
        return self.transform.copy()


class DHLink(Link):

    def __init__(self, name, d, a, alpha, theta_offset, bounds):
        super().__init__(
            name=name,
            length=max(abs(float(d)), abs(float(a)), 1.0e-9),
            bounds=bounds,
        )
        self.d = float(d)
        self.a = float(a)
        self.alpha = float(alpha)
        self.theta_offset = float(theta_offset)
        self.joint_type = 'revolute'
        self.has_rotation = True
        self.has_translation = False

    def get_link_frame_matrix(self, actuator_parameter):
        return dh_matrix(
            float(actuator_parameter) + self.theta_offset,
            self.d,
            self.a,
            self.alpha,
        )


class PointPosePlannerNode(Node):

    def __init__(self):
        super().__init__('point_pose_planner_node')

        self.declare_parameter('lift_height', 0.15)
        self.declare_parameter('pack_tcp_offset', [0.080845, 0.09195, 0.0])
        self.declare_parameter('pack_position_tolerance_m', 0.005)
        self.declare_parameter('pack_horizontal_tolerance_deg', 0.5)
        self.declare_parameter('joint_min_deg', [-170.0, -120.0, -170.0, -140.0, -120.0, -170.0])
        self.declare_parameter('joint_max_deg', [170.0, 120.0, 170.0, 140.0, 120.0, 170.0])

        self.lift_height = float(
            self.get_parameter('lift_height').value
        )
        self.pack_tcp_offset = np.asarray(
            self.get_parameter('pack_tcp_offset').value,
            dtype=float
        )
        self.pack_position_tolerance = float(
            self.get_parameter('pack_position_tolerance_m').value
        )
        self.pack_horizontal_tolerance = math.sin(math.radians(
            float(self.get_parameter('pack_horizontal_tolerance_deg').value)
        ))
        self.joint_min = np.deg2rad(np.asarray(
            self.get_parameter('joint_min_deg').value,
            dtype=float
        ))
        self.joint_max = np.deg2rad(np.asarray(
            self.get_parameter('joint_max_deg').value,
            dtype=float
        ))

        self.preferred_chain = self._make_chain(PREFERRED_ACTIVE)
        self.fallback_chain = self._make_chain(FALLBACK_ACTIVE)
        self.fk_chain = self._make_chain(tuple())

        command_qos = QoSProfile(depth=10)
        command_qos.reliability = ReliabilityPolicy.RELIABLE
        command_qos.durability = DurabilityPolicy.VOLATILE

        self.grip_plan_pub = self.create_publisher(
            Float64MultiArray,
            '/arm/joint_waypoints',
            command_qos
        )
        self.pack_plan_pub = self.create_publisher(
            Float64MultiArray,
            '/arm/joint_waypoints_pack',
            command_qos
        )
        self.joint_target_pub = self.create_publisher(
            Float64MultiArray,
            '/arm/joint_target',
            command_qos
        )
        self.rail_target_pub = self.create_publisher(
            String,
            '/rail/target_ingredient',
            10
        )
        self.control_motion_done = self.create_publisher(
            String,
            '/control/motion_done',
            10
        )
        self.create_subscription(
            Empty,
            '/llm/start_control',
            self._start_callback,
            10
        )
        self.create_subscription(
            String,
            '/llm/task_plan',
            self._llm_task_callback,
            10
        )
        self.create_subscription(
            String,
            '/vision/pick_pose',
            self._pick_callback,
            10
        )
        self.create_subscription(
            Empty,
            '/arm/motion_done',
            self._motion_done_callback,
            10
        )
        self.create_subscription(
            Empty,
            '/rail/motion_done',
            self._rail_motion_done_callback,
            10
        )

        self.state = STATE_IDLE
        self.current_ingredient = None
        self.repeat_total = 0
        self.repeat_completed = 0   
        self.pick_mode = None
        self.point1_q = None
        self.point2_q = None

        self._set_state(STATE_IDLE)

    def _set_state(self, state):
        self.state = state

    def _publish_control_motion_done(self):
        if self.current_ingredient is None:
            self.get_logger().error(
                'Current ingredient is not available.'
            )
            return

        msg = String()
        msg.data = self.current_ingredient
        self.control_motion_done.publish(msg)

        self.get_logger().info(
            f'class 전달: {self.current_ingredient}'
        )

    def _start_callback(self, _msg):
        if self.state != STATE_IDLE:
            return

        self._set_state(STATE_WAIT_INITIAL_DONE)

        try:
            self._plan_initial_pack()
        except RuntimeError as error:
            self._set_state(STATE_IDLE)
            self.get_logger().error(str(error))

    def _llm_task_callback(self, msg):
        if self.state != STATE_WAIT_LLM_TASK:
            return

        try:
            data = json.loads(msg.data)
            ingredient = data['ingredient']
            repeat_count = int(data['repeat_count'])
            self._class_to_mode(ingredient)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            self.get_logger().error(str(error))
            return

        if repeat_count < 1:
            self.get_logger().error('반복 횟수 1회 이상이어야 합니당')
            return

        self.current_ingredient = ingredient
        self.repeat_total = repeat_count
        self.repeat_completed = 0
        self.pick_mode = None
        self.point2_q = None

        rail_msg = String()
        rail_msg.data = ingredient

        self._set_state(STATE_WAIT_RAIL_DONE)
        self.rail_target_pub.publish(rail_msg)

    def _motion_done_callback(self, _msg):
        if self.state == STATE_WAIT_INITIAL_DONE:
            self._set_state(STATE_WAIT_LLM_TASK)
            return

        if self.state == STATE_WAIT_POINT1_DONE:
            self._set_state(STATE_WAIT_PICK)

            # 비전에 LLM에서 전달받은 클래스 전송
            self._publish_control_motion_done()
            return

        if self.state == STATE_WAIT_PICK_DONE:
            if self.pick_mode is None:
                self.get_logger().error(
                    'Pick mode is not available.'
                )
                return

            try:
                place_position, place_yaw = (
                    self._get_fixed_place_pose()
                )

                self._set_state(STATE_WAIT_PLACE_DONE)

                self._plan_place(
                    place_position,
                    place_yaw,
                    self.pick_mode
                )

            except RuntimeError as error:
                self.get_logger().error(str(error))
                self._set_state(STATE_WAIT_PICK)

            return

        if self.state == STATE_WAIT_PLACE_DONE:
            self.repeat_completed += 1
            self.pick_mode = None
            self.point2_q = None

            if self.repeat_completed < self.repeat_total:
                self._move_to_point(
                    POINT1,
                    STATE_WAIT_POINT1_DONE
                )
            else:
                self._move_home()

            return

        if self.state == STATE_WAIT_HOME_DONE:
            self.current_ingredient = None
            self.repeat_total = 0
            self.repeat_completed = 0
            self.pick_mode = None
            self.point1_q = None
            self.point2_q = None

            self._set_state(STATE_WAIT_LLM_TASK)

    def _rail_motion_done_callback(self, _msg):
        if self.state == STATE_WAIT_RAIL_DONE:
            self._move_to_point(POINT1, STATE_WAIT_POINT1_DONE)

    def _pick_callback(self, msg):
        if self.state != STATE_WAIT_PICK:
            return

        try:
            position, yaw, class_name = self._read_pick_message(msg)
            mode = self._class_to_mode(class_name)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        if class_name != self.current_ingredient:
            self.get_logger().error(
                f'Vision class {class_name} does not match '
                f'{self.current_ingredient}.'
            )
            return

        self.pick_mode = mode
        self._set_state(STATE_WAIT_PICK_DONE)

        try:
            self._plan_pick(position, yaw, mode)
        except RuntimeError as error:
            self.pick_mode = None
            self._set_state(STATE_WAIT_PICK)
            self.get_logger().error(str(error))

    def _move_to_point(self, point, next_state):
        try:
            q_target = self._solve_pose(
                point,
                np.zeros(DOF, dtype=float),
                DEFAULT_YAW_RAD,
                'grip'
            )
        except RuntimeError as error:
            self.get_logger().error(str(error))
            return

        if np.allclose(point, POINT1):
            self.point1_q = q_target.copy()

        msg = Float64MultiArray()
        msg.data = q_target.tolist()
        self._set_state(next_state)
        self.joint_target_pub.publish(msg)

    def _move_home(self):
        msg = Float64MultiArray()
        msg.data = np.zeros(DOF, dtype=float).tolist()
        self._set_state(STATE_WAIT_HOME_DONE)
        self.joint_target_pub.publish(msg)

    @staticmethod
    def _class_to_mode(class_name):
        if class_name in GRIP_CLASSES:
            return 'grip'
        if class_name in PACK_CLASSES:
            return 'pack'
        raise ValueError(f'Unsupported class: {class_name}')

    def _get_fixed_place_pose(self):
        if self.current_ingredient is None:
            raise RuntimeError(
                'Current ingredient is not available.'
            )

        class_name = self.current_ingredient

        place_key = (
            '소스'
            if class_name in {'소스1', '소스2', '소스3'}
            else class_name
        )

        place_pose = FIXED_PLACE_POINTS[place_key]

        position = place_pose['position'].copy()
        yaw = math.radians(float(place_pose['yaw_deg']))

        if (
            not np.all(np.isfinite(position))
            or not math.isfinite(yaw)
        ):
            raise RuntimeError(
                f'Fixed place pose for {place_key} '
                f'contains a non-finite value.'
            )

        return position, yaw

    @staticmethod
    def _read_pick_message(msg):
        try:
            data = json.loads(msg.data)

            position = np.array([
                float(data['x']),
                float(data['y']),
                float(data['z']),
            ], dtype=float)

            yaw_value = data.get('yaw', 180.0)
            if yaw_value is None:
                yaw_value = 180.0

            yaw = math.radians(float(yaw_value))
            class_name = str(data['class_name']).strip()

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError
        ) as error:
            raise ValueError(
                'Pick message must contain x, y, z, class_name, yaw'
            ) from error

        if (
            not np.all(np.isfinite(position))
            or not math.isfinite(yaw)
        ):
            raise ValueError(
                'Pick pose가 사용할 수 없는 값을 포함하고 있습니당'
            )

        return position, yaw, class_name

    ## 공압 IK에서는 q3을 홈 각도(180)로 강제
    def _solve_pack_pick_lift(self, position, previous_q):
        """
        용기 p2p와 공압 pick에서 사용하는 공압 pick/lift IK 계산 -> 5번이 평행, 6번 고정
        """
        position = np.asarray(position, dtype=float)
        previous_q = np.asarray(previous_q, dtype=float)

        lift_position = position.copy()
        lift_position[2] += self.lift_height

        q_pick = self._solve_pose(
            position,
            previous_q,
            DEFAULT_YAW_RAD,
            'pack'
        )

        q_lift = self._solve_pose(
            lift_position,
            q_pick,
            DEFAULT_YAW_RAD,
            'pack'
        )

        return q_pick, q_lift

    def _plan_pick(self, position, yaw, mode):
        position = np.asarray(position, dtype=float)

        approach = position.copy()
        approach[2] += self.lift_height

        q_motion_start = (
            self.point1_q.copy()
            if self.point1_q is not None
            else np.zeros(DOF, dtype=float)
        )

        if mode == 'pack':
            q_pick, q_lift = self._solve_pack_pick_lift(
                position,
                q_motion_start
            )
            q_approach = q_lift.copy()

            q_point2 = self._solve_pose(
                POINT2,
                q_lift,
                DEFAULT_YAW_RAD,
                'pack'
            )
        else:
            q_approach = self._solve_pose(
                approach,
                q_motion_start,
                yaw,
                'grip'
            )
            q_pick = self._solve_pose(
                position,
                q_approach,
                yaw,
                'grip'
            )
            q_lift = self._solve_pose(
                approach,
                q_pick,
                yaw,
                'grip'
            )
            q_point2 = self._solve_pose(
                POINT2,
                q_lift,
                DEFAULT_YAW_RAD,
                'grip'
            )

        self.point2_q = q_point2.copy()

        self._publish_waypoints(
            self.pack_plan_pub
            if mode == 'pack'
            else self.grip_plan_pub,
            f'{mode}_pick',
            q_approach,
            q_pick,
            q_lift,
            q_point2
        )

    def _plan_place(self, position, yaw, mode):
        if self.point2_q is None:
            raise RuntimeError(
                'Point 2 pose가 닿을 수 없는 위치에 있습니당'
            )

        position = np.asarray(position, dtype=float)

        approach = position.copy()
        approach[2] += self.lift_height

        if mode == 'pack':
            # 용기 p2p의 q_p2_lift와 같은 순서
            q_approach = self._solve_pose(
                approach,
                self.point2_q,
                DEFAULT_YAW_RAD,
                'pack'
            )

            # 용기 p2p의 q_p2와 같은 순서
            q_place = self._solve_pose(
                position,
                q_approach,
                DEFAULT_YAW_RAD,
                'pack'
            )

            q_lift = q_approach.copy()

        else:
            q_approach = self._solve_pose(
                approach,
                self.point2_q,
                yaw,
                'grip'
            )

            q_place = self._solve_pose(
                position,
                q_approach,
                yaw,
                'grip'
            )

            q_lift = self._solve_pose(
                approach,
                q_place,
                yaw,
                'grip'
            )

        self._publish_waypoints(
            self.pack_plan_pub
            if mode == 'pack'
            else self.grip_plan_pub,
            f'{mode}_place',
            q_approach,
            q_place,
            q_lift,
            q_lift
        )

    def _plan_initial_pack(self):
        p1 = INITIAL_PACK_PICK_POINT
        p2 = INITIAL_PACK_PLACE_POINT

        p2_lift = p2.copy()
        p2_lift[2] = p1[2] + self.lift_height

        q_home = np.zeros(DOF, dtype=float)
        q_p1, q_p1_lift = self._solve_pack_pick_lift(
            p1,
            q_home
        )
        q_p2_lift = self._solve_pose(
            p2_lift,
            q_p1_lift,
            DEFAULT_YAW_RAD,
            'pack'
        )
        q_p2 = self._solve_pose(
            p2,
            q_p2_lift,
            DEFAULT_YAW_RAD,
            'pack'
        )

        self._publish_waypoints(
            self.pack_plan_pub,
            'pack_full',
            q_p1,
            q_p1_lift,
            q_p2_lift,
            q_p2
        )

    def _solve_pose(self, target, previous_q, yaw, mode):
        previous_q = np.asarray(previous_q, dtype=float)

        if mode == 'pack':
            q = self._solve_pack_point(target, previous_q)

            q[2] = 0 # 고정
            q[5] = previous_q[5]
            return q

        q = self._solve_point(target, previous_q)
        q[5] = self._target_q6(yaw, q, previous_q[5])
        return q

    @staticmethod
    def _publish_waypoints(publisher, phase, q1, q2, q3, q4):
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

    def _make_chain(self, active_joints):
        base = np.eye(4, dtype=float)

        links = [OriginLink(), FixedLink('base', base)]
        active_mask = [False, False]

        for i in range(DOF):
            links.append(
                DHLink(
                    name=f'joint_{i + 1}',
                    d=DH_D[i],
                    a=DH_A[i],
                    alpha=DH_ALPHA[i],
                    theta_offset=DH_THETA_OFFSET[i],
                    bounds=(
                        float(self.joint_min[i]),
                        float(self.joint_max[i])
                    )
                )
            )
            active_mask.append(i in active_joints)

        links.append(FixedLink('ee', np.eye(4, dtype=float)))
        active_mask.append(False)

        return Chain(
            name='six_dof_dh',
            links=links,
            active_links_mask=active_mask
        )

    def _full_q(self, q):
        full = np.zeros(FULL_Q_SIZE, dtype=float)
        full[Q_FULL_START:Q_FULL_START + DOF] = q
        return full

    @staticmethod
    def _full_to_q(full):
        return np.asarray(
            full[Q_FULL_START:Q_FULL_START + DOF],
            dtype=float
        )

    def _fk_matrix(self, q):
        return self.fk_chain.forward_kinematics(self._full_q(q))

    def _fk(self, q):
        return self._fk_matrix(q)[:3, 3].copy()

    def _fk_to_pack_joint5(self, q):
        transform = np.eye(4, dtype=float)
        for i in range(5):
            transform = transform @ dh_matrix(
                q[i] + PACK_DH_THETA_OFFSET[i],
                DH_D[i],
                DH_A[i],
                PACK_DH_ALPHA[i]
            )
        return transform

    def _pack_fk(self, q):
        transform = self._fk_to_pack_joint5(q)

        return (
            transform[:3, 3]
            + transform[:3, :3] @ self.pack_tcp_offset
        )

    def _pack_horizontal_component(self, q):
        return float(
            self._fk_to_pack_joint5(q)[2, 0]
        )

    @staticmethod
    def _pack_q5(q2, q3, q4, branch, reference_q5):
        link_angle = math.atan2(
            math.cos(q2),
            math.sin(q2) * math.cos(q3)
        )
        raw_q5 = link_angle + branch - q4
        return float(
            reference_q5 + wrap_to_pi(raw_q5 - reference_q5)
        )

    def _pack_seeds(self, previous_q, q1):
        primary_postures = [
            (20.0, 20.0),
            (35.0, 30.0),
            (50.0, 40.0),
            (65.0, 55.0),
            (80.0, 65.0),
        ]

        result = []

        # q1을 목표 방향으로 변경한 seed를 우선으로 사용
        seed = previous_q.copy()
        seed[0] = q1
        seed[2] = 0

        result.append(
            np.clip(seed, self.joint_min, self.joint_max)
        )

        for q2_deg, q4_deg in primary_postures:
            seed = previous_q.copy()

            seed[0] = q1
            seed[1] = math.radians(q2_deg)
            seed[2] = 0
            seed[3] = math.radians(q4_deg)

            result.append(
                np.clip(seed, self.joint_min, self.joint_max)
            )
        return result

    def _solve_pack_point(self, target, previous_q):
        target = np.asarray(target, dtype=float)
        q1_target = self._target_q1(target, previous_q[0])
        candidates = []

        for seed in self._pack_seeds(previous_q, q1_target):
            for branch in (0.0, math.pi):

                def build_q(active_q):
                    q = previous_q.copy()

                    # 공압을 사용할 때 해 결정 변수는 q1, q2, q4만 사용
                    q[np.array([0, 1, 3], dtype=int)] = active_q
                    q[2] = 0 # 고정
                    q[4] = self._pack_q5(
                        q[1],
                        q[2],
                        q[3],
                        branch,
                        previous_q[4]
                    )
                    return q

                def residual(active_q):
                    q = build_q(active_q)

                    q5_low = max(
                        0.0,
                        self.joint_min[4] - q[4]
                    )
                    q5_high = max(
                        0.0,
                        q[4] - self.joint_max[4]
                    )

                    joint_delta = wrapped_q_delta(
                        q[np.array([0, 1, 3], dtype=int)],
                        previous_q[np.array([0, 1, 3], dtype=int)]
                    )

                    q1_target_error = wrap_to_pi(
                        q[0] - q1_target
                    )

                    q2_backward = max(0.0, -q[1])
                    q4_backward = max(0.0, -q[3])

                    continuity_weights = np.array([
                        0.25,   # q1: 목표 방향 회전을 방해하지 않도록 작게
                        1.0,    # q2
                        1.0,    # q4
                    ])

                    return np.concatenate([
                        PACK_POSITION_WEIGHT * (
                            self._pack_fk(q) - target
                        ),

                        np.array([
                            PACK_Q5_LIMIT_WEIGHT * q5_low,
                            PACK_Q5_LIMIT_WEIGHT * q5_high,
                        ]),

                        PACK_CONTINUITY_WEIGHT
                        * continuity_weights
                        * joint_delta,

                        np.array([
                            2.0
                            * PACK_CONTINUITY_WEIGHT
                            * q1_target_error,

                            2.0
                            * PACK_CONTINUITY_WEIGHT
                            * q2_backward,

                            2.0
                            * PACK_CONTINUITY_WEIGHT
                            * q4_backward,
                        ]),
                    ])

                result = least_squares(
                    residual,
                    seed[np.array([0, 1, 3], dtype=int)],
                    bounds=(
                        self.joint_min[np.array([0, 1, 3], dtype=int)],
                        self.joint_max[np.array([0, 1, 3], dtype=int)]
                    ),
                    max_nfev=PACK_MAX_NFEV
                )

                q = build_q(result.x)

                if not (
                    self.joint_min[4]
                    <= q[4]
                    <= self.joint_max[4]
                ):
                    continue

                position_error = float(
                    np.linalg.norm(
                        self._pack_fk(q) - target
                    )
                )

                horizontal_error = abs(
                    self._pack_horizontal_component(q)
                )

                joint_delta = wrapped_q_delta(
                    q[np.array([0, 1, 3], dtype=int)],
                    previous_q[np.array([0, 1, 3], dtype=int)]
                )

                q1_direction_error = abs(
                    wrap_to_pi(q[0] - q1_target)
                )

                q2_backward = max(0.0, -q[1])
                q4_backward = max(0.0, -q[3])

                total_motion = float(
                    np.linalg.norm(joint_delta)
                )

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
            raise RuntimeError(
                f'Pneumatic IK failed for target '
                f'{np.round(target, 4)}.'
            )

        min_error = min(
            item[0] for item in candidates
        )

        selected = min(
            (
                item
                for item in candidates
                if item[0] <= min_error + IK_ERROR_MARGIN_M
            ),
            key=lambda item: (
                item[1],  # 공압 링크 수평 오차
                item[2],  # q1 방향, q3 유지, 전방 자세 점수
                item[3],  # 전체 관절 이동량
            )
        )

        if selected[0] > self.pack_position_tolerance:
            raise RuntimeError(
                f'Pneumatic IK position error is too large: '
                f'{selected[0]:.6f} m'
            )

        if selected[1] > self.pack_horizontal_tolerance:
            raise RuntimeError(
                f'Pneumatic link is not horizontal enough: '
                f'{selected[1]:.6f}'
            )

        return selected[4] 
    
    def _target_q6(self, target_yaw, q, previous_q6):
        q_without_q6 = q.copy()
        q_without_q6[5] = 0.0

        rotation = self._fk_matrix(q_without_q6)[:3, :3]
        current_yaw = math.atan2(
            float(rotation[1, 0]),
            float(rotation[0, 0])
        )
        desired = wrap_to_pi(
            target_yaw - current_yaw
        )

        candidates = np.array([
            desired - 2.0 * math.pi,
            desired,
            desired + 2.0 * math.pi,
        ])
        valid = candidates[
            (candidates >= self.joint_min[5])
            & (candidates <= self.joint_max[5])
        ]

        if valid.size == 0:
            return float(np.clip(
                desired,
                self.joint_min[5],
                self.joint_max[5]
            ))

        return float(valid[np.argmin(np.abs(valid - previous_q6))])

    def _target_q1(self, target, previous_q1):
        xy = np.asarray(target[:2], dtype=float)

        if np.linalg.norm(xy) < 1.0e-8:
            return float(previous_q1)

        desired = wrap_to_pi(
            math.atan2(float(xy[1]), float(xy[0]))
        )
        candidates = np.array([
            desired - 2.0 * math.pi,
            desired,
            desired + 2.0 * math.pi,
        ])
        valid = candidates[
            (candidates >= self.joint_min[0])
            & (candidates <= self.joint_max[0])
        ]

        if valid.size == 0:
            return float(np.clip(
                desired,
                self.joint_min[0],
                self.joint_max[0]
            ))

        return float(valid[np.argmin(np.abs(valid - previous_q1))])

    def _seeds(self, previous_q, q1):
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
            result.append(
                np.clip(seed, self.joint_min, self.joint_max)
            )

        return result

    def _candidates(self, chain, target, previous_q, q1):
        result = []

        for seed in self._seeds(previous_q, q1):
            full = chain.inverse_kinematics(
                target_position=target,
                initial_position=self._full_q(seed),
                max_iter=IK_MAX_ITER
            )
            q = np.clip(
                self._full_to_q(full),
                self.joint_min,
                self.joint_max
            )
            q[0] = q1
            q[5] = previous_q[5]

            error = float(np.linalg.norm(target - self._fk(q)))
            q3_use = abs(wrap_to_pi(q[2] - previous_q[2]))
            continuity = float(np.linalg.norm(
                wrapped_q_delta(
                    q[[1, 3, 4]],
                    previous_q[[1, 3, 4]]
                )
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
    def _best(candidates):
        min_error = min(item[0] for item in candidates)
        return min(
            (
                item
                for item in candidates
                if item[0] <= min_error + IK_ERROR_MARGIN_M
            ),
            key=lambda item: item[1]
        )

    def _solve_point(self, target, previous_q):
        q1 = self._target_q1(target, previous_q[0])
        preferred = self._best(
            self._candidates(
                self.preferred_chain,
                target,
                previous_q,
                q1
            )
        )

        if preferred[0] <= PREFERRED_IK_TOLERANCE_M:
            selected = preferred
        else:
            fallback = self._best(
                self._candidates(
                    self.fallback_chain,
                    target,
                    previous_q,
                    q1
                )
            )
            selected = (
                fallback
                if fallback[0] + IK_ERROR_MARGIN_M < preferred[0]
                else preferred
            )

        if selected[0] > IK_TOLERANCE_M:
            raise RuntimeError(
                f'IK error is too large: {selected[0]:.6f} m'
            )

        return selected[2]


def main(args=None):
    rclpy.init(args=args)
    node = PointPosePlannerNode()

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
