#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from ikpy.chain import Chain
from ikpy.link import Link, OriginLink
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray


DOF = 6

BASE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=float)

DH_D = np.array([
    0.1271,
    0.0,
    0.22000,
    0.0,
    0.0,
    0.188046,
], dtype=float)

DH_A = np.array([
    0.0,
    0.0,
    0.0,
    0.220,
    0.0,
    0.0,
], dtype=float)

DH_ALPHA = np.deg2rad([
    -90.0,
    90.0,
    -90.0,
    0.0,
    90.0,
    0.0,
])

DH_THETA_OFFSET = np.deg2rad([
    0.0,
    0.0,
    0.0,
    -90.0,
    90.0,
    0.0,
])

FRONT_AZIMUTH = 0.0

PREFERRED_ACTIVE = (1, 3, 4)
FALLBACK_ACTIVE = (1, 2, 3, 4)

Q_FULL_START = 2
FULL_Q_SIZE = 9

IK_MAX_ITER = 100
PREFERRED_IK_TOLERANCE_M = 0.004
IK_TOLERANCE_M = 0.005
IK_ERROR_MARGIN_M = 0.002


def wrap_to_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def wrapped_q_delta(q_goal, q_start):
    return (q_goal - q_start + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_to_yaw(quaternion):
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


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

    def get_rotation_axis(self):
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=float)


class PointPosePlannerNode(Node):

    def __init__(self):
        super().__init__('point_pose_planner_node')

        ## yaw 값 초기 각도가 180도 이기 때문에 0도가 아니라 180도 입력해야 한다,,

        self.declare_parameter('p1', [0.35, 0.00, 0.05])
        self.declare_parameter('p1_yaw_deg', 180.0)
        self.declare_parameter('p2', [0.23, 0.0, 0.07])
        self.declare_parameter('p2_yaw_deg', 90.0)
        self.declare_parameter('lift_height', 0.15)
        self.declare_parameter('yaw_sign', 1.0)
        self.declare_parameter('yaw_offset_deg', 0.0)
        self.declare_parameter('auto_publish_parameters', True)

        self.declare_parameter(
            'joint_min_deg',
            [-170.0, -120.0, -170.0, -120.0, -120.0, -170.0]
        )
        self.declare_parameter(
            'joint_max_deg',
            [170.0, 120.0, 170.0, 120.0, 120.0, 170.0]
        )

        self.parameter_p1 = np.asarray(
            self.get_parameter('p1').value,
            dtype=float
        )
        self.parameter_p1_yaw = math.radians(
            float(self.get_parameter('p1_yaw_deg').value)
        )
        self.parameter_p2 = np.asarray(
            self.get_parameter('p2').value,
            dtype=float
        )
        self.parameter_p2_yaw = math.radians(
            float(self.get_parameter('p2_yaw_deg').value)
        )
        self.lift_height = float(
            self.get_parameter('lift_height').value
        )
        self.yaw_sign = float(
            self.get_parameter('yaw_sign').value
        )
        self.yaw_offset = math.radians(
            float(self.get_parameter('yaw_offset_deg').value)
        )
        self.auto_publish_parameters = bool(
            self.get_parameter('auto_publish_parameters').value
        )

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

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.plan_pub = self.create_publisher(
            Float64MultiArray,
            '/arm/joint_waypoints',
            qos
        )

        ## 토픽 이름 gui 아니면 llm으로 해야할 듯
        self.place_sub = self.create_subscription(
            PoseStamped,
            '/gui/start_control',
            self._place_callback,
            10
        )

        self.pick_sub = self.create_subscription(
            PoseStamped,
            '/vision/pick_pose',
            self._pick_callback,
            10
        )
        self.place_sub = self.create_subscription(
            PoseStamped,
            '/vision/place_pose',
            self._place_callback,
            10
        )

        self.place_sub = self.create_subscription(
            PoseStamped,
            '/vision/place_pack',
            self._place_callback,
            10
        )

        self.vision_pick = None
        self.vision_pick_yaw = None
        self.vision_place = None
        self.vision_place_yaw = None
        self.parameter_timer = None

        if self.auto_publish_parameters:
            self.parameter_timer = self.create_timer(
                0.5,
                self._publish_parameter_plan_once
            )

        self.get_logger().info(
            'Waiting for parameter poses or vision poses.'
        )
        self.get_logger().info(
            'Vision poses must be expressed in the robot world/base frame.'
        )

    def _publish_parameter_plan_once(self):
        if self.parameter_timer is not None:
            self.parameter_timer.cancel()

        self._plan_and_publish(
            self.parameter_p1,
            self.parameter_p1_yaw,
            self.parameter_p2,
            self.parameter_p2_yaw,
            source='parameters'
        )

    def _pick_callback(self, msg):
        self.vision_pick = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)
        self.vision_pick_yaw = quaternion_to_yaw(
            msg.pose.orientation
        )

        self.get_logger().info(
            f'Received vision P1: '
            f'position={np.round(self.vision_pick, 4)}, '
            f'yaw={math.degrees(self.vision_pick_yaw):.2f} deg'
        )
        self._publish_vision_plan_if_ready()

    def _place_callback(self, msg):
        self.vision_place = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)
        self.vision_place_yaw = quaternion_to_yaw(
            msg.pose.orientation
        )

        self.get_logger().info(
            f'Received vision P2: '
            f'position={np.round(self.vision_place, 4)}, '
            f'yaw={math.degrees(self.vision_place_yaw):.2f} deg'
        )
        self._publish_vision_plan_if_ready()

    def _publish_vision_plan_if_ready(self):
        if (
            self.vision_pick is None
            or self.vision_pick_yaw is None
            or self.vision_place is None
            or self.vision_place_yaw is None
        ):
            return

        self._plan_and_publish(
            self.vision_pick,
            self.vision_pick_yaw,
            self.vision_place,
            self.vision_place_yaw,
            source='vision'
        )

        self.vision_pick = None
        self.vision_pick_yaw = None
        self.vision_place = None
        self.vision_place_yaw = None

    def _plan_and_publish(
        self,
        p1,
        p1_yaw,
        p2,
        p2_yaw,
        source
    ):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)

        p1_lift = p1.copy()
        p1_lift[2] += self.lift_height

        p2_lift = p2.copy()
        p2_lift[2] = p1_lift[2]

        q_home = np.zeros(DOF, dtype=float)

        q_p1 = self._solve_point(p1, q_home)
        q_p1[5] = self._target_q6(
            p1_yaw,
            q_p1,
            q_home[5]
        )

        q_lift = self._solve_point(p1_lift, q_p1)
        q_lift[5] = self._target_q6(
            p1_yaw,
            q_lift,
            q_p1[5]
        )

        q_p2_lift = self._solve_point(p2_lift, q_lift)
        q_p2_lift[4] = q_lift[4]
        q_p2_lift[5] = self._target_q6(
            p1_yaw,
            q_p2_lift,
            q_lift[5]
        )

        q_p2 = self._solve_point(p2, q_p2_lift)
        q_p2[5] = self._target_q6(
            p2_yaw,
            q_p2,
            q_p2_lift[5]
        )

        msg = Float64MultiArray()
        msg.data = np.concatenate([
            q_p1,
            q_lift,
            q_p2_lift,
            q_p2,
        ]).astype(float).tolist()

        self.plan_pub.publish(msg)

        self.get_logger().info(
            f'Published plan from {source}: '
            f'P1={np.round(p1, 4)}, '
            f'P1_YAW={math.degrees(p1_yaw):.2f} deg, '
            f'P2={np.round(p2, 4)}, '
            f'P2_YAW={math.degrees(p2_yaw):.2f} deg'
        )
        self.get_logger().info(
            f'Q6 deg: '
            f'P1={math.degrees(q_p1[5]):.2f}, '
            f'LIFT={math.degrees(q_lift[5]):.2f}, '
            f'P2_LIFT={math.degrees(q_p2_lift[5]):.2f}, '
            f'P2={math.degrees(q_p2[5]):.2f}'
        )

    def _make_chain(self, active_joints):
        base = np.eye(4, dtype=float)
        base[:3, 3] = BASE_OFFSET

        links = [
            OriginLink(),
            FixedLink('base', base),
        ]
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
            name='six_dof_standard_dh',
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
        return self.fk_chain.forward_kinematics(
            self._full_q(q)
        )

    def _fk(self, q):
        return self._fk_matrix(q)[:3, 3].copy()

    def _target_q6(self, target_yaw, q, previous_q6):
        q_without_q6 = q.copy()
        q_without_q6[5] = 0.0

        rotation = self._fk_matrix(q_without_q6)[:3, :3]
        current_yaw_without_q6 = math.atan2(
            float(rotation[1, 0]),
            float(rotation[0, 0])
        )

        desired = wrap_to_pi(
            self.yaw_sign * (
                target_yaw - current_yaw_without_q6
            )
            + self.yaw_offset
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

        return float(
            valid[np.argmin(np.abs(valid - previous_q6))]
        )

    def _target_q1(self, target, previous_q1):
        xy = target[:2] - BASE_OFFSET[:2]

        if np.linalg.norm(xy) < 1.0e-8:
            return float(previous_q1)

        desired = wrap_to_pi(
            math.atan2(float(xy[1]), float(xy[0]))
            - FRONT_AZIMUTH
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

        return float(
            valid[np.argmin(np.abs(valid - previous_q1))]
        )

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

            error = float(np.linalg.norm(
                target - self._fk(q)
            ))

            q3_use = abs(
                wrap_to_pi(q[2] - previous_q[2])
            )

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

        near = [
            item
            for item in candidates
            if item[0] <= min_error + IK_ERROR_MARGIN_M
        ]

        return min(near, key=lambda item: item[1])

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

            if fallback[0] + IK_ERROR_MARGIN_M < preferred[0]:
                selected = fallback
            else:
                selected = preferred

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