#!/usr/bin/env python3

import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String

DOF = 6

DH_D = np.array([0.1271, 0.0, 0.22000, 0.0, 0.0, 0.188046], dtype=float)
DH_A = np.array([0.0, 0.0, 0.0, 0.220, 0.0, 0.0], dtype=float)
DH_ALPHA = np.deg2rad([-90.0, 90.0, -90.0, 0.0, 90.0, 0.0])
DH_THETA = np.deg2rad([0.0, 0.0, 0.0, -90.0, 90.0, 0.0])

# EE 좌표축을 Camera 좌표축 방향으로 맞추기 위한 Z축 회전
CAMERA_Z_ROTATION_DEG = -90.0

# -90도 회전된 Camera 좌표축 기준 EE -> 카메라 오프셋
CAMERA_X_OFFSET_M = -0.0335
CAMERA_Y_OFFSET_M = -0.065561
CAMERA_Z_OFFSET_M = -0.067205


def dh_transform(theta, d, a, alpha):
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


def rotation_z_transform(theta):
    ct = math.cos(theta)
    st = math.sin(theta)

    return np.array([
        [ct, -st, 0.0, 0.0],
        [st, ct, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def translation_transform(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


class TransformNode(Node):

    def __init__(self) -> None:
        super().__init__('transform_node')

        self.declare_parameter(
            'joint_names',
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',],
        )

        self.joint_names = list(
            self.get_parameter('joint_names').get_parameter_value().string_array_value
        )

        self.pending_raw_pose = None
        self.waiting_for_joint_state = False

        # 1) Base -> EE FK 뒤에 적용할 -90도 회전
        self.t_ee_rotated = rotation_z_transform(
            math.radians(CAMERA_Z_ROTATION_DEG)
        )

        # 2) -90도 회전된 Camera 좌표축 기준 오프셋
        self.t_camera_offset = translation_transform(
            CAMERA_X_OFFSET_M,
            CAMERA_Y_OFFSET_M,
            CAMERA_Z_OFFSET_M,
        )

        self.joint_state_request_pub = self.create_publisher(Empty, '/arm/request_joint_state', 10,)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10,)

        self.raw_pick_pose_sub = self.create_subscription(String, '/vision/raw_pick_pose', self._raw_pick_pose_callback, 10,)
        self.pick_pose_pub = self.create_publisher(String, '/vision/pick_pose', 10,)

    def _joint_state_callback(self, msg):
        if not self.waiting_for_joint_state:
            return

        if self.pending_raw_pose is None:
            self.waiting_for_joint_state = False
            return

        q_actual = self._extract_joint_positions(msg)

        if q_actual is None:
            return

        raw_pose = self.pending_raw_pose

        self.pending_raw_pose = None
        self.waiting_for_joint_state = False

        self._transform_and_publish(raw_pose, q_actual)

    def _extract_joint_positions(self, msg):
        if msg.name:
            if len(msg.position) < len(msg.name):
                self.get_logger().warning('JointState position array is shorter than name array')
                return None

            index_by_name = {
                name: index
                for index, name in enumerate(msg.name)
            }

            if all(
                name in index_by_name
                for name in self.joint_names
            ):
                return np.array([
                    msg.position[index_by_name[name]]
                    for name in self.joint_names
                ], dtype=float)

        if len(msg.position) >= DOF:
            return np.array(
                msg.position[:DOF],
                dtype=float,
            )

        return None

    def _fk_base_ee(self, q):
        q = np.asarray(q, dtype=float)

        t_base_ee = np.eye(4, dtype=float)

        for index in range(DOF):
            theta = float(q[index] + DH_THETA[index])

            t_base_ee = (
                t_base_ee @ dh_transform(
                    theta,
                    float(DH_D[index]),
                    float(DH_A[index]),
                    float(DH_ALPHA[index]),
                )
            )

        return t_base_ee

    def _raw_pick_pose_callback(self, msg):
        fields = [
            field.strip()
            for field in msg.data.split(',')
        ]

        class_name = fields[0]

        x_camera = float(fields[1])
        y_camera = float(fields[2])
        z_camera = float(fields[3])
        yaw_camera_deg = float(fields[4]) - 180

        values = np.array([
            x_camera,
            y_camera,
            z_camera,
            yaw_camera_deg,
        ], dtype=float)

        if not np.all(np.isfinite(values)):
            self.get_logger().error('Raw camera pose contains non-finite values.')
            return

        if self.waiting_for_joint_state:
            self.get_logger().warning('Already waiting for JointState')

        self.pending_raw_pose = {
            'class_name': class_name,
            'x': x_camera,
            'y': y_camera,
            'z': z_camera,
            'yaw': yaw_camera_deg,
        }

        self.waiting_for_joint_state = True

        self.joint_state_request_pub.publish(Empty())

    def _transform_and_publish(self, raw_pose, q_actual):
        class_name = raw_pose['class_name']
        yaw_camera_deg = float(raw_pose['yaw'])

        # Camera 좌표계에서 표현된 물체 위치
        p_camera = np.array([
            float(raw_pose['x']),
            float(raw_pose['y']),
            float(raw_pose['z']),
            1.0,
        ], dtype=float)

        # Base -> EE
        t_base_ee = self._fk_base_ee(q_actual)

        p_base = (
            t_base_ee
            @ self.t_ee_rotated
            @ self.t_camera_offset
            @ p_camera
        )

        if (
            not np.all(np.isfinite(t_base_ee))
            or not np.all(np.isfinite(p_base))
        ):
            self.get_logger().error(
                'Calculated transform contains non-finite values.'
            )
            return

        output = {
            'class_name': class_name,
            'x': float(p_base[0]),
            'y': float(p_base[1]),
            'z': float(p_base[2]) - 0.005,
            'yaw': yaw_camera_deg,
            'frame_id': 'base',
        }

        output_msg = String()
        output_msg.data = json.dumps(
            output,
            ensure_ascii=False,
        )

        self.pick_pose_pub.publish(output_msg)

        self.get_logger().info(
            '\n'
            '========== Vision Pose Transform ==========\n'
            f'class_name      : {class_name}\n'
            f'joint_q_deg     : '
            f'{np.round(np.rad2deg(q_actual), 3).tolist()}\n'
            f'base_position   : '
            f'x={p_base[0]:.4f}, '
            f'y={p_base[1]:.4f}, '
            f'z={p_base[2]:.4f} [m]\n'
            f'camera_yaw      : '
            f'{yaw_camera_deg:.2f} [deg]\n'
            f'published_json  : '
            f'{output_msg.data}\n'
            '==========================================='
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = TransformNode()

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