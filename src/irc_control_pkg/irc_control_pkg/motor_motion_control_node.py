#!/usr/bin/env python3

import math
import time
import serial
import numpy as np
import rclpy
from dynamixel_sdk import PacketHandler, PortHandler
from rclpy.node import Node
from std_msgs.msg import Empty, Float64MultiArray
from sensor_msgs.msg import JointState
from irc_control_pkg.motion_trajectory import (MotionTrajectory, motion_q_delta, CONTROL_READY)

PORT_XH = '/dev/dynamixel_0'
PORT_XM = '/dev/dynamixel_1'
ARDUINO_PORT = '/dev/ttyUSB3'

XH_IDS = [1, 2, 3, 4]
ARM_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_ID = 7
ALL_IDS = ARM_IDS + [GRIPPER_ID]
DOF = 6

BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

POSITION_MODE = 3
EXTENDED_POSITION_MODE = 4

HOME_RAW = np.full(DOF, 2048, dtype=int)
GRIPPER_HOME_RAW = 2048

GRIPPER_OPEN_DEG = {
    'noodle_thick': -53,
    'noodle_thin': -53,
    'mushroom': -15,
    'onion':    -15,
    'crab':     -30,
    'sausage':  -35,
    'spoon':    0
}

GRIPPER_CLOSE_DEG = {
    'noodle_thick': -70,
    'noodle_thin': -70,
    'mushroom': -70,
    'onion':    -70,
    'crab':     -68,
    'sausage':  -65,
    'spoon':    -54
}

FINISH_TOLERANCE_DEG = 0.2

ENABLE_MOTION = True

PROFILE_VELOCITY = 30
PROFILE_ACCELERATION = 20

JOINT_MIN = np.deg2rad([-200.0, -120.0, -170.0, -140.0, -120.0, -360.0])
JOINT_MAX = np.deg2rad([550.0, 120.0, 170.0, 140.0, 120.0, 360.0])
MAX_Q_STEP = math.radians(2.0)

GRIP_PHASES = {'grip_pick', 'grip_place', 'spoon_pick', 'spoon_place'}
PACK_PHASES = {'pack_pick', 'pack_place', 'pack_full', 'sauce_full'}


def signed_delta_tick(raw_now, raw_home):
    return (int(raw_now) - int(raw_home) + 2048) % 4096 - 2048

def to_u32(value):
    return int(value) & 0xFFFFFFFF

def to_s32(value):
    value = int(value) & 0xFFFFFFFF

    if value & 0x80000000:
        value -= 0x100000000

    return value


class HardwareMotionControlNode(Node):

    def __init__(self):
        super().__init__('hardware_motion_control_node')

        self.motion = MotionTrajectory()

        self.q_home = np.zeros(DOF, dtype=float)
        self.q_cmd_prev = self.q_home.copy()

        self.trajectory = []
        self.total_duration = 0.0
        self.final_target = self.q_home.copy()
        self.motion_active = False
        self.start_time = None

        self.port_xh = PortHandler(PORT_XH)
        self.port_xm = PortHandler(PORT_XM)
        self.packet = PacketHandler(PROTOCOL_VERSION)

        self.xh_open = False
        self.xm_open = False
        self.arduino = None
        self.closed = False

        self.setup_motors()
        self.hold_current_positions()
        self.set_all_torque(True)

        self.setup_arduino()
        self.command_pneumatic(enabled=False)

        self.motion_done_pub = self.create_publisher(Empty, '/arm/motion_done', 10)
        self.create_subscription(Float64MultiArray, '/arm/joint_waypoints', self.grip_plan_callback, 10)
        self.create_subscription(Float64MultiArray, '/arm/joint_waypoints_pack', self.pack_plan_callback, 10)
        self.create_subscription(Float64MultiArray, '/arm/joint_target', self.joint_target_callback, 10)

        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_state_request_sub = self.create_subscription(Empty, '/arm/request_joint_state', self.joint_state_request_callback, 10) # POINT1에서 실제 관절각을 요청받았을 때만 사용 -> 변환행렬 계산에 필요
        # 모션 제어 주기
        self.timer = self.create_timer(0.05, self.control_loop)
        # 시작 자세
        q_start = self.read_current_q()
        trajectory = []
        self.motion.move(trajectory, q_start, CONTROL_READY, 2.0)
        self.start_trajectory('joint_target', q_start, trajectory)

    def setup_arduino(self):
        try:
            self.arduino = serial.Serial(
                port=ARDUINO_PORT,
                baudrate=115200,
                timeout=0.2,
                write_timeout=1.0
            )

            time.sleep(2.0)

            self.arduino.reset_input_buffer()
            self.arduino.reset_output_buffer()

        except (serial.SerialException, OSError) as exc:
            raise RuntimeError(f'Arduino 연결 실패') from exc

    def grip_plan_callback(self, msg):
        self.waypoint_plan_callback(msg, GRIP_PHASES)

    def pack_plan_callback(self, msg):
        self.waypoint_plan_callback(msg, PACK_PHASES)

    def waypoint_plan_callback(self, msg, valid_phases):
        if self.motion_active:
            self.get_logger().warning('로봇팔 동작 중')
            return

        if len(msg.data) < DOF or len(msg.data) % DOF != 0:
            return

        label = (
            str(msg.layout.dim[0].label).strip()
            if msg.layout.dim
            else ''
        )

        phase, _, class_name = label.partition(':')
        class_name = class_name.strip() or None

        if phase not in valid_phases:
            return

        if phase in GRIP_PHASES and class_name not in GRIPPER_OPEN_DEG:
            return

        if phase in GRIP_PHASES and class_name not in GRIPPER_CLOSE_DEG:
            return

        joint_data = np.asarray(msg.data, dtype=float)

        if not np.all(np.isfinite(joint_data)):
            return

        waypoints = joint_data.reshape(-1, DOF)
        waypoints = np.clip(waypoints, JOINT_MIN, JOINT_MAX)

        q_start = self.read_current_q()

        try:
            trajectory = self.motion.build_phase_trajectory(
                phase,
                q_start,
                waypoints,
                class_name
            )
        except ValueError:
            return

        self.start_trajectory(
            phase,
            q_start,
            trajectory,
            class_name
        )

    def joint_target_callback(self, msg):
        if self.motion_active:
            self.get_logger().warning('로봇팔 동작 중')
            return

        if (len(msg.data) < DOF or len(msg.data) % DOF != 0):
            return

        joint_data = np.asarray(msg.data, dtype=float)

        if not np.all(np.isfinite(joint_data)):
            return

        waypoints = joint_data.reshape(-1, DOF)
        waypoints = np.clip(waypoints, JOINT_MIN, JOINT_MAX)

        q_start = self.read_current_q()
        trajectory = []

        q_previous = q_start.copy()

        ## waypoint가 따로 없을 때 이동 시간
        minimum_duration = 2.0 if len(waypoints) == 1 else 0.3
        for q_target in waypoints:
            self.motion.move(
                trajectory,
                q_previous,
                q_target,
                minimum_duration
            )

            q_previous = q_target.copy()

        self.start_trajectory('joint_target', q_start, trajectory)

    def start_trajectory(self, phase, q_start, trajectory, class_name=None):

        self.motion.connect_move_velocities(
            trajectory,
            velocity_scale=0.5
        )

        self.trajectory = trajectory
        self.total_duration = trajectory[-1]['end_time']
        self.final_target = trajectory[-1]['goal'].copy()
        self.q_cmd_prev = q_start.copy()
        self.motion_active = True
        self.start_time = time.monotonic()

        if phase == 'grip_pick':
            self.command_gripper(opened=True, class_name=class_name)

        elif phase == 'pack_pick':
            self.command_pneumatic(enabled=False)

        elif phase in {'pack_full', 'sauce_full'}:
            self.command_pneumatic(enabled=False)

    def control_loop(self):
        if not self.motion_active:
            return

        elapsed = time.monotonic() - self.start_time

        for segment in self.trajectory:
            if (
                'action' in segment
                and not segment['action_done']
                and elapsed >= segment['action_time']
            ):
                self.execute_action(segment['action'])
                segment['action_done'] = True

        current_segment = next(
            (
                segment
                for segment in self.trajectory
                if elapsed < segment['end_time']
            ),
            None
        )

        if current_segment is None:
            q_ref = self.final_target.copy()
            pack_horizontal = bool(self.trajectory[-1]['pack_horizontal'])
        elif current_segment['kind'] == 'move':
            q_ref = self.motion.quintic_joint(
                current_segment['start'],
                current_segment['goal'],
                current_segment['start_velocity'],
                current_segment['goal_velocity'],
                current_segment['start_acceleration'],
                current_segment['goal_acceleration'],
                elapsed - current_segment['start_time'],
                current_segment['duration']
            )
            pack_horizontal = bool(current_segment['pack_horizontal'])
        else:
            q_ref = current_segment['goal'].copy()
            pack_horizontal = bool(current_segment['pack_horizontal'])

        if pack_horizontal:
            q_ref[4] = self.motion.horizontal_q5(
                q_ref[1],
                q_ref[2],
                q_ref[3],
                self.q_cmd_prev[4]
            )

        q_ref = np.clip(
            q_ref,
            JOINT_MIN, JOINT_MAX
        )
        q_step = np.clip(
            motion_q_delta(q_ref, self.q_cmd_prev),
            -MAX_Q_STEP, MAX_Q_STEP
        )
        q_cmd = np.clip(
            self.q_cmd_prev + q_step,
            JOINT_MIN, JOINT_MAX
        )
        self.q_cmd_prev = q_cmd

        if ENABLE_MOTION:
            self.write_arm_positions(self.q_to_raw(q_cmd))

        if elapsed < self.total_duration:
            return

        finish_error_deg = float(np.max(np.abs(np.rad2deg(
            motion_q_delta(q_ref, q_cmd)
        ))))

        if finish_error_deg <= FINISH_TOLERANCE_DEG:
            self.motion_active = False
            self.start_time = None
            self.motion_done_pub.publish(Empty())

    def execute_action(self, action):
        if action.startswith('grip_open:'):
            class_name = action.split(':', 1)[1]
            self.command_gripper(
                opened=True,
                class_name=class_name
            )
        elif action.startswith('grip_close:'):
            class_name = action.split(':', 1)[1]
            self.command_gripper(
                opened=False,
                class_name=class_name
            )

        elif action == '공압 on':
            self.command_pneumatic(enabled=True)
        elif action == '공압 off':
            self.command_pneumatic(enabled=False)
        else:
            return

    def command_gripper(self, opened, class_name=None):
        if not ENABLE_MOTION:
            return

        if opened:
            if class_name not in GRIPPER_OPEN_DEG:
                return
        
            goal_raw = int(round(
                GRIPPER_HOME_RAW + GRIPPER_OPEN_DEG[class_name] * 4096.0 / 360.0
                    )) % 4096

        else:
            if class_name not in GRIPPER_CLOSE_DEG:
                return

            goal_raw = int(round(
                GRIPPER_HOME_RAW + GRIPPER_CLOSE_DEG[class_name] * 4096.0 / 360.0
        )) % 4096

        self._write4(
            GRIPPER_ID,
            ADDR_GOAL_POSITION,
            goal_raw
        )

    def command_pneumatic(self, enabled):
        if self.arduino is None or not self.arduino.is_open:
            return

        command = (
            b'ON\n'
            if enabled
            else b'OFF\n'
        )

        try:
            self.arduino.write(command)
            self.arduino.flush()

        except (serial.SerialException, OSError):
            return

    def read_current_q(self):
        return self.raw_to_q(self.read_arm_positions())

    def joint_state_request_callback(self, _msg):
        """
        transform_node가 비전 좌표를 받은 순간 요청하면
        실제 모터 Present Position을 한 번 읽어 발행한다. -> jointstate 발행
        """
        if self.motion_active:
            return

        q_actual = self.read_current_q()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base'
        msg.position = q_actual.tolist()
        msg.velocity = []
        msg.effort = []

        self.joint_state_pub.publish(msg)

    def hold_current_positions(self):
        current_raw = self.read_arm_positions()
        self.q_cmd_prev = self.raw_to_q(current_raw)

        gripper_raw = (
            self._read4(
                GRIPPER_ID,
                ADDR_PRESENT_POSITION
            ) % 4096
        )

        self.write_arm_positions(current_raw)
        self._write4(
            GRIPPER_ID,
            ADDR_GOAL_POSITION,
            gripper_raw
        )

    def raw_to_q(self, raw_list):
        q = np.zeros(DOF, dtype=float)

        for i in range(DOF):

            if i == 0:
                delta_tick = int(raw_list[i]) - int(HOME_RAW[i])

            else:
                delta_tick = signed_delta_tick(
                    raw_list[i], HOME_RAW[i]
                )

            motor_deg = delta_tick * 360.0 / 4096
            q[i] = math.radians(motor_deg)

        return q

    def q_to_raw(self, q):
        q = np.clip(q, JOINT_MIN, JOINT_MAX)
        raw_list = []

        for i in range(DOF):
            joint_deg = math.degrees(q[i])
            delta_tick = round(joint_deg * 4096 / 360.0)

            raw = int(HOME_RAW[i]) + int(delta_tick)

            if i != 0:
                raw %= 4096

            raw_list.append(raw)

        return raw_list

    def port(self, dxl_id):
        return (
            self.port_xh
            if dxl_id in XH_IDS
            else self.port_xm
        )

    def _write1(self, dxl_id, address, value):
        self.packet.write1ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address,
            int(value)
        )

    def _write4(self, dxl_id, address, value):
        self.packet.write4ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address,
            to_u32(value)
        )

    def _read4(self, dxl_id, address):
        value, _, _= self.packet.read4ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address
        )
        return int(value)

    def setup_motors(self):
        if not self.port_xh.openPort():
            raise RuntimeError
        self.xh_open = True

        if not self.port_xm.openPort():
            raise RuntimeError
        self.xm_open = True

        for dxl_id in ALL_IDS:
            self.packet.ping(
                self.port(dxl_id),
                dxl_id
            )

        self.set_all_torque(False)
        time.sleep(0.05)

        for dxl_id in ALL_IDS:

            operating_mode = (
                EXTENDED_POSITION_MODE
                if dxl_id == 1
                else POSITION_MODE
            )

            self._write1(
                dxl_id,
                ADDR_OPERATING_MODE,
                operating_mode
            )

            self._write4(
                dxl_id,
                ADDR_PROFILE_ACCELERATION,
                PROFILE_ACCELERATION
            )

            self._write4(
                dxl_id,
                ADDR_PROFILE_VELOCITY,
                PROFILE_VELOCITY
            )

    def set_all_torque(self, enabled):
        value = int(enabled)

        for dxl_id in ALL_IDS:
            self._write1(
                dxl_id,
                ADDR_TORQUE_ENABLE,
                value
            )

    def read_arm_positions(self):
        raw_list = []

        for dxl_id in ARM_IDS:
            raw = self._read4(
                dxl_id,
                ADDR_PRESENT_POSITION
            )

            if dxl_id == 1:
                raw = to_s32(raw)
            else:
                raw %= 4096

            raw_list.append(raw)

        return raw_list

    def write_arm_positions(self, raw_list):
        for dxl_id, raw in zip(ARM_IDS, raw_list):

            if dxl_id != 1:
                raw = int(raw) % 4096

            self._write4(
                dxl_id,
                ADDR_GOAL_POSITION,
                int(raw)
            )

    def shutdown(self):
        if self.closed:
            return

        self.closed = True
        self.timer.cancel()

        try:
            if self.arduino is not None and self.arduino.is_open:
                self.command_pneumatic(enabled=False)

        except Exception as exc:
            self.get_logger().error(f'공압 작동 안됨: {exc}')

        try:
            if self.xh_open and self.xm_open:
                self.hold_current_positions()
                time.sleep(0.05)
                self.set_all_torque(False)

        except Exception as exc:
            self.get_logger().error(f'모터 작동 안됨: {exc}')

        if self.arduino is not None and self.arduino.is_open:
            self.arduino.close()
            self.arduino = None

        if self.xh_open:
            self.port_xh.closePort()
            self.xh_open = False

        if self.xm_open:
            self.port_xm.closePort()
            self.xm_open = False


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = HardwareMotionControlNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        if node is None:
            print(f'[FATAL] {exc}')
        else:
            node.get_logger().fatal(str(exc))

    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
