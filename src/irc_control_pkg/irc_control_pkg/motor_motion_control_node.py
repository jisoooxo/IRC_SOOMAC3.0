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

PORT_XH = '/dev/ttyUSB0'
PORT_XM = '/dev/ttyUSB1'
ARDUINO_PORT = '/dev/ttyUSB2'

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

HOME_RAW = np.full(DOF, 2048, dtype=int)
GRIPPER_HOME_RAW = 2048

GRIPPER_CLOSE_DEG = {
    'noodle':   -70,
    'mushroom': -75,
    'onion':    -75,
    'crab':     -70,
    'sausage':  -65,
    'spoon':    -54
}

MIN_MOVE_TO_HOME_TIME = 3.0
MOVE_RETURN_HOME_TIME = 4.0

FINISH_TOLERANCE_DEG = 0.2

ENABLE_MOTION = True

PROFILE_VELOCITY = 20
PROFILE_ACCELERATION = 10

JOINT_MIN = np.deg2rad([-170.0, -120.0, -170.0, -140.0, -120.0, -360.0])
JOINT_MAX = np.deg2rad([170.0, 120.0, 170.0, 140.0, 120.0, 360.0])
MAX_Q_STEP = math.radians(2.0)

GRIP_PHASES = {'grip_pick', 'grip_place'}
PACK_PHASES = {'pack_pick', 'pack_place', 'pack_full'}


def wrapped_q_delta(q_goal, q_start):
    return (q_goal - q_start + np.pi) % (2.0 * np.pi) - np.pi

def signed_delta_tick(raw_now, raw_home):
    return (int(raw_now) - int(raw_home) + 2048) % 4096 - 2048

def to_u32(value):
    return int(value) & 0xFFFFFFFF


class HardwareMotionControlNode(Node):

    def __init__(self):
        super().__init__('hardware_motion_control_node')

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

        self.create_subscription(Float64MultiArray, '/arm/joint_waypoints', self.grip_plan_callback, 10)
        self.create_subscription(Float64MultiArray, '/arm/joint_waypoints_pack', self.pack_plan_callback, 10)
        self.create_subscription(Float64MultiArray, '/arm/joint_target', self.joint_target_callback, 10)

        self.motion_done_pub = self.create_publisher(Empty, '/arm/motion_done', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_state_request_sub = self.create_subscription(Empty, '/arm/request_joint_state', self.joint_state_request_callback, 10) # POINT1에서 실제 관절각을 요청받았을 때만 사용 -> 변환행렬 계산에 필요
        # 모션 제어 주기
        self.timer = self.create_timer(0.05, self.control_loop)

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

        if len(msg.data) != DOF * 4:
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

        if phase in GRIP_PHASES and class_name not in GRIPPER_CLOSE_DEG:
            return

        joint_data = np.asarray(msg.data, dtype=float)

        if not np.all(np.isfinite(joint_data)):
            return

        waypoints = joint_data.reshape(4, DOF)
        waypoints = np.clip(waypoints, JOINT_MIN, JOINT_MAX)

        q_start = self.read_current_q()

        try:
            trajectory = self.build_phase_trajectory(
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
            trajectory
        )

    def joint_target_callback(self, msg):
        if self.motion_active:
            self.get_logger().warning('로봇팔이 동작 중')
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

        for q_target in waypoints:
            minimum_duration = (
                4.0  ## waypoint가 따로 없을 때 이동 시간
                if len(waypoints) == 1
                else 0.3
            )

            self.move(
                trajectory,
                q_previous,
                q_target,
                minimum_duration
            )

            q_previous = q_target.copy()

        self.start_trajectory('joint_target', q_start, trajectory)

    def start_trajectory(self, phase, q_start, trajectory):

        self._connect_move_velocities(
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
            self.command_gripper(opened=True)

        elif phase == 'pack_pick':
            self.command_pneumatic(enabled=False)

        elif phase == 'pack_full':
            self.command_gripper(opened=True)
            self.command_pneumatic(enabled=False)

    def build_phase_trajectory(self, phase, q_start, waypoints, class_name):
        w1, w2, w3, w4 = [waypoint.copy() for waypoint in waypoints]
        trajectory = []

        if phase == 'grip_pick':
            # 현재 위치 -> pick approach : 3.0초
            self.move(trajectory, q_start, w1, 2.0)

            # approach 위치에서 0.5초 대기
            self.hold(trajectory, w1, 0.5)

            # approach -> pick : 3.0초
            self.move(trajectory, w1, w2, 2.0)

            # pick 위치에서 대기했다가 gripper close
            self.hold(trajectory, w2, 0.5, action=f'grip_close:{class_name}', action_delay=0.3)

            # pick -> lift : 3.0초
            self.move(trajectory, w2, w3, 2.0)

            # lift 위치에서 0.5초 대기
            self.hold(trajectory, w3, 0.5)

            # lift -> transfer : 6.0초
            self.move(trajectory, w3, w4, 6.0)

            return trajectory

        if phase == 'grip_place':
            # 현재 위치 -> place approach : 3.0초
            self.move(trajectory, q_start, w1, 2.0)

            # approach 위치에서 0.5초 대기
            self.hold(trajectory, w1, 0.5)

            # approach -> place : 3.0초
            self.move(trajectory, w1, w2, 2.0)

            # place 위치에서 대기했다가 gripper open
            self.hold(trajectory, w2, 0.5, action='grip_open', action_delay=0.3)

            # place -> lift : 3.0초
            self.move(trajectory, w2, w3, 2.0)

            return trajectory

        if phase == 'pack_pick':
            self.move(trajectory, q_start, w1, 2.0, True)

            self.hold(trajectory, w1, 0.5, pack_horizontal=True)

            self.move(trajectory, w1, w2, 2.0, True)

            self.hold(trajectory, w2, 0.5, action='공압 on', action_delay=0.3, pack_horizontal=True)

            self.move(trajectory, w2, w3, 2.0, True)

            self.hold(trajectory, w3, 0.5, pack_horizontal=True)

            self.move(trajectory, w3, w4, 6.0, True)

            return trajectory

        if phase == 'pack_place':
            self.move(trajectory, q_start, w1, 2.0, True)

            self.hold(trajectory, w1, 0.5, pack_horizontal=True)

            self.move(trajectory, w1, w2, 2.0, True)

            self.hold(trajectory, w2, 0.5, action='공압 off', action_delay=0.3, pack_horizontal=True)

            self.move(trajectory, w2, w3, 2.0, True)

            return trajectory

        if phase == 'pack_full': ## 초기 용기 옮기기는 공압 pick, place를 한 번에 수행
            self.move(trajectory, q_start, self.q_home, 2.0)

            self.hold(trajectory, self.q_home, 1.0)

            self.move(trajectory, self.q_home, w2, 2.0, True)

            self.move(trajectory, w2, w1, 2.0, True)

            self.hold(trajectory, w1, 0.5, action='공압 on', action_delay=0.3, pack_horizontal=True)

            self.move(trajectory, w1, w2, 2.0, True)

            self.hold(trajectory, w2, 0.5, pack_horizontal=True)

            self.move(trajectory, w2, w3, 6.0, True)

            self.move(trajectory, w3, w4, 2.0, True)

            self.hold(trajectory, w4, 0.5, action='공압 off', action_delay=0.3, pack_horizontal=True)

            self.move(trajectory, w4, self.q_home, 4.0)

            return trajectory

        return

    def move(
        self,
        trajectory,
        q_start,
        q_goal,
        minimum_duration,
        pack_horizontal=False
    ):
        duration = self._safe_move_duration(
            q_start,
            q_goal,
            minimum_duration
        )
        start_time = (
            trajectory[-1]['end_time']
            if trajectory
            else 0.0
        )

        trajectory.append({
            'kind': 'move',
            'start': np.asarray(q_start, dtype=float).copy(),
            'goal': np.asarray(q_goal, dtype=float).copy(),

            'start_velocity': np.zeros(DOF, dtype=float),
            'goal_velocity': np.zeros(DOF, dtype=float),

            'start_acceleration': np.zeros(DOF, dtype=float),
            'goal_acceleration': np.zeros(DOF, dtype=float),

            'duration': duration,
            'start_time': start_time,
            'end_time': start_time + duration,
            'pack_horizontal': bool(pack_horizontal),
        })

    @staticmethod
    def hold(
        trajectory,
        q_hold,
        duration,
        action=None,
        action_delay=0.0,
        pack_horizontal=False
    ):
        start_time = (
            trajectory[-1]['end_time']
            if trajectory
            else 0.0
        )
        segment = {
            'kind': 'hold',
            'goal': np.asarray(q_hold, dtype=float).copy(),
            'start_time': start_time,
            'end_time': start_time + float(duration),
            'pack_horizontal': bool(pack_horizontal),
        }

        if action is not None:
            segment['action'] = action
            segment['action_time'] = (start_time + float(action_delay))
            segment['action_done'] = False

        trajectory.append(segment)

    def _safe_move_duration(self, q_start, q_goal, minimum_duration):
        max_delta_deg = float(np.max(np.abs(np.rad2deg(
            wrapped_q_delta(q_goal, q_start)
        ))))
        max_speed_deg_s = math.degrees(MAX_Q_STEP) / 0.05
        required_time = 1.875 * max_delta_deg / max_speed_deg_s

        return max(float(minimum_duration), required_time * 1.2)

    def _connect_move_velocities(self, trajectory, velocity_scale=0.5):
        for segment in trajectory:
            if segment['kind'] != 'move':
                continue

            segment['start_velocity'] = np.zeros(DOF, dtype=float)
            segment['goal_velocity'] = np.zeros(DOF, dtype=float)
            segment['start_acceleration'] = np.zeros(DOF, dtype=float)
            segment['goal_acceleration'] = np.zeros(DOF, dtype=float)

        # 바로 이어지는 move → move 구간만 속도를 연결한다.
        for index in range(len(trajectory) - 1):
            previous_segment = trajectory[index]
            next_segment = trajectory[index + 1]

            if (previous_segment['kind'] != 'move' or next_segment['kind'] != 'move'):
                continue

            q_previous = previous_segment['start']
            q_middle = previous_segment['goal']
            q_next = next_segment['goal']

            previous_delta = wrapped_q_delta(q_middle, q_previous)
            next_delta = wrapped_q_delta(q_next, q_middle)

            # 앞 구간과 뒤 구간의 진행 방향이 같은 관절만 중간 속도를 유지한다.
            same_direction = (
                previous_delta * next_delta > 0.0
            )

            v_middle = (
                velocity_scale
                * wrapped_q_delta(
                    q_next,
                    q_previous
                )/ (previous_segment['duration'] + next_segment['duration'])
            )

            v_middle = np.where(same_direction, v_middle, 0.0)

            # 기존 max_q_step 기준 속도보다 커지지 않도록 제한
            max_velocity = MAX_Q_STEP / 0.05

            v_middle = np.clip(
                v_middle,
                -max_velocity,
                max_velocity
            )

            previous_segment['goal_velocity'] = v_middle.copy()
            next_segment['start_velocity'] = v_middle.copy()

    @staticmethod
    def quintic_joint(
        q_start,
        q_goal,
        v_start,
        v_goal,
        a_start,
        a_goal,
        elapsed,
        duration
    ):
        q_start = np.asarray(q_start, dtype=float)
        q_goal = q_start + wrapped_q_delta(q_goal, q_start)

        v_start = np.asarray(v_start, dtype=float)
        v_goal = np.asarray(v_goal, dtype=float)
        a_start = np.asarray(a_start, dtype=float)
        a_goal = np.asarray(a_goal, dtype=float)

        if duration <= 0.0:
            return q_goal.copy()

        T = float(duration)
        t = float(np.clip(elapsed, 0.0, T))

        c0 = q_start
        c1 = v_start
        c2 = 0.5 * a_start

        c3 = (
            20.0 * (q_goal - q_start)
            - (12.0 * v_start + 8.0 * v_goal) * T
            - (3.0 * a_start - a_goal) * T**2
        ) / (2.0 * T**3)

        c4 = (
            30.0 * (q_start - q_goal)
            + (16.0 * v_start + 14.0 * v_goal) * T
            + (3.0 * a_start - 2.0 * a_goal) * T**2
        ) / (2.0 * T**4)

        c5 = (
            12.0 * (q_goal - q_start)
            - (6.0 * v_start + 6.0 * v_goal) * T
            - (a_start - a_goal) * T**2
        ) / (2.0 * T**5)

        return (
            c0
            + c1 * t
            + c2 * t**2
            + c3 * t**3
            + c4 * t**4
            + c5 * t**5
        )

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
            q_ref = self.quintic_joint(
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
            q_ref[4] = self.horizontal_q5(
                q_ref[1],
                q_ref[2],
                q_ref[3],
                self.q_cmd_prev[4]
            )

        q_ref = np.clip(
            q_ref,
            JOINT_MIN,
            JOINT_MAX
        )
        q_step = np.clip(
            wrapped_q_delta(q_ref, self.q_cmd_prev),
            -MAX_Q_STEP,
            MAX_Q_STEP
        )
        q_cmd = np.clip(
            self.q_cmd_prev + q_step,
            JOINT_MIN,
            JOINT_MAX
        )
        self.q_cmd_prev = q_cmd

        if ENABLE_MOTION:
            self.write_arm_positions(self.q_to_raw(q_cmd))

        if elapsed < self.total_duration:
            return

        finish_error_deg = float(np.max(np.abs(np.rad2deg(
            wrapped_q_delta(q_ref, q_cmd)
        ))))

        if finish_error_deg <= FINISH_TOLERANCE_DEG:
            self.motion_active = False
            self.start_time = None
            self.motion_done_pub.publish(Empty())

    def horizontal_q5(self, q2, q3, q4, reference_q5):
        base_q5 = (
            math.atan2(
                math.cos(q2),
                math.sin(q2) * math.cos(q3)
            )- q4
        )

        candidates = []
        for branch in (0.0, math.pi):
            for turn in (
                -2.0 * math.pi,
                0.0,
                2.0 * math.pi
            ):
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

    def execute_action(self, action):
        if action == 'grip_open':
            self.command_gripper(opened=True)
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
            goal_raw = GRIPPER_HOME_RAW

        else:
            if class_name not in GRIPPER_CLOSE_DEG:
                return

            goal_raw = int(round(
                GRIPPER_HOME_RAW + GRIPPER_CLOSE_DEG[class_name] * 4096.0 / 360.0
            )) % 4096

        self._write4(
            GRIPPER_ID,
            ADDR_GOAL_POSITION,
            goal_raw,
            'write gripper position'
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
                ADDR_PRESENT_POSITION,
                'read gripper position'
            ) % 4096
        )

        self.write_arm_positions(current_raw)
        self._write4(
            GRIPPER_ID,
            ADDR_GOAL_POSITION,
            gripper_raw,
            'hold gripper position'
        )

    def raw_to_q(self, raw_list):
        q = np.zeros(DOF, dtype=float)

        for i in range(DOF):
            delta_tick = signed_delta_tick(raw_list[i], HOME_RAW[i])
            motor_deg = delta_tick * 360.0 / 4096
            q[i] = math.radians(motor_deg)

        return q

    def q_to_raw(self, q):
        q = np.clip(q, JOINT_MIN, JOINT_MAX)
        raw_list = []

        for i in range(DOF):
            joint_deg = math.degrees(q[i])
            motor_deg = joint_deg
            delta_tick = round(motor_deg * 4096 / 360.0)
            raw_list.append((int(HOME_RAW[i]) + int(delta_tick)) % 4096)

        return raw_list

    def port(self, dxl_id):
        return (
            self.port_xh
            if dxl_id in XH_IDS
            else self.port_xm
        )

    def check(self, comm, error):
        if comm != 0:
            return

        if error != 0:
            return

    def _write1(
        self,
        dxl_id,
        address,
        value,
        action
    ):
        comm, error = self.packet.write1ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address,
            int(value)
        )
        self.check(comm, error)

    def _write4(
        self,
        dxl_id,
        address,
        value,
        action
    ):
        comm, error = self.packet.write4ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address,
            to_u32(value)
        )
        self.check(comm, error)

    def _read4(
        self,
        dxl_id,
        address,
        action
    ):
        value, comm, error = self.packet.read4ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address
        )
        self.check(comm, error)
        return int(value)

    def setup_motors(self):
        if not self.port_xh.openPort():
            raise RuntimeError
        self.xh_open = True

        if not self.port_xm.openPort():
            raise RuntimeError
        self.xm_open = True

        for dxl_id in ALL_IDS:
            _, comm, error = self.packet.ping(
                self.port(dxl_id),
                dxl_id
            )
            self.check(comm, error)

        self.set_all_torque(False)
        time.sleep(0.05)

        for dxl_id in ALL_IDS:
            self._write1(
                dxl_id,
                ADDR_OPERATING_MODE,
                POSITION_MODE,
                'set position mode'
            )
            self._write4(
                dxl_id,
                ADDR_PROFILE_ACCELERATION,
                PROFILE_ACCELERATION,
                'set profile acceleration'
            )
            self._write4(
                dxl_id,
                ADDR_PROFILE_VELOCITY,
                PROFILE_VELOCITY,
                'set profile velocity'
            )

    def set_all_torque(self, enabled):
        value = (
            1
            if enabled
            else 0
        )

        for dxl_id in ALL_IDS:
            self._write1(
                dxl_id,
                ADDR_TORQUE_ENABLE,
                value,
                'set torque'
            )

    def read_arm_positions(self):
        return [
            self._read4(
                dxl_id,
                ADDR_PRESENT_POSITION,
                'read position'
            ) % 4096
            for dxl_id in ARM_IDS
        ]

    def write_arm_positions(self, raw_list):
        for dxl_id, raw in zip(ARM_IDS, raw_list):
            self._write4(
                dxl_id,
                ADDR_GOAL_POSITION,
                int(raw) % 4096,
                'write position'
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
