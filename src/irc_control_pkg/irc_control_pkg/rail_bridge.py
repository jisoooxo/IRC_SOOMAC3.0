#!/usr/bin/env python3

import json
import rclpy
from collections import deque
from rclpy.node import Node
from std_msgs.msg import Empty, String
import serial

# ==================== 레일 설정 ====================
SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUD = 115200

# 원점 센서 = 0 rev 기준 절대 회전수
RAIL_ROTATIONS = {
    'noodle_thick': 0,
    'noodle_thin': 0,
    'sausage': 210,
    'crab': 285,
    'onion': 60,
    'mushroom': 138,
    'pepperoncino': 375,
    'cheese': 375,
    'sauce_cream': 371,
    'sauce_oil': 371,
    'sauce_tomato': 371,
    'cover': 375,
    # 건네주는 위치는 285
}


class RailBridge(Node):

    def __init__(self):
        super().__init__('rail_bridge')

        # MAIN -> RAIL
        self.create_subscription(Empty, '/rail/home', self.home_callback, 10)
        self.create_subscription(String, '/rail/move_ahead', self.move_ahead_callback, 10)
        self.create_subscription(String, '/rail/motion', self.motion_callback, 10)
        self.create_subscription(String, '/reset', self.reset_callback, 10)

        # RAIL -> MAIN
        self.home_done_pub = self.create_publisher(String, '/rail/home_done', 10)
        self.motion_done_pub = self.create_publisher(String, '/rail/motion_done', 10)

        # Serial
        self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.0)

        self.rxbuf = b''
        self.move_home = False
        self.pending_class = None
        self.pending_ahead = False

        self.command_queue = deque() ## 동작 중에 들어오는 토픽 저장

        self.create_timer(0.02, self.poll_serial)

    # ==================== MAIN 명령 ====================

    def is_busy(self):
        return (
            self.move_home or self.pending_class is not None
        )

    def enqueue_command(self, command_type, class_name):

        self.command_queue.append((command_type, class_name))
        self.get_logger().info('Rail 명령 저장')


    def execute_next_command(self):

        if self.is_busy():
            return

        if not self.command_queue:
            return

        command_type, class_name = self.command_queue.popleft()

        if command_type == 'ahead':
            self.start_move(
                class_name=class_name,
                ahead=True
            )

        elif command_type == 'motion':
            self.start_move(
                class_name=class_name,
                ahead=False
            )

    def start_move(self, class_name, ahead=False):

        rotations = RAIL_ROTATIONS[class_name]

        self.pending_class = class_name
        self.pending_ahead = ahead

        if ahead:
            self.get_logger().info(
                f'Rail 미리 이동 시작: '
                f'{class_name} -> {rotations:.2f}'
            )

        else:
            self.get_logger().info(
                f'Rail 이동 시작: '
                f'{class_name} -> {rotations:.2f}'
            )

        self.send_serial(
            f'R{rotations:.2f}'
        )


    def home_callback(self, _msg):

        if self.is_busy():
            return

        self.move_home = True

        self.get_logger().info('레일 정렬 시작')

        self.send_serial('H')

    def move_ahead_callback(self, msg):

        data = json.loads(msg.data)
        class_name = str(data['class']).strip()

        # 현재 이동 중이라면 버리지 않고 저장
        if self.is_busy():
            self.enqueue_command('ahead', class_name)
            return
        
        self.start_move(class_name=class_name, ahead=True)

    def motion_callback(self, msg):

        class_name = msg.data.strip()

        # 현재 이동 중이라면 버리지 않고 저장
        if self.is_busy():
            self.enqueue_command('motion', class_name)
            return

        # 바로 실행 가능
        self.start_move(class_name=class_name, ahead=False)

    def reset_callback(self, _msg):

        self.command_queue.clear()

        self.pending_class = None
        self.pending_ahead = False

        self.move_home = True

        self.get_logger().info('레일 초기화 시작')

        self.send_serial('H')

    # ==================== SERIAL ====================

    def send_serial(self, command):
        self.ser.write((command + '\n').encode())

        self.get_logger().info(f'Serial TX: {command}')

    def poll_serial(self):
    
        n = self.ser.in_waiting
        data = self.ser.read(n) if n else b''

        if not data:
            return

        self.rxbuf += data

        while b'\n' in self.rxbuf:
            line, self.rxbuf = self.rxbuf.split(b'\n', 1)

            line = line.decode(errors='ignore').strip()

            self.handle_serial(line)

    def handle_serial(self, line):
        if not line:
            return

        # Arduino 부팅
        if line.startswith('READY'):
            return

        # ==================== HOME 완료 ====================

        if line.startswith('HOME DONE'):

            if not self.move_home:
                return

            self.move_home = False

            msg = String()
            msg.data = '완료'
            self.home_done_pub.publish(msg)

            self.get_logger().info('Rail HOME 완료')

            self.execute_next_command()

            return

        # ==================== ERROR ====================

        if line.startswith('ERR'):

            self.get_logger().error(f'Arduino error: {line}')

            self.move_home = False
            self.pending_class = None
            self.pending_ahead = False

            return

        # ==================== 원점 센서 ====================

        if 'KILL HIT' in line:

            self.move_home = False
            self.pending_class = None
            self.pending_ahead = False

            self.get_logger().warning('Rail KILL HIT')
            return

        # ==================== 일반 이동 완료 ====================

        if line.startswith('DONE'):

            if self.pending_class is None:
                return

            class_name = self.pending_class
            ahead = self.pending_ahead

            self.pending_class = None
            self.pending_ahead = False

            # --------------------------------------------
            # ahead 이동
            # --------------------------------------------

            if ahead:

                self.get_logger().info(
                    f'Rail 단계별 미리 이동 완료: '
                    f'{class_name}'
                )
                self.execute_next_command()

                return

            # --------------------------------------------
            # 일반 motion
            # --------------------------------------------

            msg = String()
            msg.data = '완료'

            self.motion_done_pub.publish(msg)

            self.get_logger().info(f'Rail 이동 완료: {class_name}')

            # 다음 대기 명령 실행
            self.execute_next_command()


def main():

    rclpy.init()

    node = RailBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        try:
            node.ser.close()
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()