#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String
import serial

# ==================== 레일 설정 ====================
SERIAL_PORT = '/dev/ttyUSB0'
SERIAL_BAUD = 115200

# 원점 센서 = 0 rev 기준 절대 회전수
RAIL_ROTATIONS = {
    'noodle_thick': 12.5,
    'noodle_thin': 12.5,
    'sausage': 25.0,
    'crab': 37.5,
    'onion': 50.0,
    'mushroom': 62.5,
    'pepperoncino': 75.0,
    'cheese': 75.0,
    'sauce_cream': 87.5,
    'sauce_oil': 87.5,
    'sauce_tomato': 87.5,
    'cover': 100.0,
}


class RailBridge(Node):

    def __init__(self):
        super().__init__('rail_bridge')

        # MAIN -> RAIL
        self.create_subscription(Empty, '/rail/home', self.home_callback, 10)
        self.create_subscription(String, '/rail/motion', self.motion_callback, 10)

        # RAIL -> MAIN
        self.home_done_pub = self.create_publisher(String, '/rail/home_done', 10)
        self.motion_done_pub = self.create_publisher(String, '/rail/motion_done', 10)

        # Serial
        self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.0)

        self.rxbuf = b''
        self.move_home = False
        self.pending_class = None

        self.create_timer(0.02, self.poll_serial)

    # ==================== MAIN 명령 ====================

    def home_callback(self, _msg):
        if self.move_home:
            return

        if self.pending_class is not None:
            return

        self.move_home = True

        self.get_logger().info('레일 정렬 시작')

        self.send_serial('H')

    def motion_callback(self, msg):
        class_name = msg.data.strip()

        if self.move_home:
            self.get_logger().warning('레일 이동 중')
            return

        if self.pending_class is not None:
            self.get_logger().warning('레일 이동 중')
            return

        rotations = RAIL_ROTATIONS[class_name]
        self.pending_class = class_name

        self.get_logger().info(f'Rail 이동: {class_name}')

        self.send_serial(f'R{rotations:.2f}')

    # ==================== SERIAL ====================

    def send_serial(self, command):
        self.ser.write((command + '\n').encode())

        self.get_logger().info(f'Serial TX: {command}')

    def poll_serial(self):
        try:
            n = self.ser.in_waiting
            data = self.ser.read(n) if n else b''
        except Exception as e:
            self.get_logger().error(f'Serial read error: {e}')
            return

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

            return

        # ==================== ERROR ====================

        if line.startswith('ERR'):
            self.get_logger().error(f'Arduino error: {line}')
            self.move_home = False
            self.pending_class = None
            return

        # ==================== 원점 센서 ====================

        if 'KILL HIT' in line:
            self.move_home = False
            self.pending_class = None
            return

        # ==================== 일반 이동 완료 ====================

        if line.startswith('DONE'):
            if self.pending_class is None:
                return

            self.pending_class = None

            msg = String()
            msg.data = '완료'
            self.motion_done_pub.publish(msg)

            self.get_logger().info('Rail 이동 완료')


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