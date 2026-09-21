#!/usr/bin/env python3

import math
import numpy as np


DOF = 6

CONTROL_READY = np.deg2rad([0.0, -90.0, 0.0, 113.0, 67.0, 0.0])
CONTROL_READY2 = np.deg2rad([180.0, -90.0, 0.0, 113.0, 67.0, 0.0])

JOINT_MIN = np.deg2rad([-200.0, -120.0, -170.0, -140.0, -120.0, -360.0])
JOINT_MAX = np.deg2rad([550.0, 120.0, 170.0, 140.0, 120.0, 360.0])

MAX_Q_STEP = math.radians(2.0)


def wrapped_q_delta(q_goal, q_start):
    return (q_goal - q_start + np.pi) % (2.0 * np.pi) - np.pi


def motion_q_delta(q_goal, q_start):
    q_goal = np.asarray(q_goal, dtype=float)
    q_start = np.asarray(q_start, dtype=float)

    raw_delta = q_goal - q_start
    delta = wrapped_q_delta(q_goal, q_start)

    # q1이 정확히 +180° 이동이면 +방향 유지
    if np.isclose(delta[0], -math.pi, atol=1e-9) and raw_delta[0] > 0.0:
        delta[0] = math.pi

    # extended branch는 실제 multi-turn 값 유지
    if q_start[0] > math.radians(200.0) or q_goal[0] > math.radians(200.0):
        delta[0] = raw_delta[0]

    return delta


class MotionTrajectory:

    def build_phase_trajectory(self, phase, q_start, waypoints, class_name):
        trajectory = []

        if phase == 'spoon_pick':
            self.hold(
                trajectory, q_start, 2.0,
                action='grip_close:spoon',
                action_delay=0.3
            )
            return trajectory

        if phase == 'spoon_place':
            self.hold(
                trajectory, q_start, 2.0,
                action='grip_open:spoon',
                action_delay=0.3
            )
            return trajectory

        if phase == 'grip_pick':
            approach, pick, lift = [q.copy() for q in waypoints]

            self.build_grip_motion(
                trajectory, q_start, approach, pick, lift,
                f'grip_close:{class_name}'
            )

            self.hold(trajectory, lift, 0.5)
            return trajectory

        if phase == 'grip_place':
            approach, place, lift = [q.copy() for q in waypoints]

            self.build_grip_motion(
                trajectory, q_start, approach, place, lift,
                f'grip_open:{class_name}'
            )

            return trajectory

        if phase == 'pack_pick':
            approach, pick, lift = [q.copy() for q in waypoints]

            self.build_pack_motion(
                trajectory, q_start, approach, pick, lift,
                '공압 on'
            )

            self.hold(trajectory, lift, 0.5, pack_horizontal=True)
            return trajectory

        if phase == 'pack_place':
            if len(waypoints) == 4:
                q_mid, approach, place, lift = [q.copy() for q in waypoints]

                self.move(trajectory, q_start, q_mid, 2.0, True)
                self.hold(trajectory, q_mid, 0.1, pack_horizontal=True)

                self.build_pack_motion(
                    trajectory, q_mid, approach, place, lift,
                    '공압 off'
                )
                return trajectory

            approach, place, lift = [q.copy() for q in waypoints]

            self.build_pack_motion(
                trajectory, q_start, approach, place, lift,
                '공압 off'
            )
            return trajectory

        if phase == 'pack_full':
            pick, pick_lift, place_lift, place = [q.copy() for q in waypoints]

            self.move(trajectory, q_start, CONTROL_READY, 2.0)
            self.hold(trajectory, CONTROL_READY, 1.0)

            self.move(trajectory, CONTROL_READY, pick_lift, 2.0, True)
            self.move(trajectory, pick_lift, pick, 2.0, True)

            self.hold(
                trajectory, pick, 1.0,
                action='공압 on',
                action_delay=0.1,
                pack_horizontal=True
            )

            self.move(trajectory, pick, pick_lift, 2.0, True)
            self.hold(trajectory, pick_lift, 0.5, pack_horizontal=True)

            self.move(trajectory, pick_lift, place_lift, 6.0, True)
            self.move(trajectory, place_lift, place, 2.0, True)

            self.hold(
                trajectory, place, 1.0,
                action='공압 off',
                action_delay=0.1,
                pack_horizontal=True
            )

            self.move(trajectory, place, place_lift, 2.0, True)
            self.move(trajectory, place_lift, CONTROL_READY, 4.0)

            return trajectory

        if phase == 'sauce_full':
            pick, pick_lift, place_lift, place = [q.copy() for q in waypoints]
            q_home2 = CONTROL_READY2.copy()

            self.move(trajectory, q_start, q_home2, 2.0)
            self.hold(trajectory, q_home2, 0.5)

            self.move(trajectory, q_home2, pick_lift, 2.0, True)
            self.move(trajectory, pick_lift, pick, 2.0, True)

            self.hold(
                trajectory, pick, 1.0,
                action='공압 on',
                action_delay=0.1,
                pack_horizontal=True
            )

            self.move(trajectory, pick, pick_lift, 2.0, True)
            self.move(trajectory, pick_lift, place_lift, 6.0, True)
            self.move(trajectory, place_lift, place, 2.0, True)

            self.hold(
                trajectory, place, 1.0,
                action='공압 off',
                action_delay=0.1,
                pack_horizontal=True
            )

            self.move(trajectory, place, place_lift, 1.0, True)
            return trajectory

        raise ValueError(f'지원하지 않는 phase: {phase}')

    def build_grip_motion(self, trajectory, q_start, approach, target, lift, action):
        self.move(trajectory, q_start, approach, 2.0)
        self.hold(trajectory, approach, 0.5)

        self.move(trajectory, approach, target, 2.0)
        self.hold(
            trajectory, target, 1.5,
            action=action,
            action_delay=0.3
        )

        self.move(trajectory, target, lift, 1.0)

    def build_pack_motion(self, trajectory, q_start, approach, target, lift, action):
        self.move(trajectory, q_start, approach, 2.0, True)
        self.hold(trajectory, approach, 0.5, pack_horizontal=True)

        self.move(trajectory, approach, target, 2.0, True)
        self.hold(
            trajectory, target, 1.5,
            action=action,
            action_delay=0.3,
            pack_horizontal=True
        )

        self.move(trajectory, target, lift, 1.0, True)

    def move(self, trajectory, q_start, q_goal, minimum_duration, pack_horizontal=False):
        duration = self.safe_move_duration(q_start, q_goal, minimum_duration)
        start_time = trajectory[-1]['end_time'] if trajectory else 0.0

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
    def hold(trajectory, q_hold, duration, action=None, action_delay=0.0, pack_horizontal=False):
        start_time = trajectory[-1]['end_time'] if trajectory else 0.0

        segment = {
            'kind': 'hold',
            'goal': np.asarray(q_hold, dtype=float).copy(),
            'start_time': start_time,
            'end_time': start_time + float(duration),
            'pack_horizontal': bool(pack_horizontal),
        }

        if action is not None:
            segment['action'] = action
            segment['action_time'] = start_time + float(action_delay)
            segment['action_done'] = False

        trajectory.append(segment)

    @staticmethod
    def safe_move_duration(q_start, q_goal, minimum_duration):
        max_delta_deg = float(np.max(np.abs(np.rad2deg(
            motion_q_delta(q_goal, q_start)
        ))))

        max_speed_deg_s = math.degrees(MAX_Q_STEP) / 0.05
        required_time = 1.875 * max_delta_deg / max_speed_deg_s

        return max(float(minimum_duration), required_time * 1.2)

    @staticmethod
    def connect_move_velocities(trajectory, velocity_scale=0.5):
        for index in range(len(trajectory) - 1):
            previous_segment = trajectory[index]
            next_segment = trajectory[index + 1]

            if previous_segment['kind'] != 'move' or next_segment['kind'] != 'move':
                continue

            q_previous = previous_segment['start']
            q_middle = previous_segment['goal']
            q_next = next_segment['goal']

            previous_delta = motion_q_delta(q_middle, q_previous)
            next_delta = motion_q_delta(q_next, q_middle)

            same_direction = previous_delta * next_delta > 0.0

            v_middle = (
                velocity_scale
                * motion_q_delta(q_next, q_previous)
                / (previous_segment['duration'] + next_segment['duration'])
            )

            v_middle = np.where(same_direction, v_middle, 0.0)

            max_velocity = MAX_Q_STEP / 0.05
            v_middle = np.clip(v_middle, -max_velocity, max_velocity)

            previous_segment['goal_velocity'] = v_middle.copy()
            next_segment['start_velocity'] = v_middle.copy()

    @staticmethod
    def quintic_joint(
        q_start, q_goal,
        v_start, v_goal,
        a_start, a_goal,
        elapsed, duration
    ):
        q_start = np.asarray(q_start, dtype=float)
        q_goal = q_start + motion_q_delta(q_goal, q_start)

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

        return c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5

    @staticmethod
    def horizontal_q5(q2, q3, q4, reference_q5):
        base_q5 = math.atan2(
            math.cos(q2),
            math.sin(q2) * math.cos(q3)
        ) - q4

        candidates = []

        for branch in (0.0, math.pi):
            for turn in (-2.0 * math.pi, 0.0, 2.0 * math.pi):
                candidate = base_q5 + branch + turn

                if JOINT_MIN[4] <= candidate <= JOINT_MAX[4]:
                    candidates.append(candidate)

        if not candidates:
            return float(np.clip(base_q5, JOINT_MIN[4], JOINT_MAX[4]))

        return float(min(
            candidates,
            key=lambda value: abs(value - reference_q5)
        ))