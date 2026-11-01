#!/usr/bin/env python3

import math
import numpy as np

DOF = 6

CONTROL_READY = np.deg2rad([0.0, -90.0, 0.0, 113.0, 67.0, 0.0])

# 공압 그리퍼 TCP 기준 건네주기 좌표
HANDOFF_READY_POSITION = np.array([0.145, -0.25, 0.02], dtype=float)

HANDOFF_APPROACH_POSITION = HANDOFF_READY_POSITION + np.array([0.0, 0.0, 0.05], dtype=float)

HANDOFF_POSITION = HANDOFF_READY_POSITION + np.array([-0.30, 0.0, 0.0], dtype=float)

HANDOFF_LIFT_POSITION = HANDOFF_POSITION + np.array([0.0, 0.0, 0.05], dtype=float)


class ResetMotion:

    def __init__(self, kinematics):
        self.kinematics = kinematics

    def build_path(self):
        q_start = CONTROL_READY.copy()

        q_approach = self.kinematics.solve_pose(
            HANDOFF_APPROACH_POSITION,
            q_start,
            math.pi,
            'pack'
        )

        q_ready = self.kinematics.solve_pose(
            HANDOFF_READY_POSITION,
            q_approach,
            math.pi,
            'pack'
        )

        q_handoff = self.kinematics.solve_pose(
            HANDOFF_POSITION,
            q_ready,
            math.pi,
            'pack'
        )

        q_lift = self.kinematics.solve_pose(
            HANDOFF_LIFT_POSITION,
            q_handoff,
            math.pi,
            'pack'
        )

        q_home = CONTROL_READY.copy()

        return [
            q_approach,
            q_ready,
            q_handoff,
            q_lift,
            q_home
        ]