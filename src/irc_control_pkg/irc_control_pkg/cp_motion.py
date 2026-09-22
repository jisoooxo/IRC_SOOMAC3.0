#!/usr/bin/env python3

import math
import numpy as np

DOF = 6

SPOON_PICK_POSITION = np.array([0.249, 0.275, 0.3], dtype=float)
SPOON_Q6 = math.radians(-90.0)

class CPMotion:

    def __init__(self, kinematics):
        self.kinematics = kinematics

    def build_path(self, class_name, repeat_count=1):
        q_start = np.zeros(DOF)

        q_spoon_pick = self.kinematics.solve_cp_path(
            SPOON_PICK_POSITION, q_start, SPOON_Q6
        )

        q_spoon_pre_pick = q_spoon_pick.copy()
        q_spoon_pre_pick[0] -= math.radians(18.0)

        # 숟가락 잡은 후 lift
        spoon_lift_position = SPOON_PICK_POSITION.copy()
        spoon_lift_position[2] += 0.10

        q_spoon_lift = self.kinematics.solve_cp_path(
            spoon_lift_position, q_spoon_pick, SPOON_Q6
        )

        # 치즈 / 페퍼론치노 접근
        if class_name == 'cheese':
            q_approach_lift = np.deg2rad([3.5, -1.4, 0.0, 85.0, 91.5, 0.0])
            q_approach = np.deg2rad([3.5, 1.2, 0.0, 101.0, 73.0, 0.0])
        else:
            q_approach_lift = np.deg2rad([1.5, 2.6, 0.0, 85.0, 66.0, 0.0])
            q_approach = np.deg2rad([1.5, 15.4, 0.0, 85.0, 62.7, 0.0])

        # 푸기
        q_touch_1 = q_approach.copy()

        if class_name == 'cheese':
            q_touch_1[0] -= math.radians(95.0)
            q_touch_1[2] = math.radians(90.0)
        else:
            q_touch_1 = np.deg2rad([-90.5, 2.3, 90.0, 108.5, 45.5, -30.0])

        q_touch_2 = q_touch_1.copy()
        q_touch_2[1] = math.radians(-90.0)

        if class_name != 'cheese':
            q_touch_2[5] = 0.0

        # 푸고 나서 lift
        q_lift = q_touch_2.copy()
        q_lift[2] -= math.radians(40.0)
        q_lift[5] -= math.radians(40.0)

        # place 준비
        if class_name == 'cheese':
            q_place_ready = np.deg2rad([-119.0, -80.0, 90.0, 130.0, 42.8, 0.0])
        else:
            q_place_ready = np.deg2rad([-131.7, -80.0, 90.0, 131.2, 46.7, 0.0])

        q_release_1 = q_place_ready.copy()
        q_release_1[5] += math.radians(100.0)

        q_release_2 = q_release_1.copy()
        q_release_2[5] -= math.radians(100.0)

        commands_1 = [
            ('motion', [q_spoon_pre_pick, q_spoon_pick]),
            ('gripper', 'spoon_pick:spoon', q_spoon_pick),
            ('motion', [
                q_spoon_lift,
                q_approach_lift,
                q_approach,
                q_touch_1,
                q_touch_2,
                q_lift,
                q_place_ready,
                q_release_1,
                q_release_2
            ]),
            ('delay', 0.5)
        ]

        commands_2 = [
            ('motion', [
                q_approach_lift,
                q_approach,
                q_touch_1,
                q_touch_2,
                q_lift,
                q_place_ready,
                q_release_1,
                q_release_2
            ]),
            ('delay', 0.5)
        ]

        commands = commands_1.copy()

        for _ in range(repeat_count - 1):
            commands.extend(commands_2)

        commands.extend([
            ('motion', [q_spoon_lift, q_spoon_pick]),
            ('gripper', 'spoon_place:spoon', q_spoon_pick),
            ('motion', [q_spoon_pre_pick])
        ])

        return commands