#!/usr/bin/env python3

import math
import numpy as np
from ikpy.chain import Chain
from ikpy.link import Link, OriginLink
from scipy.optimize import least_squares

DOF = 6

DH_D = np.array([0.1271, 0.0, 0.22000, 0.0, 0.0, 0.188046], dtype=float)
DH_A = np.array([0.0, 0.0, 0.0, 0.220, 0.0, 0.0], dtype=float)
DH_ALPHA = np.deg2rad([-90.0, 90.0, -90.0, 0.0, 90.0, 0.0])
DH_THETA = np.deg2rad([0.0, 0.0, 0.0, -90.0, 90.0, 0.0])

# 공압은 5번 모터 DH 계산에 회전행렬이 필요 없기 때문에 DH를 따로 정의
PACK_DH_THETA = DH_THETA.copy()
PACK_DH_THETA[4] = 0.0
PACK_DH_ALPHA = DH_ALPHA.copy()
PACK_DH_ALPHA[4] = 0.0

PACK_POSITION_WEIGHT = 100.0
PACK_Q5_LIMIT_WEIGHT = 100.0
PACK_CONTINUITY_WEIGHT = 0.02

JOINT_MIN = np.deg2rad([-170.0, -120.0, -170.0, -140.0, -120.0, -360.0])
JOINT_MAX = np.deg2rad([170.0, 120.0, 170.0, 140.0, 120.0, 360.0])


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


class DHLink(Link):

    def __init__(self, name, d, a, alpha, theta, bounds):
        super().__init__(
            name=name,
            length=max(abs(float(d)), abs(float(a)), 1.0e-9),
            bounds=bounds,
        )
        self.d = float(d)
        self.a = float(a)
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.joint_type = 'revolute'
        self.rotation = True
        self.translation = False

    def get_link_frame_matrix(self, actuator_parameter):
        return dh_matrix(
            float(actuator_parameter) + self.theta,
            self.d,
            self.a,
            self.alpha
        )


class Kinematics:

    def __init__(self, logger=None):
        self.logger = logger
        self.preferred_chain = self.make_chain((1, 3, 4))
        self.fallback_chain = self.make_chain((1, 2, 3, 4))
        self.fk_chain = self.make_chain(tuple())

    def solve_cp_path(self, position, previous_q, q6):
        if self.logger:
            self.logger.info(f'cp 경로 시작: target={position.tolist()}')

        q1_target = self.target_q1(position, previous_q[0])
        active_indices = np.array([1, 2, 3], dtype=int)
        candidates = []

        for seed in self.seeds(previous_q, q1_target):
            def build_q(active_q):
                q = previous_q.copy()
                q[0] = q1_target
                q[active_indices] = active_q
                q[4] = self.horizontal_q5(q[1], q[2], q[3], previous_q[4])
                q[5] = q6
                return q

            def residual(active_q):
                q = build_q(active_q)
                position_error = self.fk_matrix(q)[:3, 3] - position
                joint_delta = wrapped_q_delta(q[active_indices], previous_q[active_indices])
                return np.concatenate([100.0 * position_error, 0.02 * joint_delta])

            result = least_squares(
                residual,
                seed[active_indices],
                bounds=(JOINT_MIN[active_indices], JOINT_MAX[active_indices]),
                max_nfev=300
            )

            q = build_q(result.x)
            position_error = float(np.linalg.norm(self.fk_matrix(q)[:3, 3] - position))
            joint_motion = float(np.linalg.norm(wrapped_q_delta(q, previous_q)))
            candidates.append((position_error, joint_motion, q))

        if not candidates:
            raise RuntimeError('IK 자세 계산 실패')

        selected = min(candidates, key=lambda item: (item[0], item[1]))

        if self.logger:
            self.logger.info(
                f'cp IK 결과: target={position.tolist()}, '
                f'error={selected[0]:.4f}m, '
                f'q={np.rad2deg(selected[2]).round(1).tolist()}'
            )

        if selected[0] > 0.005:
            return

        return selected[2]

    def horizontal_q5(self, q2, q3, q4, reference_q5):
        base_q5 = math.atan2(
                math.cos(q2),
                math.sin(q2) * math.cos(q3)
            )- q4

        candidates = []

        for branch in (0.0, math.pi):
            for turn in (-2.0 * math.pi, 0.0, 2.0 * math.pi):
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

    def solve_grip_place_pose(self, target, previous_q, place_yaw):
        q = self.solve_point(target, previous_q)

        ##### Place할 때 q1 회전량만큼 목표 yaw 보정 -> 하면서 안맞으면 부호 바꾸기  #####
        corrected_yaw = wrap_to_pi(float(place_yaw) + float(q[0]))

        # 보정된 yaw를 기준으로 q6 계산
        q[5] = self.target_q6(corrected_yaw, q, previous_q[5])

        return q

    def solve_pose(self, target, previous_q, yaw, mode):
        if mode == 'pack':
            q = self.solve_pack_point(target, previous_q)

            q[2] = 0 # 고정
            q[5] = previous_q[5]
            return q

        q = self.solve_point(target, previous_q)
        q[5] = self.target_q6(yaw, q, previous_q[5])
        return q

    def make_chain(self, active_joints):
        links = [OriginLink()]
        active_mask = [False]

        for i in range(DOF):
            links.append(
                DHLink(
                    name=f'joint_{i + 1}',
                    d=DH_D[i],
                    a=DH_A[i],
                    alpha=DH_ALPHA[i],
                    theta=DH_THETA[i],
                    bounds=(float(JOINT_MIN[i]), float(JOINT_MAX[i]))
                )
            )
            active_mask.append(i in active_joints)

        return Chain(
            name='irc_dh',
            links=links,
            active_links_mask=active_mask
        )

    @staticmethod
    def full_q(q):
        return np.concatenate(([0.0], q))

    def fk_matrix(self, q):
        return self.fk_chain.forward_kinematics(self.full_q(q))

    def pack_fk_q5(self, q):
        transform = np.eye(4, dtype=float)
        for i in range(5):
            transform = transform @ dh_matrix(
                q[i] + PACK_DH_THETA[i],
                DH_D[i],
                DH_A[i],
                PACK_DH_ALPHA[i]
            )
        return transform

    def pack_fk(self, q):
        transform = self.pack_fk_q5(q)

        return (
            transform[:3, 3] + transform[:3, :3] @ np.array([0.080845, 0.09195, 0.0], dtype=float) ## 5번 모터에서 공압 끝까지 translation 적용
        )

    @staticmethod
    def pack_q5(q2, q3, q4, branch, reference_q5):
        link_angle = math.atan2(math.cos(q2), math.sin(q2) * math.cos(q3))
        raw_q5 = link_angle + branch - q4
        return float(reference_q5 + wrap_to_pi(raw_q5 - reference_q5))

    def pack_seeds(self, previous_q, q1): ## 공압 자세 휴리스틱 후보 계산
        primary_postures = [
            (20.0, 20.0),
            (35.0, 30.0),
            (50.0, 40.0),
            (65.0, 55.0),
            (80.0, 65.0),
        ]

        result = []

        # q1은 무조건 목표 방향으로
        seed = previous_q.copy()
        seed[0] = q1
        seed[2] = 0

        result.append(np.clip(seed, JOINT_MIN, JOINT_MAX))

        for q2_deg, q4_deg in primary_postures:
            seed = previous_q.copy()

            seed[0] = q1
            seed[1] = math.radians(q2_deg)
            seed[2] = 0
            seed[3] = math.radians(q4_deg)

            result.append(np.clip(seed, JOINT_MIN, JOINT_MAX))
        return result

    def solve_pack_point(self, target, previous_q):
        q1_target = self.target_q1(target, previous_q[0])
        candidates = []

        for seed in self.pack_seeds(previous_q, q1_target):
            for branch in (0.0, math.pi):

                def build_q(active_q):
                    q = previous_q.copy()

                    # 공압을 사용할 때 해 결정 변수는 q1, q2, q4만 사용
                    q[np.array([0, 1, 3], dtype=int)] = active_q
                    q[2] = 0 # 고정
                    q[4] = self.pack_q5(
                        q[1],
                        q[2],
                        q[3],
                        branch,
                        previous_q[4]
                    )
                    return q

                def residual(active_q):
                    q = build_q(active_q)

                    q5_low = max(0.0, JOINT_MIN[4] - q[4])
                    q5_high = max(0.0, q[4] - JOINT_MAX[4])
                    joint_delta = wrapped_q_delta(
                        q[np.array([0, 1, 3], dtype=int)],
                        previous_q[np.array([0, 1, 3], dtype=int)]
                    )

                    q1_target_error = wrap_to_pi(q[0] - q1_target)

                    q2_backward = max(0.0, -q[1])
                    q4_backward = max(0.0, -q[3])

                    continuity_weights = np.array([0.25, 1.0, 1.0])

                    return np.concatenate([
                        PACK_POSITION_WEIGHT * (self.pack_fk(q) - target),

                        np.array([
                            PACK_Q5_LIMIT_WEIGHT * q5_low,
                            PACK_Q5_LIMIT_WEIGHT * q5_high,
                        ]),

                        PACK_CONTINUITY_WEIGHT * continuity_weights * joint_delta,

                        np.array([
                            2.0 * PACK_CONTINUITY_WEIGHT * q1_target_error,
                            2.0 * PACK_CONTINUITY_WEIGHT * q2_backward,
                            2.0 * PACK_CONTINUITY_WEIGHT * q4_backward,
                        ]),
                    ])

                result = least_squares(
                    residual,
                    seed[np.array([0, 1, 3], dtype=int)],
                    bounds=(
                        JOINT_MIN[np.array([0, 1, 3], dtype=int)],
                        JOINT_MAX[np.array([0, 1, 3], dtype=int)]
                    ),
                    max_nfev=300 ## 최적화 해 계산 300번 제한
                )

                q = build_q(result.x)

                if not (JOINT_MIN[4] <= q[4] <= JOINT_MAX[4]):
                    continue

                position_error = float(np.linalg.norm(self.pack_fk(q) - target))
                horizontal_error = abs(self.pack_fk_q5(q)[2, 0])

                joint_delta = wrapped_q_delta(
                    q[np.array([0, 1, 3], dtype=int)],
                    previous_q[np.array([0, 1, 3], dtype=int)]
                )

                q1_direction_error = abs(wrap_to_pi(q[0] - q1_target))

                q2_backward = max(0.0, -q[1])
                q4_backward = max(0.0, -q[3])

                total_motion = float(np.linalg.norm(joint_delta))

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
            return

        min_error = min(item[0] for item in candidates)

        selected = min(
            (
                item
                for item in candidates
                if item[0] <= min_error + 0.002 ## IK 후보 여유값
            ),
            key=lambda item: (
                item[1],  # 공압 링크 수평 오차
                item[2],  # q1 방향, q3 유지, 전방 자세 점수
                item[3],  # 전체 관절 이동량
            )
        )

        if selected[0] > 0.005:
            raise RuntimeError(f'IK가 닿을 수 없는 곳에 있음: {selected[0]:.4f} m')  ## tolerance 검사

        return selected[4] 
    
    def target_q6(self, target_yaw, q, previous_q6):
        q_without_q6 = q.copy()
        q_without_q6[5] = 0.0

        rotation = self.fk_matrix(q_without_q6)[:3, :3]
        current_yaw = math.atan2(
            float(rotation[1, 0]),
            float(rotation[0, 0])
        )
        desired = wrap_to_pi(target_yaw - current_yaw)

        candidates = np.array([
            desired - 2.0 * math.pi,
            desired,
            desired + 2.0 * math.pi,
        ])
        valid = candidates[(candidates >= JOINT_MIN[5]) & (candidates <= JOINT_MAX[5])]

        if valid.size == 0:
            return float(np.clip(
                desired,
                JOINT_MIN[5],
                JOINT_MAX[5]
            ))

        return float(valid[np.argmin(np.abs(valid - previous_q6))])

    def target_q1(self, target, previous_q1):
        xy = target[:2]

        if np.linalg.norm(xy) < 1.0e-8:
            return float(previous_q1)

        desired = wrap_to_pi(math.atan2(float(xy[1]), float(xy[0])))
        candidates = np.array([
            desired - 2.0 * math.pi,
            desired,
            desired + 2.0 * math.pi,
        ])
        valid = candidates[(candidates >= JOINT_MIN[0]) & (candidates <= JOINT_MAX[0])]

        if valid.size == 0:
            return float(np.clip(
                desired,
                JOINT_MIN[0], JOINT_MAX[0]
            ))

        return float(valid[np.argmin(np.abs(valid - previous_q1))])

    def seeds(self, previous_q, q1): ## 일반 그리파 휴리스틱 자세 후보 계산
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
            result.append(np.clip(seed, JOINT_MIN, JOINT_MAX))
        return result

    def candidates(self, chain, target, previous_q, q1):
        result = []

        for seed in self.seeds(previous_q, q1):
            full = chain.inverse_kinematics(
                target_position=target,
                initial_position=self.full_q(seed),
                max_iter=100
            )
            q = np.clip(full[1:], JOINT_MIN, JOINT_MAX)
            q[0] = q1
            q[5] = previous_q[5]

            error = float(np.linalg.norm(target - self.fk_matrix(q)[:3, 3]))
            q3_use = abs(wrap_to_pi(q[2] - previous_q[2]))
            continuity = float(np.linalg.norm(
                wrapped_q_delta(q[[1, 3, 4]], previous_q[[1, 3, 4]])
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
    def best(candidates):
        min_error = min(item[0] for item in candidates)
        return min(
            (
                item
                for item in candidates
                if item[0] <= min_error + 0.002
            ),
            key=lambda item: item[1]
        )

    def solve_point(self, target, previous_q):
        q1 = self.target_q1(target, previous_q[0])
        preferred = self.best(
            self.candidates(
                self.preferred_chain,
                target,
                previous_q,
                q1
            )
        )

        if preferred[0] <= 0.005:
            selected = preferred
        else:
            fallback = self.best(
                self.candidates(
                    self.fallback_chain,
                    target,
                    previous_q,
                    q1
                )
            )
            selected = (
                fallback
                if fallback[0] + 0.002 < preferred[0]
                else preferred
            )

        if selected[0] > 0.005:
            raise RuntimeError(f'IK 위치가 닿을 수 없는 곳에 있음: {selected[0]:.4f} m')
        return selected[2]