"""
밀키트 용기 뚜껑 인식

HSV 마스크 -> findContours -> minAreaRect
MORPH_CLOSE(작은 커널) -> 미세한 끊김 방지

파이프라인: 다운스케일 -> 검정 마스크 -> findContours -> minAreaRect
-> depth로 실제 cm 크기 환산 -> 17x24.5cm 허용오차 이내 검증
"""

import cv2
import numpy as np
import pyrealsense2 as rs

# ---------------- 설정값 ----------------

TARGET_W_CM = 17.0
TARGET_H_CM = 25

SIZE_TOLERANCE_CM = 2

HSV_V_MAX = 60
HSV_S_MAX = 100

DOWNSCALE = 0.5                  # 0.5 = 가로세로 절반 해상도로 처리 (연산량 1/4)
NOISE_OPEN_KERNEL = 3            # 다운스케일된 해상도 기준, 소금-후추 노이즈 제거용
CLOSE_KERNEL = 3                 # 미세한 끊김(1~2px) 보정용 안전장치

MIN_CONTOUR_AREA_PX = 150        # 다운스케일 해상도 기준, 이보다 작은 컨투어는 노이즈로 버림


def make_black_mask_small(color_small: np.ndarray) -> np.ndarray:
    """다운스케일된 컬러 이미지에서 검정 마스크 생성"""
    hsv = cv2.cvtColor(color_small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, HSV_S_MAX, HSV_V_MAX))
    open_kernel = np.ones((NOISE_OPEN_KERNEL, NOISE_OPEN_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    close_kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def contour_depth_z(contour, depth_small: np.ndarray, depth_scale: float):
    """컨투어 내부 영역의 depth 중앙값(m) 계산. 유효 depth 없으면 None"""
    x, y, w, h = cv2.boundingRect(contour)
    local_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(local_mask, [contour - (x, y)], -1, 255, thickness=-1)

    roi_depth = depth_small[y:y + h, x:x + w]
    valid = roi_depth[(roi_depth > 0) & (local_mask > 0)]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) * depth_scale


def rect_real_size_cm(contour, z_m: float, fx: float, fy: float):
    """컨투어의 minAreaRect + depth로 실제 cm 크기 환산.
    fx, fy는 원본(다운스케일 전) 해상도 기준 intrinsics 값."""
    rect = cv2.minAreaRect(contour)
    (_, _), (w_px_small, h_px_small), _ = rect

    w_px = w_px_small / DOWNSCALE
    h_px = h_px_small / DOWNSCALE

    f_avg = (fx + fy) / 2.0
    w_cm = (w_px * z_m / f_avg) * 100.0
    h_cm = (h_px * z_m / f_avg) * 100.0

    (rcx, rcy), (rw, rh), rangle = rect
    rect_full = ((rcx / DOWNSCALE, rcy / DOWNSCALE), (rw / DOWNSCALE, rh / DOWNSCALE), rangle)

    return rect_full, w_cm, h_cm


MIN_PLANE_POINTS = 80           # 평면 피팅에 필요한 최소 3D 점 수
MAX_PLANE_POINTS = 3000         # SVD 비용 상한 (이보다 많으면 균등 서브샘플)


def _deproject_region(contour, depth_small: np.ndarray, depth_scale: float,
                      intr: rs.intrinsics) -> np.ndarray:
    """컨투어 내부의 유효 depth 픽셀을 카메라 좌표계 3D 점군(m, N x 3)으로 변환.
    (핀홀 모델만 사용 - 왜곡계수는 방향 추정에는 무시 가능한 수준)"""
    x, y, w, h = cv2.boundingRect(contour)
    local = np.zeros((h, w), np.uint8)
    cv2.drawContours(local, [contour - (x, y)], -1, 255, thickness=-1)

    roi = depth_small[y:y + h, x:x + w].astype(np.float32) * depth_scale
    ys, xs = np.where((local > 0) & (roi > 0.0))
    if len(xs) == 0:
        return np.empty((0, 3), np.float64)

    if len(xs) > MAX_PLANE_POINTS:
        sel = np.linspace(0, len(xs) - 1, MAX_PLANE_POINTS).astype(np.intp)
        ys, xs = ys[sel], xs[sel]

    z = roi[ys, xs].astype(np.float64)
    px = (x + xs) / DOWNSCALE       # 원본 해상도 픽셀 좌표
    py = (y + ys) / DOWNSCALE
    X = (px - intr.ppx) / intr.fx * z
    Y = (py - intr.ppy) / intr.fy * z
    return np.stack([X, Y, z], axis=1)


def _project_points(pts3d: np.ndarray, intr: rs.intrinsics) -> np.ndarray:
    """카메라 좌표계 3D 점(m) -> 원본 해상도 픽셀 좌표 (N x 2)"""
    z = np.clip(pts3d[:, 2], 1e-6, None)
    u = pts3d[:, 0] / z * intr.fx + intr.ppx
    v = pts3d[:, 1] / z * intr.fy + intr.ppy
    return np.stack([u, v], axis=1)


def plane_obb(pts3d: np.ndarray):
    """3D 점군에 평면을 피팅하고 평면 위에서 최소면적 사각형(OBB)을 구한다.
    기울어진 물체도 실제 크기/방향을 얻을 수 있다.
    반환 dict: center3d, normal, corners3d(4 x 3), w_cm, h_cm, tilt_deg. 실패 시 None."""
    if len(pts3d) < MIN_PLANE_POINTS:
        return None

    c = pts3d.mean(axis=0)
    q = pts3d - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    normal = vt[2]

    # depth 이상치(가장자리 튐) 1회 제거 후 재피팅
    dist = q @ normal
    keep = np.abs(dist) < max(0.01, 2.0 * np.std(dist))     # 1cm 또는 2σ 이내
    if MIN_PLANE_POINTS <= keep.sum() < len(pts3d):
        c = pts3d[keep].mean(axis=0)
        q = pts3d[keep] - c
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        normal = vt[2]

    if normal[2] > 0:              # 카메라(-Z)를 향하도록 부호 통일
        normal = -normal

    # 평면 내 직교 기저 (u = 점군 최대 분산 방향 = 장축)
    u_ax = vt[0] - normal * (vt[0] @ normal)
    u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(normal, u_ax)

    a = q @ u_ax
    b = q @ v_ax
    rect = cv2.minAreaRect(np.stack([a, b], axis=1).astype(np.float32))
    (rca, rcb), (rw, rh), _ = rect
    corners2d = cv2.boxPoints(rect)

    corners3d = (c
                 + np.outer(corners2d[:, 0], u_ax)
                 + np.outer(corners2d[:, 1], v_ax))
    center3d = c + rca * u_ax + rcb * v_ax
    tilt_deg = float(np.degrees(np.arccos(np.clip(-normal[2], -1.0, 1.0))))

    return {
        "center3d": center3d, "normal": normal, "corners3d": corners3d,
        "w_cm": float(rw * 100.0), "h_cm": float(rh * 100.0),
        "tilt_deg": tilt_deg,
    }


def matches_target_size(w_cm: float, h_cm: float) -> bool:
    case1 = (abs(w_cm - TARGET_W_CM) <= SIZE_TOLERANCE_CM and
             abs(h_cm - TARGET_H_CM) <= SIZE_TOLERANCE_CM)
    case2 = (abs(w_cm - TARGET_H_CM) <= SIZE_TOLERANCE_CM and
             abs(h_cm - TARGET_W_CM) <= SIZE_TOLERANCE_CM)
    return case1 or case2


def detect_lid(color_image: np.ndarray, depth_image: np.ndarray,
                depth_scale: float, intr: rs.intrinsics):
    """뚜껑 인식 메인 함수. 반환: 후보 리스트 [{rect, w_cm, h_cm, ok}], 다운스케일 마스크"""
    small_size = (int(color_image.shape[1] * DOWNSCALE), int(color_image.shape[0] * DOWNSCALE))
    color_small = cv2.resize(color_image, small_size, interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth_image, small_size, interpolation=cv2.INTER_NEAREST)  # depth는 보간 금지

    mask = make_black_mask_small(color_small)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA_PX:
            continue

        z_m = contour_depth_z(contour, depth_small, depth_scale)
        if z_m is None:
            continue

        rect, w_cm, h_cm = rect_real_size_cm(contour, z_m, intr.fx, intr.fy)
        box_points = cv2.boxPoints(rect).astype(np.intp)
        cx, cy = rect[0]
        normal = None
        tilt_deg = 0.0

        # depth 평면 피팅: 기울어져 보여도 실제 방향/크기를 복원
        obb = plane_obb(_deproject_region(contour, depth_small, depth_scale, intr))
        if obb is not None:
            w_cm, h_cm = obb["w_cm"], obb["h_cm"]
            box_points = _project_points(obb["corners3d"], intr).astype(np.intp)
            cx, cy = _project_points(obb["center3d"][None, :], intr)[0]
            normal = obb["normal"]
            tilt_deg = obb["tilt_deg"]

        ok = matches_target_size(w_cm, h_cm)
        candidates.append({
            "rect": rect, "w_cm": w_cm, "h_cm": h_cm, "ok": ok,
            "z_m": z_m, "cx": float(cx), "cy": float(cy),
            "box_points": box_points,
            "normal": normal, "tilt_deg": tilt_deg,
        })

    return candidates, mask


def get_best_cover(color_image: np.ndarray, depth_image: np.ndarray,
                    depth_scale: float, intr: rs.intrinsics):
    """크기가 맞는 후보 중 depth(z_m)가 가장 가까운 것 하나만 반환.
    없으면 None. 메인 노드(mealkit_pub)에서 이 함수만 호출하면 됨."""
    candidates, _ = detect_lid(color_image, depth_image, depth_scale, intr)
    ok_candidates = [c for c in candidates if c["ok"]]
    if not ok_candidates:
        return None
    return min(ok_candidates, key=lambda c: c["z_m"])


MIN_DRAW_SIZE_CM = 15.0             # 시각화 최소 크기: 가로/세로 모두 이 값 이상인 박스만 그림


def draw_candidates(color_image: np.ndarray, candidates: list) -> np.ndarray:
    """가로/세로가 모두 MIN_DRAW_SIZE_CM 이상인 모든 후보 박스를 그림.
    평면 피팅으로 복원한 방향(장축 화살표)과 기울기(tilt)를 함께 표시.
    목표 크기에 맞는 후보(ok)는 초록, 그 외는 빨강."""
    vis = color_image.copy()
    for c in candidates:
        if c["w_cm"] < MIN_DRAW_SIZE_CM or c["h_cm"] < MIN_DRAW_SIZE_CM:
            continue
        color = (0, 255, 0) if c["ok"] else (0, 0, 255)

        box = np.asarray(c["box_points"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(vis, [box], True, color, 2)

        # 장축(긴 변) 방향 화살표
        e1 = box[1] - box[0]
        e2 = box[2] - box[1]
        long_edge = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        mid = box.mean(axis=0)
        p1 = (mid - long_edge / 2).astype(int)
        p2 = (mid + long_edge / 2).astype(int)
        cv2.arrowedLine(vis, tuple(p1), tuple(p2), color, 2, tipLength=0.15)

        cx, cy = int(c["cx"]), int(c["cy"])
        label = f'{c["w_cm"]:.1f}x{c["h_cm"]:.1f}cm  tilt {c["tilt_deg"]:.0f}deg'
        cv2.putText(vis, label, (cx - 70, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            intr = depth_frame.profile.as_video_stream_profile().intrinsics
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            candidates, mask = detect_lid(color_image, depth_image, depth_scale, intr)
            vis = draw_candidates(color_image, candidates)

            cv2.imshow("lid detection", vis)
            cv2.imshow("black mask (small)", mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()