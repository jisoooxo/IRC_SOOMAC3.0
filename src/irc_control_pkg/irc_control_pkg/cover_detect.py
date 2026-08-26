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
TARGET_H_CM = 24.5
SIZE_TOLERANCE_CM = 1.5

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
        ok = matches_target_size(w_cm, h_cm)
        box_points = cv2.boxPoints(rect).astype(np.intp)
        candidates.append({
            "rect": rect, "w_cm": w_cm, "h_cm": h_cm, "ok": ok,
            "z_m": z_m, "cx": rect[0][0], "cy": rect[0][1],
            "box_points": box_points,
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


def draw_candidates(color_image: np.ndarray, candidates: list) -> np.ndarray:
    """크기가 목표 범위(tolerance 이내)에 맞는 후보만 그림"""
    vis = color_image.copy()
    for c in candidates:
        if not c["ok"]:
            continue
        box = cv2.boxPoints(c["rect"]).astype(np.int32)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
        cx, cy = c["rect"][0]
        label = f'{c["w_cm"]:.1f}x{c["h_cm"]:.1f}cm'
        cv2.putText(vis, label, (int(cx) - 60, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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