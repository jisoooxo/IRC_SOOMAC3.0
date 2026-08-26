'''
########### 퍼블리시하는 버전임

##### 나중에 추가할것
뎁스 같은 게 있을때 선택할 조건 추가하기 -> 박스 중앙에 있는 것 혹은 뎁스 균일도가 높은 걸로?
탑다운으로 본 다음 신호 주면, frame 10개정도 비교해서 가장 많이 나온 좌표로 보내주기?! -> 이건 통합할때 기준으로 하기
noodle 띠지는 색상 필터 추가하는게 나을듯 : 색상으로 면 굵기 구분 할거라면
탑다운애서 전채 프래임이 안 들어오면, 면 부분애서 위아래 컷 하는개 불안정해짐
#####
'''
#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from collections import deque

MODE = "real"  # "real" or "bag"
# PUBLISH_TARGET = "all"   # "sausage" / "crab" / "noodle" / "all"
TARGET_CLASSES = ["sausage", "crab", "noodle"] 
TARGET_SIZES = {
    "sausage": {"long": 0.10, "short": 0.02,  "Ltol": 0.01, "Stol": 0.01}, 
    "crab":    {"long": 0.08, "short": 0.025, "Ltol": 0.01, "Stol": 0.008}, # Ltolerance=0.01, Stolerance=0.005로 했었음
    "noodle":  {"long": 0.03, "short": 0.018,  "Ltol": 0.008, "Stol": 0.008}, # {"long": 0.03, "short": 0.018,  "Ltol": 0.008, "Stol": 0.008},
}

CROP_RATIOS = {
    "sausage": {"vertical": 0.05,   "horizontal": 0.07}, 
    "crab":    {"vertical": 0.05,   "horizontal": 0.07},
    "noodle":  {"vertical": 0.3, "horizontal": 0.05},   
}

# YOLO_PT_PATH = "/home/leejunmi/ros2_ws/src/vision/vision/best_6.pt"  
YOLO_PT_PATH = '/home/pc/irc_ws/irc_ws/src/irc_control_pkg/irc_control_pkg/best_6.pt'
yolo_model = YOLO(YOLO_PT_PATH)

# =====================
# SAM2
# =====================
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"         # 세번째로 작은 모델
# SAM2_CKPT   = "/home/leejunmi/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CKPT   = '/home/pc/sam2/checkpoints/sam2.1_hiera_base_plus.pt'
# SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"            # 두번째로 작은 모델
# SAM2_CKPT   = "/home/leejunmi/sam2/checkpoints/sam2.1_hiera_small.pt"
# SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"          # 가장 작은 모델
# SAM2_CKPT   = "/home/leejunmi/sam2/checkpoints/sam2.1_hiera_tiny.pt" 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

sam2_model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
predictor  = SAM2ImagePredictor(sam2_model)
print(f"SAM2 로딩 성공")

# =====================
# 초기 설정
# =====================
pipeline = rs.pipeline()
config = rs.config()

if MODE == "real":
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
elif MODE == "bag":
    config.enable_device_from_file("/home/leejunmi/realsense_bag/0729(2).db3")
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

align = rs.align(rs.stream.color)
spatial      = rs.spatial_filter()
temporal     = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()


def preprocess_depth(depth_frame):
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)
    depth_frame = depth_frame.as_depth_frame() # 필터 적용
    return depth_frame


def pixel_to_real_size(w_px, h_px, depth_m, intrinsics):
    fx = intrinsics.fx
    fy = intrinsics.fy
    real_w = w_px * depth_m / fx
    real_h = h_px * depth_m / fy
    return real_w, real_h

def get_min_depth_mask(depth_masked, depth_scale, tolerance_m=0.05, min_dist_px=80):
    roi_depth = depth_masked.astype(np.float32) * depth_scale # m 단위

    valid_all = roi_depth[roi_depth >= 0.2] # 최소 20cm 이상만
    if len(valid_all) == 0:
        return None

    min_depth = valid_all.min()
    depth_lower = max(0.2, min_depth)
    depth_upper = min_depth + tolerance_m

    ys, xs = np.where((roi_depth >= depth_lower) & (roi_depth <= depth_upper))
    if len(xs) == 0:
        return None

    print(f'min_depth:{min_depth:.3f}m')

    if len(xs) > 500:
        idx = np.random.choice(len(xs), 500, replace=False)
        xs, ys = xs[idx], ys[idx]

    points = list(zip(xs.tolist(), ys.tolist()))

    # 뎁스 마스크 받아서 최소뎁스로부터 5cm 더 보고 너무 가까운 점은 하나로 줄여서 후보점 생성
    return deduplicate_points(points, min_dist_px=min_dist_px)


def deduplicate_points(points, min_dist_px):
    kept = []
    for p in points:
        if all(np.hypot(p[0]-q[0], p[1]-q[1]) >= min_dist_px for q in kept):
            kept.append(p)
    return kept


def sam2_seg(depth_full, points_full, target):
    '''후보점을 댑스 오름차순으로 정렬, 가까운 것부터 크기 검증하고 처음 검출되는 댑스+1cm 해당하는 후보점까지만 연산 '''
    pts_with_depth = []
    for px, py in points_full:
        d = depth_full[py, px]
        if d > 0:
            pts_with_depth.append((d, px, py))
    pts_with_depth.sort() # 댑스 기준 오름차순으로 정렬

    found_boxes = []
    first_hit_depth = None

    for d, px, py in pts_with_depth:
        # 첫 검출 이후, depth가 1cm 이상 멀어지면 종료
        if first_hit_depth is not None and (d - first_hit_depth) * depth_scale > 0.01:
            break

        masks, scores, logits = predictor.predict( # 후보점 sam2 연산 
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        for mi in range(len(masks)):
            mask = masks[mi].astype(bool)
            rotate_box = get_rotated_bbox_from_mask(mask) # 마스크 감싸는 OBB 검출
            if rotate_box is None:
                continue
            ok, real_long, real_short = check_real_size( # 크기 검증
                rotate_box, mask, depth_full, depth_scale, intrinsics, target=target)
            if ok: # 통과하면 후보점에 추가
                ys, xs = np.where(mask > 0)
                valid = depth_full[ys, xs]
                valid = valid[valid > 0]
                rotate_box["mask"] = mask
                rotate_box["center_depth"] = np.median(valid) if len(valid) > 0 else float('inf')
                rotate_box["real_long"] = real_long
                rotate_box["real_short"] = real_short
                found_boxes.append(rotate_box)

                if first_hit_depth is None:
                    first_hit_depth = d   # 첫 성공 depth 기록
                break   # 이 후보점에선 마스크 하나만

    if not found_boxes:
        return None, None

    best_box = min(found_boxes, key=lambda x: x["center_depth"])

    return found_boxes, best_box


def get_rotated_bbox_from_mask(mask):
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect

    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)

    long_side = max(w, h)
    short_side = min(w, h)

    return {
        "center": (cx, cy),
        "box_points": box_points,
        "angle": angle,
        "long_side_px": long_side,
        "short_side_px": short_side,
    }


def check_real_size(rotate_box, mask, depth_img, depth_scale, intrinsics, target, fill_ratio_threshold=0.8):
    ys, xs = np.where(mask > 0)
    roi = depth_img[ys, xs]
    valid = roi[roi > 0]
    if len(valid) == 0:
        return False, None, None

    # 박스 면적 대비 마스크 채움 비율 체크(박스의 80퍼 이상) -> 잘못잡힌거 방지
    box_area = rotate_box["long_side_px"] * rotate_box["short_side_px"]
    mask_area = mask.sum()
    if box_area > 0 and (mask_area / box_area) < fill_ratio_threshold:
        return False, None, None

    depth_m = np.median(valid) * depth_scale
    real_long, real_short = pixel_to_real_size(
        rotate_box["long_side_px"], rotate_box["short_side_px"], depth_m, intrinsics
    )

    t = TARGET_SIZES[target]

    if target == 'noodle':
        def within(real, target_val, tol):
            diff = real - target_val
            return abs(diff) < tol
    else:
        def within(real, target_val, tol):
            diff = real - target_val
            allowed = tol if diff > 0 else tol / 2
            return abs(diff) < allowed

    ok = within(real_long, t["long"], t["Ltol"]) and \
         within(real_short, t["short"], t["Stol"])

    return ok, real_long, real_short


def get_points_in_box(box_points, n=5):
    """
    회전 박스 내부에 n개의 점 생성 (중심 + 네 모서리 방향 안쪽)
    box_points: 4개 꼭짓점 (풀 프레임 좌표)
    """
    box_points = np.array(box_points, dtype=np.float32)
    center = box_points.mean(axis=0)

    points = [tuple(center.astype(int))]  # 중심점

    # 중심과 각 꼭짓점 사이 중간 지점 (안쪽으로)
    for corner in box_points:
        mid = center + (corner - center) * 0.5   # 중심~꼭짓점의 중간
        points.append(tuple(mid.astype(int)))

    return points[:n]


def get_orientation_from_box(box_points):
    """
    box_points: minAreaRect의 4개 꼭짓점
    return: 긴 축이 y축과 이루는 각도 (시계방향 +, y축 평행=0, -90~90)
    """
    box_points = np.array(box_points, dtype=np.float32)

    # 네 변 중 가장 긴 변(긴 축) 찾기
    edges = [
        (box_points[1] - box_points[0]),
        (box_points[2] - box_points[1]),
    ]
    lengths = [np.linalg.norm(e) for e in edges]
    long_edge = edges[np.argmax(lengths)]

    dx, dy = long_edge[0], long_edge[1]

    # y축 기준 각도, 시계방향 +
    angle = np.degrees(np.arctan2(dx, dy))

    # -90 ~ 90 범위로 정규화 (긴 축은 180도 대칭)
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return -angle 


frame_toggle = 0  
def get_grid_points_checkerboard(depth_full, shrunk_poly, grid_cols=8, grid_rows=5, parity=0, overlay=None):
    # OBB 축정렬 bbox
    x1 = int(shrunk_poly[:,0].min())
    y1 = int(shrunk_poly[:,1].min())
    x2 = int(shrunk_poly[:,0].max())
    y2 = int(shrunk_poly[:,1].max())

    w = x2 - x1
    h = y2 - y1
    cell_w = w / grid_cols
    cell_h = h / grid_rows

    # OBB 마스크 (이 안에 있는 점만)
    H, W = depth_full.shape
    obb_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(obb_mask, [shrunk_poly.astype(np.int32)], 255)

    points = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if (row + col) % 2 != parity:
                continue
            cx = int(x1 + cell_w * (col + 0.5))
            cy = int(y1 + cell_h * (row + 0.5))

            # OBB 안에 있는 점만
            if 0 <= cy < H and 0 <= cx < W and obb_mask[cy, cx] > 0:
                points.append((cx, cy))
                # if overlay is not None:
                #     cv2.circle(overlay, (cx, cy), 3, (0, 255, 0), -1) # 시각화

    return points

def shrink_obb(cx, cy, w, h, r, top=0.0, bottom=0.0, left=0.0, right=0.0):
    """
    top/bottom: 긴 축 방향 위아래 제거 비율
    left/right: 짧은 축 방향 양옆 제거 비율
    """
    # 긴 쪽/짧은 쪽 판별
    if w >= h:
        # w가 긴 축 → w방향이 "상하", h방향이 "좌우"
        long_ratio_top    = top
        long_ratio_bottom = bottom
        short_ratio_left  = left
        short_ratio_right = right
        new_w = w * (1 - long_ratio_top - long_ratio_bottom)
        new_h = h * (1 - short_ratio_left - short_ratio_right)
        shift_along_long  = w * (long_ratio_bottom - long_ratio_top) / 2
        shift_along_short = h * (short_ratio_right - short_ratio_left) / 2
    else:
        # h가 긴 축 → h방향이 "상하", w방향이 "좌우"
        new_w = w * (1 - left - right)
        new_h = h * (1 - top - bottom)
        shift_along_long  = h * (bottom - top) / 2
        shift_along_short = w * (right - left) / 2

    cos_r, sin_r = np.cos(r), np.sin(r)

    if w >= h:
        new_cx = cx + shift_along_long * cos_r - shift_along_short * sin_r
        new_cy = cy + shift_along_long * sin_r + shift_along_short * cos_r
    else:
        new_cx = cx - shift_along_short * cos_r + shift_along_long * sin_r
        new_cy = cy - shift_along_short * sin_r - shift_along_long * cos_r

    dx, dy = new_w / 2, new_h / 2
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    return (corners @ R.T) + np.array([new_cx, new_cy])

def get_obb_masked_depth(yolo_result, class_names, target_class, depth_full, cls_name):
    ''' 특정 클래스 OBB에서 가장 큰 박스 찾고 박스 안의 depth 마스크 return
    depth_mask: 바운딩박스 내부 뎁스, shrunk_poly: 줄어든 박스 꼭짓점 '''
    if yolo_result.obb is None or len(yolo_result.obb) == 0:
        return None, None

    cls_ids = yolo_result.obb.cls.cpu().numpy()
    xywhr = yolo_result.obb.xywhr.cpu().numpy()

    best_area = -1
    best = None
    for i, cid in enumerate(cls_ids):
        if class_names[int(cid)] != target_class:
            continue
        cx, cy, w, h, r = xywhr[i]
        if w * h > best_area: # 면적 가장 큰 것 찾기
            best_area = w * h
            best = (cx, cy, w, h, r)

    if best is None:
        return None, None

    cx, cy, w, h, r = best
    ratios = CROP_RATIOS.get(cls_name, {"vertical": 0.0, "horizontal": 0.0})
    shrunk_poly = shrink_obb(cx, cy, w, h, r,
                             top=ratios["vertical"], bottom=ratios["vertical"],
                             left=ratios["horizontal"], right=ratios["horizontal"])

    H, W = depth_full.shape
    obb_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(obb_mask, [shrunk_poly.astype(np.int32)], 255)

    depth_masked = depth_full.copy()
    depth_masked[obb_mask == 0] = 0 # BB 밖 depth 0으로

    return depth_masked, shrunk_poly

def vision_start_callback(msg): # 수정
    global current_target
    data = msg.data.strip()
    if data in TARGET_CLASSES:
        current_target = data
        detection_history[data].clear() # 새 신호마다 10frame 새로 연산
        print(f"SAM2 시작: {current_target}")
    else: #data == "none":
        current_target = None
        print("SAM2 중지")

# def get_most_frequent_detection(history, pixel_threshold=20):
#     """
#     픽셀 좌표 기준으로 비슷한 위치끼리 그룹핑
#     가장 많이 검출된 그룹의 median 반환
#     pixel_threshold: 이 픽셀 이내면 같은 위치로 판단
#     """
#     if not history:
#         return None

#     items = list(history)
#     groups = []   # [(대표점, [같은 그룹 아이템들])]

#     for item in items:
#         cx, cy = item["cx"], item["cy"]
#         matched = False
#         for rep, group in groups:
#             # 대표점과 현재 점의 픽셀 거리
#             if np.hypot(cx - rep[0], cy - rep[1]) < pixel_threshold:
#                 group.append(item)
#                 matched = True
#                 break
#         if not matched:
#             groups.append(((cx, cy), [item]))

#     # 가장 많이 검출된 그룹
#     best_group = max(groups, key=lambda g: len(g[1]))[1]

#     xs = sorted(h["xyz"][0] for h in best_group)
#     ys = sorted(h["xyz"][1] for h in best_group)
#     zs = sorted(h["xyz"][2] for h in best_group)
#     angles = sorted(h["angle"] for h in best_group)
#     cxs = sorted(h["cx"] for h in best_group)   
#     cys = sorted(h["cy"] for h in best_group)  
#     mid = len(best_group) // 2

#     return {
#         "xyz": (xs[mid], ys[mid], zs[mid]),
#         "angle": angles[mid],
#         "cx_med": cxs[mid],   
#         "cy_med": cys[mid],  
#         "count": len(best_group),
#         "total": len(items),
#     }

def get_most_frequent_detection(history, pixel_threshold=20, min_count=2): ## 수정
    if not history:
        return None

    items = list(history)
    groups = []   # [(대표점, [아이템들])]

    for item in items:
        cx, cy = item["cx"], item["cy"]
        matched = False
        for rep, group in groups:
            if np.hypot(cx - rep[0], cy - rep[1]) < pixel_threshold:
                group.append(item)
                matched = True
                break
        if not matched:
            groups.append(((cx, cy), [item]))

    # 각 그룹의 median depth (z축)
    def group_median_depth(group):
        zs = sorted(h["xyz"][2] for h in group)
        return zs[len(zs) // 2]

    # depth 가장 작은 그룹
    min_depth_group = min(groups, key=lambda g: group_median_depth(g[1]))
    min_depth_items = min_depth_group[1]

    if len(min_depth_items) > min_count:
        best_group = min_depth_items
    else:
        # 가장 많이 나온 그룹
        best_group = max(groups, key=lambda g: len(g[1]))[1]

    xs = sorted(h["xyz"][0] for h in best_group)
    ys = sorted(h["xyz"][1] for h in best_group)
    zs = sorted(h["xyz"][2] for h in best_group)
    angles = sorted(h["angle"] for h in best_group)
    cxs = sorted(h["cx"] for h in best_group)
    cys = sorted(h["cy"] for h in best_group)
    mid = len(best_group) // 2

    return {
        "xyz": (xs[mid], ys[mid], zs[mid]),
        "angle": angles[mid],
        "cx_med": cxs[mid],
        "cy_med": cys[mid],
        "count": len(best_group),
        "total": len(items),
    }


def draw_last_published(overlay, last_published):
    if last_published is None:
        return
    lcx = int(last_published["cx"])
    lcy = int(last_published["cy"])
    lxyz = last_published["xyz"]
    langle = last_published["angle"]
    lcls = last_published["cls"]
    cv2.circle(overlay, (lcx, lcy), 10, (0, 255, 0), 3)
    cv2.putText(overlay, f"{lcls}", (lcx-40, lcy-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 0), 1)
    cv2.putText(overlay, f"x:{lxyz[0]:.3f} y:{lxyz[1]:.3f} z:{lxyz[2]:.3f}",
                (lcx-60, lcy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(overlay, f"yaw:{langle:.1f}deg",
                (lcx-30, lcy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# =====================
# main
# =====================
rclpy.init()
node = Node("mealkit_pub")
pub = node.create_publisher(String, '/vision/raw_pick_pose', 10)
sub = node.create_subscription(String, '/control/motion_done', vision_start_callback, 10) # 나중에 추가
print("node init")

current_target = None
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
prev_centers = {cls: [] for cls in TARGET_CLASSES}

num = 10 # 몇 프레임 보고나서 결정할건지
detection_history = {cls: deque(maxlen=num) for cls in TARGET_CLASSES}
last_published = None

try:
    while True:
        rclpy.spin_once(node, timeout_sec=0)
        frames  = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        depth_frame = preprocess_depth(depth_frame)

        if not color_frame or not depth_frame:
            continue
        
        frame_full = np.asanyarray(color_frame.get_data())  # BGR, 풀 프레임
        depth_full = np.asanyarray(depth_frame.get_data())
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        overlay = frame_full.copy()

        rgb_full = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
        yolo_result = yolo_model(frame_full, verbose=False)[0]

        if current_target is None:
            draw_last_published(overlay, last_published) # 결정된 퍼블리시 객체 표시
            cv2.imshow("frame", overlay)
            if cv2.waitKey(1) == 27:
                break
            continue

        predictor.set_image(rgb_full)  
                
        for cls_name in TARGET_CLASSES:
            if cls_name != current_target:   # current_target만 처리
                continue

            depth_masked, shrunk_poly = get_obb_masked_depth(
                yolo_result, yolo_model.names, cls_name, depth_full, cls_name)
            if depth_masked is None:
                prev_centers[cls_name] = []
                continue

            # OBB 시각화(줄인것)
            cv2.polylines(overlay, [shrunk_poly.astype(np.int32)], True, (255, 255, 0), 1)

            # 후보점 생성
            if cls_name != 'noodle':
                local_points = get_min_depth_mask(depth_masked, depth_scale)
                points_full = local_points if local_points else []
            else: # 면이면 후보영역 작게해서 검출
                parity = frame_toggle % 2
                frame_toggle += 1
                points_full = get_grid_points_checkerboard(
                    depth_full, shrunk_poly, grid_cols=8, grid_rows=5,
                    parity=parity, overlay=overlay)

            # 이전 프레임에서 검출된 재료 중앙점 이어붙이기
            if prev_centers[cls_name]:
                H, W = depth_full.shape
                obb_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(obb_mask, [shrunk_poly.astype(np.int32)], 255)
                for cxf, cyf in prev_centers[cls_name]:
                    if 0 <= cyf < H and 0 <= cxf < W and obb_mask[cyf, cxf] > 0:
                        if all(np.hypot(cxf-p[0], cyf-p[1]) > 30 for p in points_full):
                            points_full.append((cxf, cyf))

            if not points_full:
                prev_centers[cls_name] = []
                continue

            ##################### 후보점 프레임에 표시
            # if prev_centers[cls_name]:
            #     for cxf, cyf in prev_centers[cls_name]:
            #         if x1 <= cxf < x2 and y1 <= cyf < y2:
            #             if all(np.hypot(cxf - p[0], cyf - p[1]) > 30 for p in points_full):
            #                 points_full.append((cxf, cyf))

            # if not points_full:
            #     prev_centers[cls_name] = []
            #     continue
            # for px, py in points_full:
            #     cv2.circle(overlay, (int(px), int(py)), 3, (0, 255, 0), -1)
            ####################

            found_boxes, best_box = sam2_seg(depth_full, points_full, cls_name) # 후보점을 sam2로 넘기기 -> 크기+깊이 검증
            if found_boxes is None:
                prev_centers[cls_name] = []
                continue


            new_centers = []
            for box in found_boxes: # 후보 박스들
                is_best = (box is best_box)
                color = (0, 0, 255) if is_best else (0, 255, 255)

                cx_full = int(box["center"][0])
                cy_full = int(box["center"][1])
                cv2.circle(overlay, (cx_full, cy_full), 5, (0, 0, 0), -1)

                depth_m = box['center_depth'] * depth_scale
                text = f"{depth_m*100:.1f}cm"
                cv2.putText(overlay, text, (cx_full - 40, cy_full + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                if is_best and box.get("mask") is not None:
                    mask = box["mask"]
                    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                cv2.drawContours(overlay, [box["box_points"]], 0, color, 2)

                if is_best:
                    # best_box 안에 점 5개 생성해서 저장
                    pts5 = get_points_in_box(box["box_points"], n=5)
                    new_centers.extend(pts5)
                    # 시각화 
                    for px, py in pts5:
                        cv2.circle(overlay, (px, py), 2, (255, 0, 255), -1)
                else:
                    # 나머지 박스는 중앙점 하나만
                    new_centers.append((cx_full, cy_full))

            ###########
            ## publish 
            ###########
            if best_box is not None:
                cx = int(best_box["center"][0])
                cy = int(best_box["center"][1])
                depth_m = best_box["center_depth"] * depth_scale
                if depth_m > 0:
                    xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
                    angle = get_orientation_from_box(best_box["box_points"])

                    # 픽셀 좌표도 함께 저장
                    detection_history[cls_name].append({
                        "xyz": xyz, "angle": angle,
                        "cx": cx, "cy": cy  
                    })

                    N = num
                    if len(detection_history[cls_name]) >= N:
                        result = get_most_frequent_detection(detection_history[cls_name])
                        if result is not None:
                            x, y, z = result["xyz"]
                            msg = String()
                            msg.data = f"{cls_name},{x:.4f},{y:.4f},{z:.4f},{result['angle']:.2f}"
                            pub.publish(msg)
                            print(f"[{cls_name}] 발행: {msg.data}")

                            # 발행한 픽셀 좌표 저장 (화면 표시용)
                            last_published = {
                                "cls": cls_name,
                                "cx": result["cx_med"],   # 아래에서 추가
                                "cy": result["cy_med"],
                                "angle": result["angle"],
                                "xyz": result["xyz"],
                            }

                        detection_history[cls_name].clear()
                        current_target = None

        draw_last_published(overlay, last_published)  
        cv2.imshow("frame", overlay)
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
