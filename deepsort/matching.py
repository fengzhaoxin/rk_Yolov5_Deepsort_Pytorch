import numpy as np
import scipy.optimize
from .kalman_filter import chi2inv95

INFTY_COST = 1e+5  # 无穷大代价，用于无法匹配的情况

def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    """
    最小代价匹配：将检测与Track进行一一匹配，代价超过阈值则视为未匹配。
    常用于数据关联。
    :param distance_metric: 距离度量函数
    :param max_distance: 最大允许距离
    :param tracks: Track对象列表
    :param detections: Detection对象列表
    :param track_indices: 参与匹配的Track索引
    :param detection_indices: 参与匹配的Detection索引
    :return: (匹配对, 未匹配Track索引, 未匹配Detection索引)
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices
    # 计算代价矩阵
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    # 超过最大距离的直接赋为无穷大
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    # 匈牙利算法求最优匹配
    row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    # 统计未匹配的Detection
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    # 统计未匹配的Track
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    # 统计有效匹配
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections

def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None):
    """
    级联匹配：优先匹配最近刚丢失的Track，逐步扩大匹配范围。
    适合处理长时间未更新的Track。
    :param distance_metric: 距离度量函数
    :param max_distance: 最大允许距离
    :param cascade_depth: 级联深度（最大未更新帧数）
    :param tracks: Track对象列表
    :param detections: Detection对象列表
    :param track_indices: 参与匹配的Track索引
    :param detection_indices: 参与匹配的Detection索引
    :return: (匹配对, 未匹配Track索引, 未匹配Detection索引)
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
        # 只匹配time_since_update等于当前level的Track
        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:
            continue
        matches_l, _, unmatched_detections = min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=INFTY_COST, only_position=False):
    """
    用卡尔曼门控过滤代价矩阵，超出门控距离的赋为无穷大。
    :param kf: KalmanFilter对象
    :param cost_matrix: 原始代价矩阵
    :param tracks: Track对象列表
    :param detections: Detection对象列表
    :param track_indices: Track索引
    :param detection_indices: Detection索引
    :param gated_cost: 超出门控距离的代价值
    :param only_position: 是否只用位置分量
    :return: 过滤后的代价矩阵
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix

def iou(bbox, candidates):
    """
    计算一个检测框与一组候选框的IOU（交并比）。
    :param bbox: 单个检测框(x, y, w, h)
    :param candidates: 候选框数组(N, 4)
    :return: IOU数组
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis], np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis], np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    计算IOU代价矩阵，常用于未确认Track的数据关联。
    :param tracks: Track对象列表
    :param detections: Detection对象列表
    :param track_indices: Track索引
    :param detection_indices: Detection索引
    :return: IOU代价矩阵
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INFTY_COST
            continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix 