import numpy as np
from .kalman_filter import KalmanFilter
from .detection import Track
from .matching import min_cost_matching, matching_cascade, gate_cost_matrix, iou_cost

class SimpleTracker:
    """
    简单的多目标跟踪器，管理所有Track对象，实现预测、更新、数据关联等功能。
    """
    def __init__(self, metric, max_iou_distance, max_age, n_init):
        """
        初始化跟踪器。
        :param metric: 距离度量对象（如最近邻余弦距离）
        :param max_iou_distance: IOU最大距离阈值
        :param max_age: 允许丢失的最大帧数
        :param n_init: 新目标确认所需的最小帧数
        """
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter()  # 卡尔曼滤波器
        self.tracks = []         # 当前所有Track对象
        self._next_id = 1        # 下一个Track的ID

    def predict(self):
        """
        对所有Track进行卡尔曼预测，更新其状态。
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """
        用新一帧的检测结果更新所有Track。
        包括数据关联、状态更新、新目标初始化、特征库维护等。
        :param detections: 检测结果列表（Detection对象）
        """
        # 关联检测与Track
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        # 更新已匹配的Track
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        # 标记未匹配的Track为missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        # 初始化新的Track
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        # 移除已删除的Track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # 更新特征库
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        """
        数据关联：将检测结果与现有Track进行匹配。
        优先用特征距离，未确认Track用IOU。
        :param detections: 检测结果列表
        :return: (匹配对, 未匹配Track索引, 未匹配检测索引)
        """
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 计算特征距离，并用卡尔曼门控过滤
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix(self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
            return cost_matrix
        # 已确认和未确认Track分开处理
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        # 先用特征距离关联已确认Track
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)
        # 未确认Track和刚丢失的Track用IOU关联
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            iou_cost, self.max_iou_distance, self.tracks, detections, iou_track_candidates, unmatched_detections)
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """
        用未匹配的检测结果初始化新的Track。
        :param detection: Detection对象
        """
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))
        self._next_id += 1 