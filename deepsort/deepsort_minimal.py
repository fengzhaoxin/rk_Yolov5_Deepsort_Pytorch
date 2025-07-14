import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import scipy.linalg

# ====== ReID特征提取网络 ======
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super().__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)

def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample)]
        else:
            blocks += [BasicBlock(c_out, c_out)]
    return nn.Sequential(*blocks)

class Net(nn.Module):
    def __init__(self, num_classes=751, reid=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.layer1 = make_layers(64, 64, 2, False)
        self.layer2 = make_layers(64, 128, 2, True)
        self.layer3 = make_layers(128, 256, 2, True)
        self.layer4 = make_layers(256, 512, 2, True)
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.classifier(x)
        return x

class Extractor:
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

# ====== 跟踪相关 ======
class Detection:
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self._n_init = n_init
        self._max_age = max_age
    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
    def is_confirmed(self):
        return self.state == TrackState.Confirmed
    def is_deleted(self):
        return self.state == TrackState.Deleted

# ====== 卡尔曼滤波 ======
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}
class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               1e-5,
               10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance
    def predict(self, mean, covariance):
        std_pos = [self._std_weight_position * mean[3]] * 3 + [self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3]] * 3 + [self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance
    def project(self, mean, covariance):
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
    def gating_distance(self, mean, covariance, measurements, only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

# ====== 距离度量 ======
def _cosine_distance(a, b):
    a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
    b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

class NearestNeighborDistanceMetric:
    def __init__(self, matching_threshold, budget=None):
        self._metric = _nn_cosine_distance
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}
    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}
    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix

# ====== 数据关联 ======
INFTY_COST = 1e+5

def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
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
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:
            continue
        matches_l, _, unmatched_detections = min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=INFTY_COST, only_position=False):
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix

def iou(bbox, candidates):
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

# ====== 极简DeepSort主类 ======
class DeepSortMinimal:
    def __init__(self, model_path='checkpoint/ckpt.t7', max_dist=0.2, min_confidence=0.3, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.extractor = Extractor(model_path, use_cuda=use_cuda)
        self.metric = NearestNeighborDistanceMetric(max_dist, nn_budget)
        self.tracker = self._init_tracker(max_iou_distance, max_age, n_init)
        self.width = None
        self.height = None
    def _init_tracker(self, max_iou_distance, max_age, n_init):
        class SimpleTracker:
            def __init__(self, metric, max_iou_distance, max_age, n_init):
                self.metric = metric
                self.max_iou_distance = max_iou_distance
                self.max_age = max_age
                self.n_init = n_init
                self.kf = KalmanFilter()
                self.tracks = []
                self._next_id = 1
            def predict(self):
                for track in self.tracks:
                    track.predict(self.kf)
            def update(self, detections):
                matches, unmatched_tracks, unmatched_detections = self._match(detections)
                for track_idx, detection_idx in matches:
                    self.tracks[track_idx].update(self.kf, detections[detection_idx])
                for track_idx in unmatched_tracks:
                    self.tracks[track_idx].mark_missed()
                for detection_idx in unmatched_detections:
                    self._initiate_track(detections[detection_idx])
                self.tracks = [t for t in self.tracks if not t.is_deleted()]
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
                def gated_metric(tracks, dets, track_indices, detection_indices):
                    features = np.array([dets[i].feature for i in detection_indices])
                    targets = np.array([tracks[i].track_id for i in track_indices])
                    cost_matrix = self.metric.distance(features, targets)
                    cost_matrix = gate_cost_matrix(self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
                    return cost_matrix
                confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
                unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
                matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)
                iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
                unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
                matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(iou_cost, self.max_iou_distance, self.tracks, detections, iou_track_candidates, unmatched_detections)
                matches = matches_a + matches_b
                unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
                return matches, unmatched_tracks, unmatched_detections
            def _initiate_track(self, detection):
                mean, covariance = self.kf.initiate(detection.to_xyah())
                self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))
                self._next_id += 1
        return SimpleTracker(self.metric, max_iou_distance, max_age, n_init)
    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if conf > self.min_confidence]
        self.tracker.predict()
        self.tracker.update(detections)
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        else:
            bbox_tlwh = np.array(bbox_xywh)
        bbox_tlwh[:, 0] = bbox_tlwh[:, 0] - bbox_tlwh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_tlwh[:, 1] - bbox_tlwh[:, 3] / 2.
        return bbox_tlwh
    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x, y, w, h = box
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, ori_img.shape[1] - 1)
            y2 = min(y2, ori_img.shape[0] - 1)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features 