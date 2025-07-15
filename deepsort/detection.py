import numpy as np

class Detection:
    """
    检测结果对象，包含检测框、置信度和ReID特征。
    """
    def __init__(self, tlwh, confidence, feature):
        """
        :param tlwh: 检测框，格式为(x, y, w, h)，左上角坐标
        :param confidence: 检测置信度
        :param feature: ReID特征向量
        """
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
    def to_xyah(self):
        """
        转换为(x, y, a, h)格式，x/y为中心点，a为宽高比，h为高。
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

class TrackState:
    """
    Track的生命周期状态枚举。
    """
    Tentative = 1  # 初始状态，未确认
    Confirmed = 2  # 已确认
    Deleted = 3    # 已删除

class Track:
    """
    单个目标的跟踪对象，包含卡尔曼状态、特征、生命周期等。
    """
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        """
        :param mean: 卡尔曼均值
        :param covariance: 卡尔曼协方差
        :param track_id: 跟踪ID
        :param n_init: 新目标确认所需帧数
        :param max_age: 最大丢失帧数
        :param feature: 初始ReID特征
        """
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
        """
        返回当前状态的检测框，格式为(x, y, w, h)。
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    def predict(self, kf):
        """
        用卡尔曼滤波器预测下一个状态。
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    def update(self, kf, detection):
        """
        用检测结果更新卡尔曼状态，并保存特征。
        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
    def mark_missed(self):
        """
        标记为missed，若超出最大丢失帧数则删除。
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
    def is_confirmed(self):
        """
        是否已确认。
        """
        return self.state == TrackState.Confirmed
    def is_deleted(self):
        """
        是否已被删除。
        """
        return self.state == TrackState.Deleted 