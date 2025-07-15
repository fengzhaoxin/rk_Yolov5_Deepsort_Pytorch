import numpy as np

def _cosine_distance(a, b):
    """
    计算两个特征集合之间的余弦距离。
    :param a: 特征数组A，shape=(N, D)
    :param b: 特征数组B，shape=(M, D)
    :return: 距离矩阵，shape=(N, M)
    """
    a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
    b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_cosine_distance(x, y):
    """
    计算每个y与x集合的最小余弦距离。
    :param x: 特征数组X，shape=(N, D)
    :param y: 特征数组Y，shape=(M, D)
    :return: shape=(M,)
    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

class NearestNeighborDistanceMetric:
    """
    最近邻距离度量，用于管理每个目标的特征库，并计算与新特征的距离。
    支持特征库上限（budget）。
    """
    def __init__(self, matching_threshold, budget=None):
        """
        :param matching_threshold: 匹配阈值（距离大于此值视为不匹配）
        :param budget: 每个目标最多保存多少历史特征
        """
        self._metric = _nn_cosine_distance
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}  # 每个track_id对应的特征列表
    def partial_fit(self, features, targets, active_targets):
        """
        更新特征库，只保留活跃目标的特征。
        :param features: 新特征数组
        :param targets: 新特征对应的track_id数组
        :param active_targets: 当前活跃的track_id列表
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}
    def distance(self, features, targets):
        """
        计算每个目标与新特征的距离矩阵。
        :param features: 新特征数组，shape=(N, D)
        :param targets: 目标track_id数组，shape=(M,)
        :return: 代价矩阵，shape=(M, N)
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix 