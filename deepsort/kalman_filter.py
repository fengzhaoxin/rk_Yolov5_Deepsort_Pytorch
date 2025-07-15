import numpy as np
import scipy.linalg

# 卡方分布阈值表，用于门控距离判断
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}

class KalmanFilter:
    """
    多目标跟踪用的标准卡尔曼滤波器。
    用于预测和更新目标的状态（位置、速度等）。
    """
    def __init__(self):
        """
        初始化卡尔曼滤波器参数。
        状态向量8维：[x, y, a, h, vx, vy, va, vh]
        其中x/y为中心，a为宽高比，h为高，vx等为速度。
        """
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt  # 位置和速度耦合
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        用初始检测框初始化卡尔曼状态。
        :param measurement: [x, y, a, h]，中心点、宽高比、高
        :return: 均值向量、协方差矩阵
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        # 位置和速度的初始方差
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
        """
        用运动模型预测下一时刻的状态。
        :param mean: 当前均值
        :param covariance: 当前协方差
        :return: 预测后的均值、协方差
        """
        std_pos = [self._std_weight_position * mean[3]] * 3 + [self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3]] * 3 + [self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """
        将状态投影到观测空间（只保留位置相关分量）。
        :param mean: 当前均值
        :param covariance: 当前协方差
        :return: 投影后的均值、协方差
        """
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """
        用观测值（检测框）更新卡尔曼状态。
        :param mean: 预测均值
        :param covariance: 预测协方差
        :param measurement: 观测值[x, y, a, h]
        :return: 更新后的均值、协方差
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        计算观测与预测的马氏距离，用于门控（判断是否关联）。
        :param mean: 预测均值
        :param covariance: 预测协方差
        :param measurements: 一组观测值
        :param only_position: 是否只用位置分量
        :return: 每个观测的马氏距离
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha 