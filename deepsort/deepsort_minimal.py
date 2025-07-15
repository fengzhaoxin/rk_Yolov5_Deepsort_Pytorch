import numpy as np
import torch
from .reid_model import Extractor
from .detection import Detection
from .distance_metric import NearestNeighborDistanceMetric
from .simple_tracker import SimpleTracker

class DeepSortMinimal:
    """
    极简版DeepSort多目标跟踪主类。
    用于结合检测框、ReID特征和卡尔曼滤波，实现目标的跨帧跟踪。
    适合新手理解和二次开发。
    """
    def __init__(self, model_path='checkpoint/ckpt.t7', max_dist=0.2, min_confidence=0.3, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        """
        初始化DeepSort跟踪器。
        :param model_path: ReID模型权重路径
        :param max_dist: 特征距离阈值
        :param min_confidence: 检测置信度阈值
        :param max_iou_distance: IOU最大距离阈值
        :param max_age: 允许丢失的最大帧数
        :param n_init: 新目标确认所需的最小帧数
        :param nn_budget: 特征库最大容量
        :param use_cuda: 是否使用GPU
        """
        self.min_confidence = min_confidence  # 检测置信度阈值
        self.extractor = Extractor(model_path, use_cuda=use_cuda)  # ReID特征提取器
        self.metric = NearestNeighborDistanceMetric(max_dist, nn_budget)  # 特征距离度量
        self.tracker = SimpleTracker(self.metric, max_iou_distance, max_age, n_init)  # 跟踪器
        self.width = None   # 当前帧宽度
        self.height = None  # 当前帧高度

    def update(self, bbox_xywh, confidences, ori_img):
        """
        用一帧的检测结果更新跟踪器，返回跟踪结果。
        :param bbox_xywh: 检测框（中心点x, y, w, h）数组，shape=(N,4)
        :param confidences: 检测置信度数组，shape=(N,)
        :param ori_img: 原始图像（用于裁剪ReID特征）
        :return: 跟踪结果数组，每行为[x1, y1, x2, y2, track_id]
        """
        # 记录当前帧的尺寸
        self.height, self.width = ori_img.shape[:2]
        # 提取每个检测框的ReID特征
        features = self._get_features(bbox_xywh, ori_img)
        # 将检测框格式从xywh转为tlwh（左上角x, y, w, h）
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        # 只保留置信度高于阈值的检测，构建Detection对象
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if conf > self.min_confidence]
        # 预测所有Track的状态（卡尔曼滤波）
        self.tracker.predict()
        # 用当前检测结果更新Track
        self.tracker.update(detections)
        # 整理输出结果
        outputs = []
        for track in self.tracker.tracks:
            # 只输出已确认且刚更新的Track
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
        """
        将检测框从中心点格式(x, y, w, h)转为左上角格式(x, y, w, h)。
        :param bbox_xywh: 检测框数组
        :return: tlwh格式检测框数组
        """
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
        """
        将检测框从左上角格式(x, y, w, h)转为左上右下格式(x1, y1, x2, y2)。
        并保证坐标不越界。
        :param bbox_tlwh: 单个检测框
        :return: x1, y1, x2, y2
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        """
        根据检测框裁剪图片区域，提取ReID特征。
        :param bbox_xywh: 检测框数组
        :param ori_img: 原始图像
        :return: 特征数组
        """
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