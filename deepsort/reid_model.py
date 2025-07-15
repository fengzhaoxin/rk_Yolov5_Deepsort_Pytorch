import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np

class BasicBlock(nn.Module):
    """
    基础残差块，用于构建ReID特征提取网络。
    包含两个卷积层和可选的下采样。
    """
    def __init__(self, c_in, c_out, is_downsample=False):
        """
        :param c_in: 输入通道数
        :param c_out: 输出通道数
        :param is_downsample: 是否下采样（stride=2）
        """
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
        # 如果通道数不一致或需要下采样，使用1x1卷积调整
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
        """
        前向传播，包含残差连接。
        """
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)

def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    """
    构建多个BasicBlock串联的层。
    :param c_in: 输入通道数
    :param c_out: 输出通道数
    :param repeat_times: 重复次数
    :param is_downsample: 第一层是否下采样
    :return: nn.Sequential
    """
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample)]
        else:
            blocks += [BasicBlock(c_out, c_out)]
    return nn.Sequential(*blocks)

class Net(nn.Module):
    """
    ReID特征提取主网络。
    结构为多层卷积+残差块+池化。
    """
    def __init__(self, num_classes=751, reid=True):
        """
        :param num_classes: 分类类别数（用于训练）
        :param reid: 是否只输出归一化特征（True为只提特征，False为分类）
        """
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
        """
        前向传播，输出归一化特征或分类结果。
        """
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
    """
    ReID特征提取器，负责加载模型和批量提特征。
    """
    def __init__(self, model_path, use_cuda=True):
        """
        :param model_path: 模型权重路径
        :param use_cuda: 是否使用GPU
        """
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.size = (64, 128)  # 输入图片尺寸
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        预处理图片裁剪区域，归一化并转为Tensor。
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        """
        批量提取ReID特征。
        :param im_crops: 图片裁剪区域列表
        :return: 特征数组
        """
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy() 