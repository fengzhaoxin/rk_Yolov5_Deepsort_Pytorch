# YOLOv5+DeepSort行人跟踪

本项目为基于Pytorch的YOLOv5+DeepSort多目标跟踪，仅支持视频文件输入，并且只检测和跟踪“人”（类别0）。所有参数和流程已极简化，适合快速实验和二次开发。

## 特点
- 只支持视频文件输入（如mp4等），不支持摄像头、图片序列等。
- 只检测和跟踪类别0（人）。
- 输入分辨率(img-size)和输出目录(output)已写死，无需配置。
- 自动保存带跟踪ID的视频到`inference`目录。
- 每帧会打印检测耗时和检测到的人数。
- 代码结构极简，便于理解和修改。

## 依赖
- Python 3.7+
- PyTorch
- OpenCV
- yolov5（已集成在本项目）

## 用法

1. 准备好YOLOv5权重（建议使用只检测单目标的模型，如`person_yolov5n.pt`）。
2. 将待检测视频放在项目目录下，例如`test.mp4`。
3. 运行命令：

```bash
python track.py --yolo_weights yolov5/weights/person_yolov5n.pt --source test.mp4
```

- 结果视频会自动保存在`inference/test.mp4`。
- 运行过程中会在终端输出每帧检测耗时和人数。

## 代码说明
- `track.py`：主入口，极简化的检测+跟踪+视频保存流程。
- `deepsort/deepsort_minimal.py`：极简版DeepSort实现。

## DeepSort可调参数说明

- `REID_CKPT`: ReID模型权重路径，默认`checkpoint/ckpt.t7`
- `MAX_DIST`: 外观特征最大距离阈值，默认0.2
- `MIN_CONFIDENCE`: 检测框最小置信度，低于该值的目标会被过滤，默认0.3
- `NMS_MAX_OVERLAP`: NMS最大重叠阈值，默认0.5
- `MAX_IOU_DISTANCE`: IOU最大距离阈值，默认0.7
- `MAX_AGE`: 目标最大丢失帧数，超过则删除track，默认70
- `N_INIT`: 目标初始化所需连续检测帧数，默认3
- `NN_BUDGET`: 每个目标保存的最大特征数，默认100

如需自定义这些参数，可在`deepsort_minimal.py`中修改对应默认值。

## 注意事项
- 仅支持单个视频文件输入。
- 只检测和跟踪“单个目标”，若有多个目标则其它类别会被忽略。
- 输出视频参数与输入视频一致。

## 重要说明

- 本项目内置的`yolov5`目录为**rockchip**的yolov5代码。
- 如果更换为官方yolov5或其它第三方yolov5代码，可能因接口差异导致无法运行。
- 请勿随意替换`yolov5`目录，否则可能出现推理、数据加载等接口不兼容的问题。
- 如果需要训练自己的ReID模型，请参考 [ZQPei/deep_sort_pytorch.git](https://github.com/ZQPei/deep_sort_pytorch.git) 项目。

## 拉取项目注意事项

- 本项目的`yolov5`目录为**git submodule**，首次拉取项目请使用：

```bash
git clone --recurse-submodules <本项目地址>
```

- 如果已经拉取但未包含`yolov5`子模块，请执行：

```bash
git submodule update --init --recursive
```

- 请勿随意替换`yolov5`目录，否则可能出现接口不兼容问题。