# python track.py --source test.mp4 --save-vid --yolo_weights yolov5/weights/person_yolov5n.pt
import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from deepsort.deepsort_minimal import DeepSortMinimal
import argparse
import os
import time
import cv2
import torch

# img-size和output路径写死
IMG_SIZE = 640  # 输入图片分辨率
OUTPUT_DIR = 'inference'  # 输出文件夹

def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def detect(yolo_weights, source, conf_thres, iou_thres, device):
    dataset = LoadImages(source, img_size=IMG_SIZE)
    deepsort = DeepSortMinimal()
    device = select_device(device)
    model = attempt_load(yolo_weights, device=device)
    model.half() if device.type != 'cpu' else None
    model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device).type_as(next(model.parameters())))
    vid_writer = None
    save_path = OUTPUT_DIR + '/' + os.path.basename(source)
    for _, img, im0, vid_cap, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if device.type != 'cpu' else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time.time()
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0], agnostic=False)
        for det in pred:
            outputs = []
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                xywhs = torch.Tensor([xyxy_to_xywh(*xyxy) for *xyxy, _, _ in det])
                confss = torch.Tensor([[conf.item()] for *_, conf, _ in det])
                outputs = deepsort.update(xywhs, confss, im0)
                if len(outputs) > 0:
                    draw_boxes(im0, outputs[:, :4], outputs[:, -1])
            t2 = time.time()
            print(f'Done. (Detection time: {t2-t1:.3f}s, Persons: {len(outputs)})')
            if vid_writer is None:
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)
    print('Results saved to', save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, required=True, help='video file path')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    with torch.no_grad():
        detect(args.yolo_weights, args.source, args.conf_thres, args.iou_thres, args.device)
