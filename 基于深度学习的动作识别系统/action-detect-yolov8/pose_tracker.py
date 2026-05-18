#coding:utf-8
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import Config
import os


class PoseTracker:

    def __init__(self, model_path=None, conf=0.25, iou=0.7):
        if model_path is None:
            model_path = Config.pose_model_path
        self.conf = conf
        self.iou = iou
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.loaded = False
        self.use_fallback = False
        self.model = None

        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path, task='pose')
                self.model(np.zeros((48, 48, 3), dtype=np.uint8), device=self.device, verbose=False)
                self.loaded = True
                print(f"姿态估计模型加载成功: {model_path}")
            except Exception as e:
                print(f"姿态估计模型加载失败: {e}")

        if not self.loaded:
            fallback_path = Config.detect_model_path
            if os.path.exists(fallback_path):
                try:
                    print(f"尝试加载备用检测模型: {fallback_path}")
                    self.model = YOLO(fallback_path, task='detect')
                    self.model(np.zeros((48, 48, 3), dtype=np.uint8), device=self.device, verbose=False)
                    self.loaded = True
                    self.use_fallback = True
                    print("备用检测模型加载成功!")
                except Exception as e:
                    print(f"备用检测模型加载失败: {e}")
                    self.model = None
            else:
                print("=" * 50)
                print("警告: 没有可用的模型!")
                print("请下载 yolov8n-pose.pt 放到项目根目录")
                print("下载链接: https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt")
                print("=" * 50)

    def detect(self, frame):
        if not self.loaded:
            return []
        if self.use_fallback:
            results = self.model(frame, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)
            return self._parse_results_fallback(results[0])
        results = self.model(frame, conf=self.conf, iou=self.iou,
                             device=self.device, verbose=False)
        return self._parse_results(results[0], tracking=False)

    def track(self, frame, persist=True):
        if not self.loaded:
            return []
        if self.use_fallback:
            results = self.model.track(frame, conf=self.conf, iou=self.iou,
                                       persist=persist, device=self.device, verbose=False)
            return self._parse_results_fallback(results[0], tracking=True)
        results = self.model.track(frame, conf=self.conf, iou=self.iou,
                                   persist=persist, device=self.device, verbose=False)
        return self._parse_results(results[0], tracking=True)

    def _parse_results(self, result, tracking=False):
        persons = []
        if result.boxes is None or len(result.boxes) == 0:
            return persons

        boxes = result.boxes
        keypoints = result.keypoints

        for i in range(len(boxes)):
            person = {}
            person['bbox'] = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            person['conf'] = float(boxes.conf[i].cpu())
            person['cls'] = int(boxes.cls[i].cpu())

            if tracking and boxes.id is not None:
                person['track_id'] = int(boxes.id[i].cpu())
            else:
                person['track_id'] = -1

            if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) > i:
                person['keypoints'] = keypoints.xy[i].cpu().numpy()
                person['keypoints_conf'] = (keypoints.conf[i].cpu().numpy()
                                            if keypoints.conf is not None
                                            else np.ones(Config.num_keypoints))
            else:
                person['keypoints'] = np.zeros((Config.num_keypoints, 2))
                person['keypoints_conf'] = np.zeros(Config.num_keypoints)

            persons.append(person)

        return persons

    def _parse_results_fallback(self, result, tracking=False):
        persons = []
        if result.boxes is None or len(result.boxes) == 0:
            return persons

        boxes = result.boxes

        for i in range(len(boxes)):
            person = {}
            bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            person['bbox'] = bbox
            person['conf'] = float(boxes.conf[i].cpu())
            person['cls'] = int(boxes.cls[i].cpu())

            if tracking and boxes.id is not None:
                person['track_id'] = int(boxes.id[i].cpu())
            else:
                person['track_id'] = -1

            x1, y1, x2, y2 = bbox
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            cx, cy = x1 + w // 2, y1 + h // 2

            keypoints = np.array([
                [cx, cy],
                [cx - w * 0.1, cy - h * 0.15],
                [cx + w * 0.1, cy - h * 0.15],
                [cx - w * 0.15, cy - h * 0.1],
                [cx + w * 0.15, cy - h * 0.1],
                [cx - w * 0.2, cy],
                [cx + w * 0.2, cy],
                [cx - w * 0.25, cy + h * 0.15],
                [cx + w * 0.25, cy + h * 0.15],
                [cx - w * 0.3, cy + h * 0.3],
                [cx + w * 0.3, cy + h * 0.3],
                [cx - w * 0.15, cy + h * 0.35],
                [cx + w * 0.15, cy + h * 0.35],
                [cx - w * 0.15, cy + h * 0.6],
                [cx + w * 0.15, cy + h * 0.6],
                [cx - w * 0.15, cy + h * 0.85],
                [cx + w * 0.15, cy + h * 0.85],
            ], dtype=np.float32)

            person['keypoints'] = keypoints
            person['keypoints_conf'] = np.ones(Config.num_keypoints) * 0.5

            persons.append(person)

        return persons

    def reset_tracker(self):
        if self.loaded:
            self.model.predictor = None
