#coding:utf-8
import os
import re
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import Config
import argparse


def extract_scene_info(filename):
    match = re.match(r'([a-z]+-\d+)-cam\d+-rgb-(\d+)', filename)
    if match:
        scene = match.group(1)
        frame_num = int(match.group(2))
        return scene, frame_num
    return None, None


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1)


def read_yolo_label(label_path, img_w, img_h):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                boxes.append({'cls': cls_id, 'bbox': [x1, y1, x2, y2]})
    return boxes


def prepare_from_images(image_dir, label_dir, output_path, seq_length=30):
    pose_model = YOLO(Config.pose_model_path, task='pose')
    device = 0 if torch.cuda.is_available() else 'cpu'

    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    scenes = defaultdict(list)
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        clean_name = re.sub(r'_png\.rf\.[a-f0-9]+', '', base_name)
        clean_name = re.sub(r'\.rf\.[a-f0-9]+', '', clean_name)
        scene, frame_num = extract_scene_info(clean_name)
        if scene is not None:
            scenes[scene].append((frame_num, img_file))

    for scene in scenes:
        scenes[scene].sort(key=lambda x: x[0])

    all_sequences = []
    all_labels = []

    for scene, frames in scenes.items():
        scene_keypoints = []
        scene_labels = []

        for frame_num, img_file in frames:
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

            if not os.path.exists(label_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            gt_boxes = read_yolo_label(label_path, w, h)

            results = pose_model(img, conf=0.25, device=device, verbose=False)

            if results[0].keypoints is None or len(results[0].keypoints) == 0:
                continue

            det_boxes = results[0].boxes.xyxy.cpu().numpy()
            kpts = results[0].keypoints.xy.cpu().numpy()

            for gt in gt_boxes:
                best_iou = 0
                best_idx = -1
                for j, det_box in enumerate(det_boxes):
                    iou_val = compute_iou(gt['bbox'], det_box.tolist())
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = j

                if best_iou > 0.3 and best_idx < len(kpts):
                    x1, y1, x2, y2 = gt['bbox']
                    bw = max(x2 - x1, 1)
                    bh = max(y2 - y1, 1)
                    norm_kpts = kpts[best_idx].copy().astype(float)
                    norm_kpts[:, 0] = (norm_kpts[:, 0] - x1) / bw
                    norm_kpts[:, 1] = (norm_kpts[:, 1] - y1) / bh
                    norm_kpts = np.clip(norm_kpts, 0, 1)

                    scene_keypoints.append(norm_kpts.flatten())
                    scene_labels.append(gt['cls'])

        if len(scene_keypoints) >= seq_length:
            for start in range(0, len(scene_keypoints) - seq_length + 1, max(seq_length // 2, 1)):
                seq = scene_keypoints[start:start + seq_length]
                seq_labels = scene_labels[start:start + seq_length]
                label = max(set(seq_labels), key=seq_labels.count)
                all_sequences.append(np.array(seq))
                all_labels.append(label)
        elif len(scene_keypoints) > 0:
            seq = list(scene_keypoints)
            seq_labels = list(scene_labels)
            while len(seq) < seq_length:
                seq.append(scene_keypoints[-1])
                seq_labels.append(scene_labels[-1])
            label = max(set(seq_labels), key=seq_labels.count)
            all_sequences.append(np.array(seq))
            all_labels.append(label)

    if len(all_sequences) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path,
                 sequences=np.array(all_sequences),
                 labels=np.array(all_labels))
        print(f"数据已保存到 {output_path}")
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"共 {len(all_sequences)} 个序列，类别分布: {dict(zip(unique.tolist(), counts.tolist()))}")
    else:
        print("未能提取到有效的训练数据")


def prepare_from_videos(video_dir, output_path, seq_length=30, label_map=None):
    pose_model = YOLO(Config.pose_model_path, task='pose')
    detect_model = YOLO(Config.detect_model_path, task='detect')
    device = 0 if torch.cuda.is_available() else 'cpu'

    if label_map is None:
        label_map = {
            'fall': 1, '摔倒': 1,
            'near-fall': 2, '将要摔倒': 2, 'nearfall': 2,
            'sitting': 3, '坐下': 3, 'sit': 3,
            'standing': 4, '站立': 4, 'stand': 4,
            'walking': 5, '行走': 5, 'walk': 5,
            'bending': 0, '屈身': 0, 'bend': 0,
        }

    all_sequences = []
    all_labels = []

    video_files = [f for f in os.listdir(video_dir)
                   if f.lower().endswith(('.avi', '.mp4', '.wmv', '.mkv'))]

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_label = None

        name_lower = video_file.lower()
        for keyword, label_id in label_map.items():
            if keyword in name_lower:
                video_label = label_id
                break

        if video_label is None:
            print(f"无法确定视频 {video_file} 的标签，跳过")
            continue

        cap = cv2.VideoCapture(video_path)
        track_keypoints = defaultdict(list)
        track_labels = defaultdict(list)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose_results = pose_model.track(frame, conf=0.25, persist=True,
                                            device=device, verbose=False)
            det_results = detect_model(frame, conf=0.25, device=device, verbose=False)

            if pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
                boxes = pose_results[0].boxes
                kpts = pose_results[0].keypoints

                for i in range(len(boxes)):
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu())
                    else:
                        track_id = i

                    bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    kpt = kpts.xy[i].cpu().numpy()

                    x1, y1, x2, y2 = bbox
                    bw = max(x2 - x1, 1)
                    bh = max(y2 - y1, 1)
                    norm_kpts = kpt.copy().astype(float)
                    norm_kpts[:, 0] = (norm_kpts[:, 0] - x1) / bw
                    norm_kpts[:, 1] = (norm_kpts[:, 1] - y1) / bh
                    norm_kpts = np.clip(norm_kpts, 0, 1)

                    track_keypoints[track_id].append(norm_kpts.flatten())

                    if det_results[0].boxes is not None and len(det_results[0].boxes) > 0:
                        best_iou_val = 0
                        best_cls = video_label
                        for j in range(len(det_results[0].boxes)):
                            det_b = det_results[0].boxes.xyxy[j].cpu().numpy().tolist()
                            iou_val = compute_iou(bbox, det_b)
                            if iou_val > best_iou_val:
                                best_iou_val = iou_val
                                best_cls = int(det_results[0].boxes.cls[j].cpu())
                        track_labels[track_id].append(best_cls)
                    else:
                        track_labels[track_id].append(video_label)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count} 帧...")

        cap.release()

        for track_id, kpts_list in track_keypoints.items():
            if len(kpts_list) < seq_length:
                continue

            labels_list = track_labels[track_id]
            for start in range(0, len(kpts_list) - seq_length + 1, max(seq_length // 2, 1)):
                seq = kpts_list[start:start + seq_length]
                seq_labels = labels_list[start:start + seq_length]
                label = max(set(seq_labels), key=seq_labels.count)
                all_sequences.append(np.array(seq))
                all_labels.append(label)

        print(f"视频 {video_file} 处理完成，共 {frame_count} 帧")

    if len(all_sequences) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path,
                 sequences=np.array(all_sequences),
                 labels=np.array(all_labels))
        print(f"数据已保存到 {output_path}")
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"共 {len(all_sequences)} 个序列，类别分布: {dict(zip(unique.tolist(), counts.tolist()))}")
    else:
        print("未能提取到有效的训练数据")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='行为识别训练数据准备')
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'video'],
                        help='数据来源模式: image(图片数据集) 或 video(视频文件)')
    parser.add_argument('--input', type=str, default='dataset/act-dataset/train',
                        help='输入路径(图片模式为数据集目录，视频模式为视频目录)')
    parser.add_argument('--output', type=str, default='dataset/action_data.npz',
                        help='输出文件路径')
    parser.add_argument('--seq-length', type=int, default=30,
                        help='序列长度')

    args = parser.parse_args()

    if args.mode == 'image':
        image_dir = os.path.join(args.input, 'images')
        label_dir = os.path.join(args.input, 'labels')
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"图片目录 {image_dir} 或标注目录 {label_dir} 不存在")
            exit(1)
        prepare_from_images(image_dir, label_dir, args.output, args.seq_length)
    else:
        if not os.path.exists(args.input):
            print(f"视频目录 {args.input} 不存在")
            exit(1)
        prepare_from_videos(args.input, args.output, args.seq_length)
