#coding:utf-8
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse


WEIZMANN_MAP = {
    'bend': 6,
    'walk': 0,
}

KTH_MAP = {
    'handwaving': 4,
    'running': 1,
    'walking': 0,
}

UCF101_MAP = {
    'WalkingWithDog': 0,
    'BodyWeightSquats': 6,
    'Punch': 4,
    'TaiChi': 3,
    'PushUps': 6,
    'PullUps': 6,
    'Lunges': 6,
    'JumpingJack': 4,
    'SkateBoarding': 1,
    'Skiing': 1,
}

ACTION_NAMES = {
    0: '行走',
    1: '跑步',
    2: '坐下',
    3: '站立',
    4: '举手',
    5: '摔倒',
    6: '弯腰',
}

NUM_KEYPOINTS = 17


def process_video(video_path, pose_model, device, seq_length=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0

    keypoints_sequence = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        try:
            results = pose_model(frame, conf=0.25, device=device, verbose=False)

            if results[0].keypoints is None or len(results[0].keypoints) == 0:
                continue

            kpts = results[0].keypoints.xy
            if kpts is None or len(kpts) == 0:
                continue

            kpt = kpts[0].cpu().numpy()
            if kpt.shape[0] != NUM_KEYPOINTS:
                continue

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                bbox = results[0].boxes.xyxy[0].cpu().numpy().astype(int).tolist()
            else:
                h, w = frame.shape[:2]
                valid_pts = kpt[kpt[:, 0] > 0]
                if len(valid_pts) == 0:
                    continue
                x_min = int(np.min(valid_pts[:, 0])) - 10
                y_min = int(np.min(valid_pts[:, 1])) - 10
                x_max = int(np.max(valid_pts[:, 0])) + 10
                y_max = int(np.max(valid_pts[:, 1])) + 10
                bbox = [max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)]

            x1, y1, x2, y2 = bbox
            bw = max(x2 - x1, 1)
            bh = max(y2 - y1, 1)

            norm_kpts = kpt.copy().astype(float)
            norm_kpts[:, 0] = (kpt[:, 0] - x1) / bw
            norm_kpts[:, 1] = (kpt[:, 1] - y1) / bh
            norm_kpts = np.clip(norm_kpts, 0, 1)

            feature = norm_kpts.flatten()
            keypoints_sequence.append(feature)

        except Exception:
            continue

    cap.release()

    sequences = []
    if len(keypoints_sequence) >= seq_length:
        for start in range(0, len(keypoints_sequence) - seq_length + 1, max(seq_length // 2, 1)):
            seq = keypoints_sequence[start:start + seq_length]
            sequences.append(np.array(seq))

    return sequences, frame_count


def process_all(data_dir, output_path, seq_length=30):
    print("正在加载姿态估计模型...")
    device = 0 if torch.cuda.is_available() else 'cpu'
    pose_model = YOLO('yolov8n-pose.pt', task='pose')

    all_sequences = []
    all_labels = []

    ucf_dir = os.path.join(data_dir, 'UCF-101')
    if os.path.exists(ucf_dir):
        print(f"\n===== UCF-101 数据集 =====")
        _process_dataset(ucf_dir, UCF101_MAP, pose_model, device, seq_length,
                         all_sequences, all_labels)

    flat_map = {}
    for k, v in WEIZMANN_MAP.items():
        flat_map[k] = v
    for k, v in KTH_MAP.items():
        flat_map[k] = v

    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if not os.path.isdir(item_path):
            continue
        if item == 'UCF-101':
            continue

        if item in flat_map:
            print(f"\n===== {item} ({ACTION_NAMES[flat_map[item]]}) =====")
            _process_flat_dir(item_path, flat_map[item], pose_model, device,
                              seq_length, all_sequences, all_labels, item)

    print(f"\n===== 从原始数据集提取摔倒数据 =====")
    _extract_fall_from_dataset(pose_model, device, seq_length,
                                all_sequences, all_labels)

    if len(all_sequences) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path,
                 sequences=np.array(all_sequences),
                 labels=np.array(all_labels))
        print(f"\n数据已保存到 {output_path}")
        print(f"共 {len(all_sequences)} 个序列")

        unique, counts = np.unique(all_labels, return_counts=True)
        print("类别分布:")
        for u, c in zip(unique, counts):
            print(f"  {ACTION_NAMES[u]}: {c} 个序列")
    else:
        print("未能提取到有效的训练数据")


def _process_dataset(dataset_dir, category_map, pose_model, device, seq_length,
                     all_sequences, all_labels):
    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for category in categories:
        if category not in category_map:
            continue

        label = category_map[category]
        category_path = os.path.join(dataset_dir, category)
        video_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.avi', '.mp4', '.wmv', '.mkv'))]

        print(f"\n处理类别: {category} ({ACTION_NAMES[label]}) - {len(video_files)} 个视频")

        for i, video_file in enumerate(video_files):
            video_path = os.path.join(category_path, video_file)
            print(f"  [{i+1}/{len(video_files)}] {video_file}", end=" ", flush=True)

            sequences, frame_count = process_video(video_path, pose_model, device, seq_length)

            if sequences:
                all_sequences.extend(sequences)
                all_labels.extend([label] * len(sequences))
                print(f"-> {frame_count}帧, 提取 {len(sequences)} 个序列", flush=True)
            else:
                print(f"-> {frame_count}帧, 无有效序列", flush=True)


def _process_flat_dir(dir_path, label, pose_model, device, seq_length,
                      all_sequences, all_labels, name=""):
    video_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.avi', '.mp4', '.wmv', '.mkv'))]

    print(f"\n处理: {name} ({ACTION_NAMES[label]}) - {len(video_files)} 个视频")

    for i, video_file in enumerate(video_files):
        video_path = os.path.join(dir_path, video_file)
        print(f"  [{i+1}/{len(video_files)}] {video_file}", end=" ", flush=True)

        sequences, frame_count = process_video(video_path, pose_model, device, seq_length)

        if sequences:
            all_sequences.extend(sequences)
            all_labels.extend([label] * len(sequences))
            print(f"-> {frame_count}帧, 提取 {len(sequences)} 个序列", flush=True)
        else:
            print(f"-> {frame_count}帧, 无有效序列", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='处理视频数据')
    parser.add_argument('--input', type=str, default='data11',
                        help='输入数据目录')
    parser.add_argument('--output', type=str, default='dataset/action_data.npz',
                        help='输出文件路径')
    parser.add_argument('--seq-length', type=int, default=30,
                        help='序列长度')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"输入目录 {args.input} 不存在")
        exit(1)

    process_all(args.input, args.output, args.seq_length)


def _extract_fall_from_dataset(pose_model, device, seq_length,
                                all_sequences, all_labels):
    from collections import defaultdict

    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join('dataset', 'act-dataset', split, 'images')
        label_dir = os.path.join('dataset', 'act-dataset', split, 'labels')
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            continue

        print(f"  处理 {split} 集...")
        image_files = sorted([f for f in os.listdir(img_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        fall_kpts = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(label_dir, base_name + '.txt')
            if not os.path.exists(label_path):
                continue

            has_fall = False
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) in [1, 2]:
                        has_fall = True
                        break

            if not has_fall:
                continue

            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            try:
                results = pose_model(img, conf=0.25, device=device, verbose=False)
                if results[0].keypoints is None or len(results[0].keypoints) == 0:
                    continue
                kpts = results[0].keypoints.xy
                if kpts is None or len(kpts) == 0:
                    continue
                kpt = kpts[0].cpu().numpy()
                if kpt.shape[0] != NUM_KEYPOINTS:
                    continue

                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    bbox = results[0].boxes.xyxy[0].cpu().numpy().astype(int).tolist()
                else:
                    h, w = img.shape[:2]
                    valid_pts = kpt[kpt[:, 0] > 0]
                    if len(valid_pts) == 0:
                        continue
                    bbox = [max(0, int(np.min(valid_pts[:, 0])) - 10),
                            max(0, int(np.min(valid_pts[:, 1])) - 10),
                            min(w, int(np.max(valid_pts[:, 0])) + 10),
                            min(h, int(np.max(valid_pts[:, 1])) + 10)]

                x1, y1, x2, y2 = bbox
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)
                norm_kpts = kpt.copy().astype(float)
                norm_kpts[:, 0] = (kpt[:, 0] - x1) / bw
                norm_kpts[:, 1] = (kpt[:, 1] - y1) / bh
                norm_kpts = np.clip(norm_kpts, 0, 1)
                fall_kpts.append(norm_kpts.flatten())
            except Exception:
                continue

        if len(fall_kpts) >= seq_length:
            for start in range(0, len(fall_kpts) - seq_length + 1, max(seq_length // 2, 1)):
                seq = fall_kpts[start:start + seq_length]
                all_sequences.append(np.array(seq))
                all_labels.append(5)
            print(f"    提取摔倒序列: {len(range(0, len(fall_kpts) - seq_length + 1, max(seq_length // 2, 1)))} 个")
        elif len(fall_kpts) > 0:
            while len(fall_kpts) < seq_length:
                fall_kpts.append(fall_kpts[-1])
            all_sequences.append(np.array(fall_kpts))
            all_labels.append(5)
            print(f"    提取摔倒序列: 1 个")
