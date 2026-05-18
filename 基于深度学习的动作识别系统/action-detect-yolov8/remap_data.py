#coding:utf-8
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np

OLD_TO_NEW = {
    0: 6,
    1: 4,
    2: 1,
    3: 0,
    4: 3,
    5: 5,
}

ACTION_NAMES = {0: '行走', 1: '跑步', 2: '坐下', 3: '站立', 4: '举手', 5: '摔倒', 6: '弯腰'}


def remap(npz_path, output_path):
    print("加载已有数据...")
    data = np.load(npz_path)
    old_sequences = data['sequences']
    old_labels = data['labels']

    print(f"原始数据: {len(old_sequences)} 个序列")
    unique, counts = np.unique(old_labels, return_counts=True)
    print("原始类别分布:")
    OLD_NAMES = {0: '屈身', 1: '挥手', 2: '跑步', 3: '行走', 4: '站立', 5: '跌倒'}
    for u, c in zip(unique, counts):
        print(f"  {OLD_NAMES.get(int(u), int(u))}: {c}")

    new_sequences = []
    new_labels = []

    for i in range(len(old_sequences)):
        old_label = int(old_labels[i])
        if old_label in OLD_TO_NEW:
            new_sequences.append(old_sequences[i])
            new_labels.append(OLD_TO_NEW[old_label])

    new_sequences = np.array(new_sequences)
    new_labels = np.array(new_labels)

    print(f"\n重映射后: {len(new_sequences)} 个序列")
    unique, counts = np.unique(new_labels, return_counts=True)
    print("新类别分布:")
    for u, c in zip(unique, counts):
        print(f"  {ACTION_NAMES[u]}: {c} 个序列")

    np.savez(output_path, sequences=new_sequences, labels=new_labels)
    print(f"\n数据已保存到 {output_path}")


if __name__ == '__main__':
    remap('dataset/action_data.npz', 'dataset/action_data.npz')
