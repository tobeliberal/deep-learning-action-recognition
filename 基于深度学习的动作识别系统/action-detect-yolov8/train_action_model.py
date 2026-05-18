#coding:utf-8
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import Config
from action_recognizer import ActionLSTM
import argparse


class ActionDataset(Dataset):

    def __init__(self, sequences, labels, augment=False):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.augment:
            seq = self._augment(seq)

        return seq, label

    def _augment(self, seq):
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(seq) * 0.02
            seq = seq + noise

        if torch.rand(1).item() > 0.7:
            scale = 0.9 + torch.rand(1).item() * 0.2
            seq = seq * scale

        if torch.rand(1).item() > 0.8:
            shift = (torch.rand(1).item() - 0.5) * 0.05
            seq = seq + shift

        if torch.rand(1).item() > 0.7:
            start = torch.randint(0, 3, (1,)).item()
            if start > 0:
                seq = torch.cat([seq[start:], seq[:start]], dim=0)

        return seq


def compute_class_weights(labels, num_classes):
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = np.ones(num_classes, dtype=float)
    for u, c in zip(unique, counts):
        weights[u] = total / (num_classes * c)
    return torch.FloatTensor(weights)


def train(data_path, epochs=150, batch_size=64, lr=0.001, hidden_size=256,
          num_layers=3, dropout=0.4, save_path=None):
    if save_path is None:
        save_path = Config.action_model_path

    data = np.load(data_path)
    sequences = data['sequences']
    labels = data['labels']

    print(f"训练数据: {len(sequences)} 个序列")
    print(f"序列形状: {sequences.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print("类别分布:")
    for u, c in zip(unique, counts):
        print(f"  {Config.action_names.get(int(u), u)}: {c}")

    class_weights = compute_class_weights(labels, Config.num_actions)
    print(f"\n类别权重: {class_weights.tolist()}")

    indices = np.arange(len(sequences))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.85 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = ActionDataset(sequences[train_idx], labels[train_idx], augment=True)
    val_dataset = ActionDataset(sequences[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")

    model = ActionLSTM(
        input_size=Config.num_keypoints * 2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=Config.num_actions,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2)

    best_val_acc = 0
    best_val_f1 = 0
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_seq, batch_label in train_loader:
            batch_seq = batch_seq.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            output = model(batch_seq)
            loss = criterion(output, batch_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += batch_label.size(0)
            train_correct += (predicted == batch_label).sum().item()

        scheduler.step()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_seq, batch_label in val_loader:
                batch_seq = batch_seq.to(device)
                batch_label = batch_label.to(device)

                output = model(batch_seq)
                loss = criterion(output, batch_label)

                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                val_total += batch_label.size(0)
                val_correct += (predicted == batch_label).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_label.cpu().numpy())

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_f1 = compute_macro_f1(all_labels, all_preds, Config.num_actions)

        if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  模型已保存 (Val Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n早停: {patience} 轮无改善")
                break

    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"最佳验证F1: {best_val_f1:.4f}")
    print(f"模型保存路径: {save_path}")

    print("\n各类别准确率:")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    per_class_acc(model, val_loader, device)


def compute_macro_f1(y_true, y_pred, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1s.append(f1)
    return np.mean(f1s)


def per_class_acc(model, val_loader, device):
    correct = {}
    total = {}
    with torch.no_grad():
        for batch_seq, batch_label in val_loader:
            batch_seq = batch_seq.to(device)
            batch_label = batch_label.to(device)
            output = model(batch_seq)
            _, predicted = torch.max(output, 1)
            for i in range(len(batch_label)):
                label = batch_label[i].item()
                pred = predicted[i].item()
                if label not in total:
                    total[label] = 0
                    correct[label] = 0
                total[label] += 1
                if label == pred:
                    correct[label] += 1

    for c in sorted(total.keys()):
        acc = correct[c] / max(total[c], 1)
        print(f"  {Config.action_names.get(c, c)}: {acc:.4f} ({correct[c]}/{total[c]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM行为识别模型训练')
    parser.add_argument('--data', type=str, default='dataset/action_data.npz')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--save', type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"训练数据 {args.data} 不存在")
        exit(1)

    train(args.data, args.epochs, args.batch_size, args.lr,
          args.hidden_size, args.num_layers, args.dropout, args.save)
