from models.IL_fewshot import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser
from utils.losses import ObjectNormalizedL2Loss

from time import perf_counter
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import random
import matplotlib.pyplot as plt

# 设置随机种子，保证实验的可重复性
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args).to(device)

    backbone_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {'params': non_backbone_params.values()},
            {'params': backbone_params.values(), 'lr': args.backbone_lr}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)

    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_val_ae']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best = float('inf')

    criterion = ObjectNormalizedL2Loss()

    train_dataset = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot
    )
    val_dataset = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers
    )

    # 记录各项指标
    epochs_list = []
    train_loss_list = []
    val_loss_list = []
    train_mae_list = []
    val_mae_list = []

    print("Starting training...")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        start = perf_counter()

        train_loss = 0.0
        val_loss = 0.0
        train_ae = 0.0
        val_ae = 0.0

        model.train()
        for img, bboxes, density_map in train_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)

            optimizer.zero_grad()
            out, aux_out = model(img, bboxes)
            # print("this is desity_map.shape",out.shape)
            num_objects = density_map.sum()

            main_loss = criterion(out, density_map, num_objects)
            aux_loss = sum([args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out])
            loss = main_loss + aux_loss
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss += main_loss.item() * img.size(0)
            train_ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum().item()

        model.eval()
        with torch.no_grad():
            for img, bboxes, density_map in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)
                out, aux_out = model(img, bboxes)

                main_loss = criterion(out, density_map, num_objects)
                aux_loss = sum([args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out])
                loss = main_loss + aux_loss

                val_loss += main_loss.item() * img.size(0)
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum().item()

        scheduler.step()

        train_loss_list.append(train_loss / len(train_loader.dataset))
        val_loss_list.append(val_loss / len(val_loader.dataset))
        train_mae_list.append(train_ae / len(train_loader.dataset))
        val_mae_list.append(val_ae / len(val_loader.dataset))
        epochs_list.append(epoch)

        print(
            f"Epoch: {epoch}",
            f"Train loss: {train_loss / len(train_loader.dataset):.3f}",
            f"Val loss: {val_loss / len(val_loader.dataset):.3f}",
            f"Train MAE: {train_ae / len(train_loader.dataset):.3f}",
            f"Val MAE: {val_ae / len(val_loader.dataset):.3f}",
            f"Epoch time: {perf_counter() - start:.3f} seconds",
        )

        if val_ae / len(val_loader.dataset) < best:
            best = val_ae / len(val_loader.dataset)
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_ae': best
            }
            torch.save(checkpoint, os.path.join(args.model_path, f'{args.model_name}.pt'))

    # 生成图表
    def plot_and_save(data, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(data, marker='o')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(os.path.join(args.model_path, filename))
        plt.close()

    plot_and_save(epochs_list, 'Learning Rate Over Epochs', 'Learning Rate', 'learning_rate.png')
    plot_and_save(train_loss_list, 'Train Loss Over Epochs', 'Train Loss', 'train_loss.png')
    plot_and_save(val_loss_list, 'Validation Loss Over Epochs', 'Validation Loss', 'val_loss.png')
    plot_and_save(train_mae_list, 'Train MAE Over Epochs', 'Train MAE', 'train_mae.png')
    plot_and_save(val_mae_list, 'Validation MAE Over Epochs', 'Validation MAE', 'val_mae.png')

    plt.figure(figsize=(10, 6))
    plt.plot(val_mae_list, marker='o')
    plt.axhline(y=best, color='r', linestyle='--', label='Best Val MAE')
    plt.title('Validation MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(args.model_path, 'val_mae_with_best.png'))
    plt.close()

# 主程序入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser('IL_fewshot', parents=[get_argparser()])
    args = parser.parse_args()
    train(args)
