import os
import argparse

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from models.IL_fewshot import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser


@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = build_model(args).to(device)
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    model.load_state_dict(state_dict)
    model.eval()

    for split in ['test']:
        vis_dir = os.path.join(args.vis_dir, split)
        os.makedirs(vis_dir, exist_ok=True)

        dataset = FSC147Dataset(
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers,
            shuffle=False
        )

        ae, se = 0.0, 0.0

        for i, batch in enumerate(loader):
            if len(batch) == 4:
                img, bboxes, density_map, img_names = batch
            else:
                raise ValueError("Expected 4 items in batch (img, bboxes, density_map, img_name).")

            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)

            out, _ = model(img, bboxes)

            gt_count = density_map.flatten(1).sum(dim=1)
            pred_count = out.flatten(1).sum(dim=1)

            ae += torch.abs(gt_count - pred_count).sum().item()
            se += ((gt_count - pred_count) ** 2).sum().item()

            for j in range(img.shape[0]):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                # 原图
                image_np = TF.to_pil_image(img[j].cpu().clamp(0, 1))
                axes[0].imshow(image_np)
                axes[0].set_title('Input Image')
                axes[0].axis('off')

                # GT 密度图
                gt_density = density_map[j].squeeze().cpu()
                gt_cnt = gt_density.sum().item()
                im1 = axes[1].imshow(gt_density, cmap='jet')
                axes[1].set_title(f'GT Density Map\nCount: {gt_cnt:.1f}')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                # 预测密度图
                pred_density = out[j].squeeze().detach().cpu()
                pred_cnt = pred_density.sum().item()
                im2 = axes[2].imshow(pred_density, cmap='jet')
                axes[2].set_title(f'Predicted Density Map\nCount: {pred_cnt:.1f}')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                plt.tight_layout()

                base_name = os.path.splitext(os.path.basename(img_names[j]))[0]
                save_path = os.path.join(vis_dir, f'{base_name}.png')
                plt.savefig(save_path)
                plt.close()

        print(
            f"{split.capitalize()} set:",
            f"MAE: {ae / len(dataset):.2f}",
            f"RMSE: {(se / len(dataset)) ** 0.5:.2f}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('IL_fewshot', parents=[get_argparser()])
    parser.add_argument('--vis_dir', default='test', help='Directory to save visualizations')
    args = parser.parse_args()

    evaluate(args)

