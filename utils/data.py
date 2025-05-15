import os  # 导入操作系统接口模块
import json  # 导入JSON解析模块
import argparse  # 导入命令行参数解析模块

from PIL import Image  # 导入PIL库，用于图像处理
import numpy as np  # 导入Numpy库，支持多维数组操作
from scipy.ndimage import gaussian_filter  # 从SciPy库导入高斯滤波器，用于密度图处理

import torch  # 导入PyTorch，深度学习框架
from torch.utils.data import Dataset  # 从PyTorch导入Dataset，用于自定义数据集
from torchvision.ops import box_convert  # 从Torchvision导入box_convert，用于边界框转换
from torchvision import transforms as T  # 导入transforms模块，处理图像转换
from torchvision.transforms import functional as TVF  # 导入transform功能模块

from tqdm import tqdm  # 导入TQDM，用于显示进度条


def tiling_augmentation(img, bboxes, density_map, resize, jitter, tile_size, hflip_p):
    """
    图像数据增强，包括平铺、水平翻转、颜色抖动等操作。
    """

    # 应用水平翻转
    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    # 创建平铺图像
    def make_tile(x, num_tiles, hflip, hflip_p, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                if hflip[j, k] < hflip_p:
                    t = TVF.hflip(t)
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))
    # 是否进行水平翻转
    hflip = torch.rand(num_tiles, num_tiles)

    # 平铺图像
    img = make_tile(img, num_tiles, hflip, hflip_p, jitter)
    img = resize(img[..., :int(y_tile*y_target), :int(x_tile*x_target)])

    # 平铺密度图
    density_map = make_tile(density_map, num_tiles, hflip, hflip_p)
    density_map = density_map[..., :int(y_tile*y_target), :int(x_tile*x_target)]
    original_sum = density_map.sum()
    density_map = resize(density_map)
    density_map = density_map / density_map.sum() * original_sum

    # 翻转边界框
    if hflip[0, 0] < hflip_p:
        bboxes[:, [0, 2]] = x_target - bboxes[:, [2, 0]]
    bboxes = bboxes / torch.tensor([x_tile, y_tile, x_tile, y_tile])
    return img, bboxes, density_map


class FSC147Dataset(Dataset):
    """
    FSC147数据集类，用于加载和处理数据。
    """

    def __init__(
        self, data_path, img_size, split='train', num_objects=3,
        tiling_p=0.5, zero_shot=False
    ):
        self.split = split  # 数据集的划分(train/val/test)
        self.data_path = data_path  # 数据集路径
        self.horizontal_flip_p = 0.5  # 水平翻转概率
        self.tiling_p = tiling_p  # 平铺增强概率
        self.img_size = img_size  # 图像大小
        self.resize = T.Resize((img_size, img_size))  # 图像缩放大小
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)  # 随机颜色抖动
        self.num_objects = num_objects  # 边界框中的目标数量
        self.zero_shot = zero_shot  # 是否为零样本任务
        with open(
            os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)  # 加载数据集的划分文件
            self.image_names = splits[split]  # 根据划分获取相应的图片名称
        with open(
            os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)  # 加载图像的注释文件

    def __getitem__(self, idx: int):
        """
        获取数据集中的图像和相应的注释信息（边界框和密度图）
        """
        # 加载图像
        img = Image.open(os.path.join(
            self.data_path,
            'images_384_VarV2',
            self.image_names[idx]
        )).convert("RGB")
        w, h = img.size
        if self.split != 'train':
            img = T.Compose([
                T.ToTensor(),
                self.resize,
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                self.resize,
            ])(img)

        # 获取边界框
        bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size

        # 加载密度图
        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            'gt_density_map_adaptive_512_512_object_VarV2',
            os.path.splitext(self.image_names[idx])[0] + '.npy',
        ))).unsqueeze(0)

        # 如果图像大小不为512，则重新调整密度图大小
        if self.img_size != 512:
            original_sum = density_map.sum()
            density_map = self.resize(density_map)
            density_map = density_map / density_map.sum() * original_sum

        # 数据增强
        tiled = False
        if self.split == 'train' and torch.rand(1) < self.tiling_p:
            tiled = True
            tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
            img, bboxes, density_map = tiling_augmentation(
                img, bboxes, density_map, self.resize,
                self.jitter, tile_size, self.horizontal_flip_p
            )

        if self.split == 'train':
            if not tiled:
                img = self.jitter(img)
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # 进行水平翻转
        if self.split == 'train' and not tiled and torch.rand(1) < self.horizontal_flip_p:
            img = TVF.hflip(img)
            density_map = TVF.hflip(density_map)
            bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]

        return img, bboxes, density_map,self.image_names[idx]

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.image_names)


def generate_density_maps(data_path, target_size=(512, 512)):
    """
    生成密度图并保存到指定路径。
    """

    density_map_path = os.path.join(
        data_path,
        f'gt_density_map_adaptive_{target_size[0]}_{target_size[1]}_object_VarV2'
    )
    if not os.path.isdir(density_map_path):
        os.makedirs(density_map_path)  # 如果目标文件夹不存在，则创建

    with open(
        os.path.join(data_path, 'annotation_FSC147_384.json'), 'rb'
    ) as file:
        annotations = json.load(file)  # 加载注释文件

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 使用GPU或CPU
    for i, (image_name, ann) in enumerate(tqdm(annotations.items())):  # 迭代每张图像
        _, h, w = T.ToTensor()(Image.open(os.path.join(
            data_path,
            'images_384_VarV2',
            image_name
        ))).size()
        h_ratio, w_ratio = target_size[0] / h, target_size[1] / w  # 计算缩放比率

        #
