from .backbone import Backbone
from .new_trans import TransformerEncoder
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

# 假设 GAFS 和 GCAM 已经在同一文件夹下


class IL_fewshot(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_ope_iterative_steps: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        backbone_name: str,
        swav_backbone: bool,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
        nms_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3
    ):
        super(IL_fewshot, self).__init__()

        # 保存参数
        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.nms_threshold = nms_threshold
        self.nms_iou_threshold = nms_iou_threshold

        # 初始化backbone模型
        self.backbone = Backbone(
            backbone_name, pretrained=True, dilation=False, reduction=reduction,
            swav=swav_backbone, requires_grad=train_backbone
        )

        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, emb_dim, kernel_size=1
        )

        # 添加 GAFS 和 GCAM 模块



        if num_encoder_layers > 0:
            self.encoder = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )

        self.ope = OPEModule(
            num_ope_iterative_steps, emb_dim, kernel_dim, num_objects, num_heads,
            reduction, layer_norm_eps, mlp_factor, norm_first, activation, norm, zero_shot
        )

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
            for _ in range(num_ope_iterative_steps*2 - 1)
        ])

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    def apply_nms(self, dmap):
        bs, c, h, w = dmap.size()
        output_dmaps = []

        for b in range(bs):
            heatmap = dmap[b].view(-1)
            mask = heatmap > self.nms_threshold
            if mask.sum() == 0:
                output_dmaps.append(dmap[b].unsqueeze(0))
                continue

            pos = torch.nonzero(mask).squeeze(1)
            scores = heatmap[pos]
            boxes = self.convert_to_boxes(pos, h, w)

            keep = nms(boxes, scores, self.nms_iou_threshold)
            filtered_dmap = torch.zeros_like(dmap[b])
            filtered_dmap.view(-1)[pos[keep]] = heatmap[pos[keep]]

            output_dmaps.append(filtered_dmap.unsqueeze(0))

        return torch.cat(output_dmaps, dim=0)

    def convert_to_boxes(self, pos, h, w):
        y_coords = pos // w
        x_coords = pos % w
        boxes = torch.stack([x_coords, y_coords, x_coords + 1, y_coords + 1], dim=1).float()
        return boxes

    def forward(self, x, bboxes):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects
        # print("this is input", x.shape)

        # gcam = GCAM(num_channels=3, hidden_dim=128, score_pooling='mean').to(device)
        # gafs = GAFS(num_channels=channels, hidden_dim=128).to(device) # 修改输入通道
        # gcam_features = gcam(x).to(device)  # 使用 GCAM 模块
        backbone_features, channels = self.backbone(x)
        # print("this is backbone shape",backbone_features.shape)

        # 使用 GAFS 模块处理特征
        # processed_features = gafs(gcam_features).to(device)  # 使用 GAFS 模块
        src = self.input_proj(backbone_features)

        bs, c, h, w = src.size()
        pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)
        src = src.flatten(2).permute(2, 0, 1)
        # print("this is src.shape",src.shape)

        if self.num_encoder_layers > 0:
            image_features = self.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)
        else:
            image_features = src

        f_e = image_features.permute(1, 2, 0).reshape(-1, self.emb_dim, h, w)

        # 原有的 OPEModule
        all_prototypes = self.ope(f_e, pos_emb, bboxes)  # 使用 OPEModule

        # 使用 GCAM 处理 OPEModule 的输出


        outputs = list()
        for i in range(all_prototypes.size(0)):
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]

            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)
            ).view(
                bs, num_objects, self.emb_dim, h, w
            ).max(dim=1)[0]

            if i == all_prototypes.size(0) - 1:
                predicted_dmaps = self.regression_head(response_maps)
            else:
                predicted_dmaps = self.aux_heads[i](response_maps)
            outputs.append(predicted_dmaps)

        return outputs[-1], outputs[:-1]


def build_model(args):
    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    return IL_fewshot(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
        num_ope_iterative_steps=args.num_ope_iterative_steps,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        swav_backbone=args.swav_backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        dropout=args.dropout,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=args.pre_norm,
        activation=nn.GELU,
        norm=True,
        nms_threshold=args.nms_threshold if hasattr(args, 'nms_threshold') else 0.5,
        nms_iou_threshold=args.nms_iou_threshold if hasattr(args, 'nms_iou_threshold') else 0.3
    )
