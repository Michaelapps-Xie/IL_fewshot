import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEnhancementModule(nn.Module):
    def __init__(self, input_dim, num_samples, num_enhancements):
        super(FeatureEnhancementModule, self).__init__()
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.num_enhancements = num_enhancements

        # 初始化狄利克雷分布的超参数
        self.alpha = nn.Parameter(torch.ones(num_samples), requires_grad=False)

    def forward(self, features):
        """
        :param features: 原始样本特征，形状为 (batch_size, num_samples, input_dim)
        :return: 增强后的特征集合，形状为 (batch_size, num_samples + num_enhancements, input_dim)
        """
        batch_size = features.size(0)

        # 生成 N 个增强特征
        enhanced_features = []
        for _ in range(self.num_enhancements):
            # 从狄利克雷分布中抽样
            weight_vector = F.softmax(torch.multinomial(F.softmax(self.alpha), 1).float(), dim=-1)

            # 计算加权特征
            enhanced_feature = torch.sum(features * weight_vector.unsqueeze(-1), dim=1)
            enhanced_features.append(enhanced_feature.unsqueeze(1))

        # 将增强特征与原始特征结合
        enhanced_features = torch.cat(enhanced_features, dim=1)  # (batch_size, num_enhancements, input_dim)
        all_features = torch.cat((features, enhanced_features), dim=1)  # (batch_size, num_samples + num_enhancements, input_dim)

        return all_features

# 示例用法
if __name__ == "__main__":
    batch_size = 4
    num_samples = 3
    input_dim = 128
    num_enhancements = 5

    # 随机生成样本特征
    features = torch.randn(batch_size, num_samples, input_dim)

    # 实例化特征增强模块
    enhancement_module = FeatureEnhancementModule(input_dim, num_samples, num_enhancements)

    # 获取增强后的特征
    enhanced_features = enhancement_module(features)
    print(enhanced_features.shape)  # 输出形状应为 (batch_size, num_samples + num_enhancements, input_dim)
