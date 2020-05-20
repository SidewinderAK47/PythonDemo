import os

import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset


class TransE(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim=100, margin=5, batch_size=100):
        super(TransE, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot

        self.dim = dim
        self.margin = margin
        self.learning_rate = 0.001
        self.batch_size = batch_size

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)   # weight.data 取嵌入的向量表示    xavier_uniform_ 对前面的向量进行初始化定制 每层网络输入与输出相同
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def forward(self, data):
        score = self.get_score(data)
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)

        # 计算损失
        return p_score, n_score

    def get_score(self, data):
        batch_h = data['h']
        batch_r = data['r']
        batch_t = data['t']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        h = F.normalize(h, 2, -1)  # normalize(input, p=2, dim=1, eps=1e-12, out=None) 将头实体，尾实体，关系，标准化 L2范数； 欧式单位向量
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)
        score = h + r - t
        score = torch.norm(score, 2, -1).flatten()
        # torch.norm 对Tensor求范式 torch.norm(input, p=2) → float 对倒数第一维 的第p_norm范式 transE中p_norm为L1范式
        # 计算完范式之后，-1维没了，变成[[batch_seq]] (1,batch_seq)
        # flatten() 变成一维张量 默认按 最后一维，顺序   [batch_seq]
        return score


    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        # view将维度转变为[[batch_size]]   permute交换维度之后，变成了(batch_size,1) 为黄金三元组的分数
        return positive_score

    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:]
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)  # 负三元组的分数
        return negative_score

    # 加载保存的模型
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    # 存储模型到硬盘
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), os.path.join(path))




class MarginLoss(nn.Module):
    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))  # 固定Margin参数
        self.margin.requires_grad = False

    # 损失函数的输入
    def forward(self, p_score, n_score):
        return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
        # 写的非常秒啊兄弟，不愧是清华大佬写的 秒啊，兄弟;  判断都不用了  ().mean()返回的是标量
        # torch.max(p_score - n_score, -self.margin) 将比-margin小的（正例和负例差要比margin远的），全部替换成-margin (是正确向量描述规则)
        # mean对所元素取算数平均值；
        # 为什么不先加margin再取均值？呢{理解，margin为常数，mean()外部和内部是一样的}
        # 也就是 小于-margin的被替换成了0，大于-margin 的

# print(t)
# corrupt_head_prob = np.random.binomial(1, 0.5)
# print(corrupt_head_prob)


class TrainDataset(Dataset):

    def __init__(self, trainTotal):
        self.trainTripleList = range(trainTotal)
        self.trainTotal = trainTotal
        print(self.trainTotal)

    def __len__(self):
        return self.trainTotal

    def __getitem__(self, item):
        return self.trainTripleList[item]