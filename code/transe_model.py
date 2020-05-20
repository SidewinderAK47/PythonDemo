import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim):
        super(TransE, self).__init__()
        self.ent_embeddings = nn.Embedding(ent_tot, dim)
        self.rel_embeddings = nn.Embedding(rel_tot, dim)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)   # weight.data 取嵌入的向量表示    xavier_uniform_ 对前面的向量进行初始化定制 每层网络输入与输出相同
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def forward(self,  pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        # print(neg_h)
        pos_h = self.ent_embeddings(pos_h)
        pos_t = self.ent_embeddings(pos_t)
        pos_r = self.rel_embeddings(pos_r)
        neg_h = self.ent_embeddings(neg_h)
        neg_t = self.ent_embeddings(neg_t)
        neg_r = self.rel_embeddings(neg_r)
        pos_h = F.normalize(pos_h, 2, -1)  # normalize(input, p=2, dim=1, eps=1e-12, out=None) 将头实体，尾实体，关系，标准化 L2范数； 欧式单位向量
        pos_t = F.normalize(pos_t, 2, -1)
        pos_r = F.normalize(pos_r, 2, -1)
        neg_h = F.normalize(neg_h, 2, -1)
        neg_t = F.normalize(neg_t, 2, -1)
        neg_r = F.normalize(neg_r, 2, -1)
        p_score = self.get_score(pos_h, pos_r, pos_t)
        n_score = self.get_score(neg_h, neg_t, neg_r)
        return p_score, n_score

    def get_score(self, h, r, t):
        score = h + r - t
        score = torch.norm(score, 2, -1).flatten()  # .unsqueeze(-1)    #  .flatten()直接压成一维
        # torch.norm 对Tensor求范式 torch.norm(input, p=2) → float 对倒数第一维 的第p_norm范式 transE中p_norm为L1范式
        # 计算完范式之后，-1维没了，变成[[batch_seq]] (1,batch_seq)
        # flatten() 变成一维张量 默认按 最后一维，顺序   [batch_seq]
        return score

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
    def forward(self, p_score, n_score, batch_size):
        #
        # n_score=n_score.reshape([n_score.shape[0]//25,25])
        # p_score=p_score.unsqueeze(dim=-1)
        # res=p_score - n_score

        p_score = p_score.view(-1, batch_size).permute(1, 0)
        n_score = n_score.view(-1, batch_size).permute(1, 0)
        # 返回25个负例，与正例里面的最远
        return (torch.max(p_score-n_score, -self.margin)).mean() + self.margin # 0不好写，改成这种写法
        # 写的非常秒啊兄弟，不愧是清华大佬写的 秒啊，兄弟;  判断都不用了  ().mean()返回的是标量
        # torch.max(p_score - n_score, -self.margin) 将比-margin小的（正例和负例差要比margin远的），全部替换成-margin (是正确向量描述规则)
        # mean对所元素取算数平均值；
        # 为什么不先加margin再取均值？呢{理解，margin为常数，mean()外部和内部是一样的}
        # 也就是 小于-margin的被替换成了0，大于-margin 的
