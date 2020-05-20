import os

import torch
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np

from TransE import TransE, TrainDataset


class TransETest:

    def __init__(self):
        # 求的参数
        self.hit1 = 0
        self.hit3 = 0
        self.hit10 = 0
        self.mr = 0
        self.mmr = 0

        self.r_rank = 0
        self.r_filter_rank = 0
        self.l_rank = 0
        self.l_filter_rank = 0
        # self.l_s = 0
        # self.l_filter_s = 0
        # self.r_s = 0
        # self.r_filter_s = 0


        self.l1_tot = 0
        self.l3_tot = 0
        self.l10_tot = 0

        self.l1_filter_tot = 0
        self.l3_filter_tot = 0
        self.l10_filter_tot = 0

        self.r1_tot = 0
        self.r3_tot = 0
        self.r10_tot = 0

        self.r1_filter_tot = 0
        self.r3_filter_tot = 0
        self.r10_filter_tot = 0


        # 配置参数
        self.use_gpu = False
        # 全局数据存储
        self.entity2id = {}
        self.id2entity = {}

        self.relation2id = {}
        self.id2relation = {}

        self.entityTotal = 0
        self.relationTotal = 0

        self.train_triples = []
        self.trainTotal = 0

        self.valid_triples = []
        self.validTotal = 0

        self.testTotal = 0
        self.test_triples = []
        # self.ok = {}
        # 快速查询三元组字典
        self.hr2t = {}
        self.tr2h = {}
        self.hr2t_1 = None
        self.tr2h_1 = None
        self.model = None

    def prepare_test_data(self):
        self.read_entity()
        self.read_relation()
        # self.hr2t_1 = np.zeros((self.entityTotal, self.relationTotal), dtype=np.int)
        # self.tr2h_1 = np.zeros((self.entityTotal, self.relationTotal), dtype=np.int)
        #print(self.hr2t_1.shape)

        self.read_valid_triple()
        self.read_test_triple()
        self.read_train_triple()
        print("entity_num:", self.entityTotal)
        print("relation_num:", self.relationTotal)
        print("train_triple:", self.trainTotal)
        print("valid_triple:", self.validTotal)
        print("test_triple:", self.testTotal)

        self.model = TransE(self.entityTotal, self.relationTotal, dim=100)
        self.model.load_checkpoint(path='model_params.pkl')
        if self.use_gpu:
            self.model = self.model.cuda()

        # print(self.model.ent_embeddings.weight.data)

        self.test()





    def read_entity(self):
        with open(os.path.join("../data", 'FB15k', 'entity2id.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                entityStr, entityId = line.strip().split('\t')
                # print('name:', entityStr, 'id:', entityId)
                self.entity2id[entityStr] = int(entityId)
                self.id2entity[entityId] = entityStr

            self.entityTotal = len(self.entity2id)

    def read_relation(self):
        with open(os.path.join("../data", 'FB15k', 'relation2id.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                relationStr, relationId = line.strip().split('\t')
                # print('name:', relationStr, 'id:', relationId)
                self.relation2id[relationStr] = int(relationId)
                self.id2relation[relationId] = relationStr
            self.relationTotal = len(self.relation2id)

    def read_valid_triple(self):
        with open(os.path.join("../data", 'FB15k', 'valid.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                h, t, r = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.valid_triples.append((h, r, t))
                self.hr2t_add(h, r, t)
                self.tr2h_add(t, r, h)
                # self.hr2t_1[h][r][t] = 1
                # self.tr2h_1[t][r][h] = 1
            self.validTotal = len(self.valid_triples)

    def read_test_triple(self):
        with open(os.path.join("../data", 'FB15k', 'test.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                h, t, r = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.test_triples.append((h, r, t))
                self.hr2t_add(h, r, t)
                self.tr2h_add(t, r, h)
                # self.hr2t_1[h][r][t] = 1
                # self.tr2h_1[t][r][h] = 1
            self.testTotal = len(self.test_triples)

    def read_train_triple(self):
        with open(os.path.join("../data", 'FB15k', 'train.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                h, t, r = line.strip().split('\t')
                # print(h, r, t)
                h, t, r = self.entity2id[h], self.entity2id[t], self.relation2id[r]
                self.train_triples.append((h, r, t))
                # self.ok_add(h, r, t)  # 加入正确的三元组
                self.hr2t_add(h, r, t)
                self.tr2h_add(t, r, h)
                # self.hr2t_1[h][r][t] = 1
                # self.tr2h_1[t][r][h] = 1
            self.trainTotal = len(self.train_triples)

    # def ok_add(self, h, r, t):
    #     if (h, r) not in self.ok:
    #         self.ok[(h, r)] = {}
    #     if t not in self.ok[(h, r)]:
    #         self.ok[(h, r)][t] = 1

    def hr2t_add(self, h, r, t):
        if (h, r) not in self.hr2t:
            self.hr2t[(h, r)] = {}
        if t not in self.hr2t[(h, r)]:
            self.hr2t[(h, r)][t] = 1

    def tr2h_add(self, t, r, h):
        if (t, r) not in self.tr2h:
            self.tr2h[(t, r)] = {}
        if h not in self.tr2h[(t, r)]:
            self.tr2h[(t, r)][h] = 1

    def test(self):
        # testIndex = DataLoader(dataset=TrainDataset(self.testTotal), batch_size=1, shuffle=False)
        test_range = tqdm(range(2000, 4000))  # 进度条
        for index in test_range:
            # start = datetime.now()
            h, r, t = self.test_triples[index]
            # print(h, r, t)

            self.test_tail(h, r, t)
            # end = datetime.now()
            # print('test_each_tail:', end - start)
            # start = datetime.now()
            self.test_head(h, r, t)
            # end = datetime.now()
            # print('test_each_head:', end - start)

        self.l_rank /= self.testTotal
        self.r_rank /= self.testTotal
        self.l_filter_rank /= self.testTotal
        self.r_filter_rank /= self.testTotal

        self.l1_tot /= self.testTotal
        self.l3_tot /= self.testTotal
        self.l10_tot /= self.testTotal
        self.l1_filter_tot /= self.testTotal
        self.l3_filter_tot /= self.testTotal
        self.l10_filter_tot /= self.testTotal
        self.r1_tot /= self.testTotal
        self.r3_tot /= self.testTotal
        self.r10_tot /= self.testTotal
        self.r1_filter_tot /= self.testTotal
        self.r3_filter_tot /= self.testTotal
        self.r10_filter_tot /= self.testTotal
        print("metric: \t\t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n")

        print("l(raw): \t\t\t %f \t %f \t %f \t %f " % (self.l_rank, self.l10_tot, self.l3_tot, self.l1_tot))
        print("r(raw): \t\t\t %f \t %f \t %f \t %f " % (self.r_rank, self.r10_tot, self.r3_tot, self.r1_tot))
        print("averaged(raw):\t\t %f \t %f \t %f \t %f " % ((self.l_rank + self.r_rank) / 2,
              (self.l10_tot + self.r10_tot) / 2, (self.l3_tot + self.r3_tot) / 2, (self.l1_tot + self.r1_tot) / 2))


    # h,r ,t normal python 对象
    def test_head(self, h, r, t):
        tmp_h = torch.LongTensor([[i] for i in range(self.entityTotal)]) #
        tmp_r = (torch.LongTensor([r])).repeat(self.entityTotal, 1)
        tmp_t = (torch.LongTensor([t])).repeat(self.entityTotal, 1)
        if self.use_gpu:
            tmp_h, tmp_r, tmp_t = tmp_h.cuda(), tmp_r.cuda(), tmp_t.cuda()
        # r = self.model.rel_embeddings(r)
        # t = self.model.ent_embeddings(t)
        # print(r)
        # print(t)
        score = self.model.get_score({'h': tmp_h, 'r': tmp_r, 't': tmp_t})
        # print(score)
        minimal = score[h]
        # sorted_score = torch.sort(score)
        # print(sorted_score)
        score = score.cpu().data.numpy()
        tmp = self.tr2h[(t, r)]
        l_s = 0
        l_filter_s = 0
        for i, _score in enumerate(score):
            if i != h:
                # print(i, _score)
                if _score < minimal:
                    l_s += 1
                    if i not in tmp:
                        l_filter_s += 1
        # r_s在遇到测试三元组中的尾实体的时候，还会+1;因此r_s <10
        if l_s < 10:
            self.l10_tot += 1
        if l_filter_s < 10:
            self.l10_filter_tot += 1
        if l_s < 3:
            self.l3_tot += 1
        if l_filter_s < 3:
            self.l3_filter_tot += 1
        if l_s < 1:
            self.l1_tot += 1
        if l_filter_s < 1:
            self.l1_filter_tot += 1

        self.l_filter_rank += (1+l_filter_s)
        self.l_rank += (1+l_s)

    def test_tail(self, h, r, t):
        tmp_h = (torch.LongTensor([h])).repeat(self.entityTotal, 1)
        tmp_r = (torch.LongTensor([r])).repeat(self.entityTotal, 1)
        tmp_t = torch.LongTensor([[i] for i in range(self.entityTotal)])
        if self.use_gpu:
            tmp_h, tmp_r, tmp_t = tmp_h.cuda(), tmp_r.cuda(), tmp_t.cuda()
        # r = self.model.rel_embeddings(r)
        # t = self.model.ent_embeddings(t)
        # print(r)
        # print(t)

        score = self.model.get_score({'h': tmp_h, 'r': tmp_r, 't': tmp_t})


        # print(score)

        minimal = score[h]
        # sorted_score = torch.sort(score)
        # print(sorted_score)
        score = score.cpu().data.numpy()
        # start = datetime.now()
        r_s = 0
        r_filter_s = 0
        tmp = self.hr2t[(h, r)]
        for i, _score in enumerate(score):
            if i != t:
                # print(i, _score)
                if _score < minimal:
                    r_s += 1
                    if i not in tmp:
                        r_filter_s += 1
        # end = datetime.now()
        # print('test_each_tail:', end - start)
        # r_s在遇到测试三元组中的尾实体的时候，还会+1;因此r_s <10
        if r_s < 10:
            self.r10_tot += 1
        if r_filter_s < 10:
            self.r10_filter_tot += 1
        if r_s < 3:
            self.r3_tot += 1
        if r_filter_s < 3:
            self.r3_filter_tot += 1
        if r_s < 1:
            self.r1_tot += 1
        if r_filter_s < 1:
            self.r1_filter_tot += 1

        self.r_filter_rank += (1+r_filter_s)
        self.r_rank += (1+r_s)

    def test_rel(self):
        pass



test = TransETest()
test.prepare_test_data()


