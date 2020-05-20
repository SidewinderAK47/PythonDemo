import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from KGData import LoadTrainData
from TransE import TransE, TrainDataset
from tqdm import tqdm
from datetime import datetime
import time

print(torch.cuda.is_available())




class TrainerDemo:

    def __init__(self):
        # 模型参数
        self.dim = 200
        self.train_times = 1000
        self.margin = 5
        self.norm = 2
        self.learning_rate = 0.001
        self.batch_size = 100
        self.bern = True
        self.neg = 25
        self.use_gpu = True

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

        self.left_entity = {}
        self.right_entity = {}

        self.headRelation2Tail = {}
        self.tailRelation2Head = {}

        self.left_num = {}
        self.right_num = {}
        self.data = {}
        self.model = None
        self.optimizer = None
        self.criterion = None

        self.train()

    def train(self):
        trainData = LoadTrainData(self.entity2id, self.id2entity, self.relation2id, self.id2relation, self.train_triples
                                  , self.valid_triples, self.test_triples, self.headRelation2Tail, self.tailRelation2Head,
                                  self.left_entity, self.right_entity, self.left_num, self.right_num)

        self.entityTotal, self.relationTotal, self.trainTotal, self.validTotal, self.testTotal = trainData.get_total()

        self.model = TransE(self.entityTotal, self.relationTotal, dim=100, batch_size=self.batch_size)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.criterion = nn.MarginRankingLoss(margin=5.0)
        self.margin = torch.Tensor([self.margin])
        self.margin.requires_grad = False
        if self.use_gpu:
            self.model = self.model.cuda()
            self.margin = self.margin.cuda()

        prob = 500
        index_loader = DataLoader(dataset=TrainDataset(self.trainTotal), batch_size=self.batch_size, shuffle=True)
        training_range = tqdm(range(self.train_times))  # 进度条
        for epoch in training_range:                    # 一个epoch 花费时间51.15秒
            running_loss = 0.0

            for batch in index_loader:
                # start = time.time()
                # print(len(batch)*26)
                self.data['h'] = [0] * self.batch_size * (1 + self.neg)
                self.data['r'] = [0] * self.batch_size * (1 + self.neg)
                self.data['t'] = [0] * self.batch_size * (1 + self.neg)
                self.data['y'] = [0] * self.batch_size * (1 + self.neg)
                # 获取每个batch数据
                i = 0

                for index in batch:
                    # print("----------")
                    # print(index)
                    # print("----------")
                    # print(type(index))
                    # 收集正样本
                    head = self.train_triples[index][0]
                    rel = self.train_triples[index][1]
                    tail = self.train_triples[index][2]
                    self.data['h'][i] = head
                    self.data['r'][i] = rel
                    self.data['t'][i] = tail
                    self.data['y'][i] = 1
                    # print(self.data['h'][i], self.data['r'][i], self.data['t'][i], self.data['y'][i])
                    last = self.batch_size

                    for neg in range(self.neg):
                        self.data['h'][last + i] = head
                        self.data['r'][last + i] = rel
                        self.data['t'][last + i] = tail
                        self.data['y'][last + i] = -1

                        if self.bern:
                            prob = 1000 * self.left_num[rel] / (self.left_num[rel] + self.right_num[rel])
                        rmd = random.random() * 1000
                        # print("rmd:", rmd, "prob:", prob)
                        if rmd < prob:
                            while True:
                                corrupt_head = random.randint(0, self.entityTotal-1)
                                if corrupt_head not in self.left_entity[rel]:
                                    self.data['h'][last + i] = corrupt_head
                                    break
                        else:
                            while True:
                                corrupt_tail = random.randint(0, self.entityTotal-1)
                                if corrupt_tail not in self.right_entity[rel]:
                                    self.data['t'][last + i] = corrupt_tail
                                    break
                        # print(self.data['h'][i + last], self.data['r'][i + last], self.data['t'][i + last],
                        #       self.data['y'][i + last])
                        last += self.batch_size
                    # print("---------------------")
                    i += 1

                # 获取完毕batch数据

                # 中间写上代码块

                # print(self.data['h'])
                # print(self.data['r'])
                # print(self.data['t'])
                # print(self.data['y'])
                # 转变成tensor
                for key in self.data:
                    self.data[key] = self.to_var(self.data[key])

                p_score, n_score = self.model(self.data)
                # print(p_score.size())
                # print(n_score.size())
                loss = (torch.max(p_score - n_score, - self.margin)).mean() + self.margin
                running_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # end = time.time()
                # print('Running time: %s Seconds' % (end - start))
                # 处理之后
            training_range.set_description("Epoch %d | loss: %f" % (epoch, loss))  # 设置当前阶段的输出

        cur_time = datetime.now().strftime('%Y-%m-%d')
        self.model.save_checkpoint('.', 'model_params'+cur_time+'.pkl')

    def to_var(self, x):
        if self.use_gpu:
            return torch.Tensor(x).long().cuda()
        else:
            return torch.Tensor(x).long()





TrainerDemo()
