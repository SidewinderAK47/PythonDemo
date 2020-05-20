import multiprocessing as mp
import time
from multiprocessing import freeze_support

import torch.nn as nn
import torch
from torch import optim
from tqdm import tqdm

from dataset import KnowledgeGraph
from transe_model import TransE, MarginLoss


class TransEDemo:
    def __init__(self, kg, args):
        self.n_generator = 4
        self.n_rank_calculator = 4
        self.kg = kg
        self.args = args
        self.batch_size = 100
        self.learning_rate = 1
        self.use_gpu = True
        if not torch.cuda.is_available():
            self.use_gpu = False
        self.model = TransE(kg.n_entity, kg.n_relation, dim=200)

        self.criterion = MarginLoss(margin=5)
        if self.use_gpu:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # print("我被创建了！！！")

    def to_var(self, x):
        if self.use_gpu:
            return torch.LongTensor(x).cuda()
        else:
            return torch.LongTensor(x)

    def launch_training(self, epoch): #(self, neighbor, epoch):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        # 先开启raw_batch_queue队列处理进程；
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()   #"n_triple": None
        # print('-----Start training-----')

        # start = time.time()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            # print(len(raw_batch))
            raw_batch_queue.put(raw_batch)  # 每一个batch_size为100个（h,r,t）放入进程队列
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)       # 放入n_generator个None让n_generator个进程运行结束;
        # print('-----Constructing training batches-----')

        epoch_loss = 0
        n_used_triple = 0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            # print(batch_pos)
            # print(len(batch_neg))
            pos_h = self.to_var([x[0] for x in batch_pos])
            pos_r = self.to_var([x[2] for x in batch_pos])
            pos_t = self.to_var([x[1] for x in batch_pos])
            neg_h = self.to_var([x[0] for x in batch_neg])
            neg_r = self.to_var([x[2] for x in batch_neg])
            neg_t = self.to_var([x[1] for x in batch_neg])
            #
            p_score, n_score = self.model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            batch_loss = self.criterion(p_score, n_score)
            #
            self.optimizer.zero_grad()
            batch_loss.backward()
            epoch_loss += batch_loss
            n_used_triple += len(batch_pos)
        # print('Epoch {}, epoch loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss.item(), time.time() - start))
        return epoch_loss.item()

    def launch_evaluation(self):
        print('-----Start evaluation-----')
        start = time.time()
        eval_result_queue = mp.JoinableQueue()   # 使用了 JoinableQueue
        rank_result_queue = mp.Queue()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        # 遍历所有的 test 三元组
        for eval_triple in self.kg.test_triples:
            # 调用训练
            idx_head_prediction, idx_tail_prediction = 0, 0
            # self.session.run(fetches=[self.idx_head_prediction, self.idx_tail_prediction],
            #                                                             feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
        # 在eval队列中放置n_rank_calculator结束标志；
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()  # 阻塞进程，等待队列为空的时候结束
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_mr_raw = 0
        head_mrr_raw = 0
        head_hits1_raw = 0
        head_hits3_raw = 0
        head_hits5_raw = 0
        head_hits10_raw = 0
        tail_mr_raw = 0
        tail_mrr_raw = 0
        tail_hits1_raw = 0
        tail_hits3_raw = 0
        tail_hits5_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_mr_filter = 0
        head_mrr_filter = 0
        head_hits1_filter = 0
        head_hits3_filter = 0
        head_hits5_filter = 0
        head_hits10_filter = 0
        tail_mr_filter = 0
        tail_mrr_filter = 0
        tail_hits1_filter = 0
        tail_hits3_filter = 0
        tail_hits5_filter = 0
        tail_hits10_filter = 0
        # 在这个进程中将队列中计算好的 结果都提取出来;
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_mr_raw += (head_rank_raw + 1)
            head_mrr_raw += 1 / (head_rank_raw + 1)
            if head_rank_raw < 1:
                head_hits1_raw += 1
            if head_rank_raw < 3:
                head_hits3_raw += 1
            if head_rank_raw < 5:
                head_hits5_raw += 1
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_mr_raw += (tail_rank_raw + 1)
            tail_mrr_raw += 1 / (tail_rank_raw + 1)
            if tail_rank_raw < 1:
                tail_hits1_raw += 1
            if tail_rank_raw < 3:
                tail_hits3_raw += 1
            if tail_rank_raw < 5:
                tail_hits5_raw += 1
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_mr_filter += (head_rank_filter + 1)
            head_mrr_filter += 1 / (head_rank_filter + 1)
            if head_rank_filter < 1:
                head_hits1_filter += 1
            if head_rank_filter < 3:
                head_hits3_filter += 1
            if head_rank_filter < 5:
                head_hits5_filter += 1
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_mr_filter += (tail_rank_filter + 1)
            tail_mrr_filter += 1 / (tail_rank_filter + 1)
            if tail_rank_filter < 1:
                tail_hits1_filter += 1
            if tail_rank_filter < 3:
                tail_hits3_filter += 1
            if tail_rank_filter < 5:
                tail_hits5_filter += 1
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_mr_raw /= n_used_eval_triple
        head_mrr_raw /= n_used_eval_triple
        head_hits1_raw /= n_used_eval_triple
        head_hits3_raw /= n_used_eval_triple
        head_hits5_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_mr_raw /= n_used_eval_triple
        tail_mrr_raw /= n_used_eval_triple
        tail_hits1_raw /= n_used_eval_triple
        tail_hits3_raw /= n_used_eval_triple
        tail_hits5_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            head_mr_raw, head_mrr_raw, head_hits1_raw, head_hits3_raw, head_hits5_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            tail_mr_raw, tail_mrr_raw, tail_hits1_raw, tail_hits3_raw, tail_hits5_raw, tail_hits10_raw))
        print('------Average------')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            (head_mr_raw + tail_mr_raw) / 2,
            (head_mrr_raw + tail_mrr_raw) / 2,
            (head_hits1_raw + tail_hits1_raw) / 2,
            (head_hits3_raw + tail_hits3_raw) / 2,
            (head_hits5_raw + tail_hits5_raw) / 2,
            (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_mr_filter /= n_used_eval_triple
        head_mrr_filter /= n_used_eval_triple
        head_hits1_filter /= n_used_eval_triple
        head_hits3_filter /= n_used_eval_triple
        head_hits5_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_mr_filter /= n_used_eval_triple
        tail_mrr_filter /= n_used_eval_triple
        tail_hits1_filter /= n_used_eval_triple
        tail_hits3_filter /= n_used_eval_triple
        tail_hits5_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            head_mr_filter, head_mrr_filter, head_hits1_filter, head_hits3_filter, head_hits5_filter,
            head_hits10_filter))
        print('-----Tail prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            tail_mr_filter, tail_mrr_filter, tail_hits1_filter, tail_hits3_filter, tail_hits5_filter,
            tail_hits10_filter))
        print('-----Average-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            (head_mr_filter + tail_mr_filter) / 2,
            (head_mrr_filter + tail_mrr_filter) / 2,
            (head_hits1_filter + tail_hits1_filter) / 2,
            (head_hits3_filter + tail_hits3_filter) / 2,
            (head_hits5_filter + tail_hits5_filter) / 2,
            (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.4f}s'.format(time.time() - start))
        print('-----Finish evaluation-----')
        return (head_hits1_filter + tail_hits1_filter) / 2

    def calculate_rank(self, in_queue, out_queue):
        # 不停遍历从队列中取
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()


if __name__ == '__main__':
    freeze_support()
    args = None
    kg = KnowledgeGraph('../data/FB15k')
    demo = TransEDemo(kg, args)
    training_range = tqdm(range(1000))  # 进度条
    for epoch in training_range:
        start = time.time()
        epoch_loss = demo.launch_training(epoch+1)
        training_range.set_description('Epoch {}, epoch loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss,
                                                                                        time.time() - start))

    demo.model.save_checkpoint('checkpoint/model_params.pkl')
    # TransEDemo(KnowledgeGraph('../data/FB15k'), args).launch_training(1000)
        # .launch_evaluation()
