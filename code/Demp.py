import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np

# BATCH_SIZE = 3
# x = torch.Tensor(10, 10)
# y = torch.linspace(10, 1, 10)
# z = torch.LongTensor([[1, 2 ,3]])
# # torch_dataset = Data.TensorDataset(x, y)
#
# Embedding = nn.Embedding(10, 10)
#
# embed = Embedding(z)
# print(embed)
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler

batch_triplet = np.random.random((5, 3, 1))
print(batch_triplet)
# np.random.permutation(batch_triplet.shape[0])
np.random.shuffle(batch_triplet, )
print("---------1----------")
print(batch_triplet)


np.random.shuffle(batch_triplet)
print("-------2------------")
print(batch_triplet)

np.random.shuffle(batch_triplet)
print("-------------------")
print(batch_triplet)

np.random.shuffle(batch_triplet)
print("-------------------")
print(batch_triplet)

print("++++++++")
print(batch_triplet[0])
batch_triplet[0][0] = 1.01
print(batch_triplet[0])
# import multiprocessing
#
# MSG_NUM=20
# END_MSG = "END_MSG"
# q = multiprocessing.Queue(50)

# t = list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
# print(t)
# t = list(BatchSampler(RandomSampler(range(10)), batch_size=3, drop_last=False))
# print(t)


import time
# define a processor
# class Consumer(multiprocessing.Process):
#     def run(self):
#         print("enter", self.name)
#         batch = list()
#         c = 0
#         while True:
#             print("wait for action")
#             action = q.get()
#             if (action==END_MSG):
#                 break
#             else:
#                 print(action)
#         print("exit", self.name)
#
#
# if __name__=="__main__":
#     p = Consumer()
#     p.start()
#     for i in range(MSG_NUM):
#         q.put(i)
#     q.put(END_MSG)
#     print("producer finished, join consumer")
#     p.join()
